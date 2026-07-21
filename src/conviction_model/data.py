"""
data.py -- Phase 1 (docs/conviction_model/CONVICTION_MODEL_PLAN.md): loads
per-ticker history from ml_dataset.parquet and builds the multi-resolution
window tensors encoder.py's 4 branches consume.

Normalization mirrors src/rl_agent/data.py::FEATURE_NORM's semantics (divide
price-level channels by their value-at-t, sign-preserving log1p squash for
heavy-tailed ratios, float = passthrough scale) but isn't imported directly --
this project's feature set (macro, fundamental z-scores) barely overlaps
rl_agent's, and rl_agent's window_tensor() masks across an ASSET-slot axis
(top50 membership turnover) that doesn't exist here (one ticker at a time).
"""

from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .paths import DATASET_PATH

DAILY_WINDOW = 60
WEEKLY_WINDOW = 104     # ~2y at weekly resolution
MONTHLY_WINDOW = 120    # 10y at monthly resolution
QUARTERLY_WINDOW = 40   # 10y at quarterly resolution (fundamentals branch)

PRICE_LEVEL_COLS = ("adj_close", "adj_high", "adj_low")
TECHNICAL_COLS = ("return_1m", "return_3m", "return_6m", "price_vs_ma60",
                   "volatility_ratio_20_60", "rsi_14", "drawdown", "volume_ratio_20d")
MACRO_COLS = ("excess_return", "real_return", "selic_trend_20d")
FUND_COLS = (
    "pl_zhist_5y", "pvp_zhist_5y", "roe_zhist_5y", "net_margin_zhist_5y",
    "ebitda_margin_zhist_5y", "debt_equity_zhist_5y", "net_debt_ebitda_zhist_5y",
    "earnings_yield_zhist_5y", "book_to_market_zhist_5y", "current_ratio_zhist_5y",
    "asset_turnover_zhist_5y", "cagr_earnings_5y_final", "cagr_revenue_5y_final",
    "cagr_earnings_defined", "cagr_revenue_defined", "n_quarters_available",
    "div_yield_12m", "div_count_12m", "has_dividends",
)

DAILY_FEATURES = PRICE_LEVEL_COLS + TECHNICAL_COLS
WEEKLY_FEATURES = PRICE_LEVEL_COLS + TECHNICAL_COLS
MONTHLY_FEATURES = PRICE_LEVEL_COLS + TECHNICAL_COLS + MACRO_COLS
QUARTERLY_FEATURES = FUND_COLS

# "self" -> divide by value-at-t (price levels only, eq. 18 convention).
# "log1p" -> sign(x)*log1p(|x|), for pl/pvp's and debt ratios' heavy tails.
# float -> nan_to_num(x) * scale (passthrough at 1.0).
FEATURE_NORM = {
    "adj_close": "self", "adj_high": "self", "adj_low": "self",
    "return_1m": 1.0, "return_3m": 1.0, "return_6m": 1.0,
    "price_vs_ma60": 1.0, "volatility_ratio_20_60": 1.0,
    "rsi_14": 0.01, "drawdown": 1.0, "volume_ratio_20d": 1.0,
    "excess_return": 1.0, "real_return": 1.0, "selic_trend_20d": 1.0,
    "pl_zhist_5y": "log1p", "pvp_zhist_5y": "log1p",
    "roe_zhist_5y": 1.0, "net_margin_zhist_5y": 1.0, "ebitda_margin_zhist_5y": 1.0,
    "debt_equity_zhist_5y": "log1p", "net_debt_ebitda_zhist_5y": "log1p",
    "earnings_yield_zhist_5y": 1.0, "book_to_market_zhist_5y": 1.0,
    "current_ratio_zhist_5y": 1.0, "asset_turnover_zhist_5y": 1.0,
    "cagr_earnings_5y_final": 1.0, "cagr_revenue_5y_final": 1.0,
    "cagr_earnings_defined": 1.0, "cagr_revenue_defined": 1.0,
    "n_quarters_available": 0.05, "div_yield_12m": 1.0, "div_count_12m": 0.25,
    "has_dividends": 1.0,
}


@lru_cache(maxsize=None)
def load_ticker_daily_frame(ticker: str) -> pd.DataFrame:
    """I/O: one ticker's full daily history, indexed by trade_date, columns =
    every column the daily/weekly/monthly branches need. Fundamentals load
    separately (load_ticker_quarterly_frame) -- daily rows repeat each
    quarter's forward-filled fundamentals, so pulling FUND_COLS here would be
    redundant with the dedup that function already does."""
    cols = list(dict.fromkeys(DAILY_FEATURES + WEEKLY_FEATURES + MONTHLY_FEATURES))
    table = pq.read_table(DATASET_PATH, columns=["ticker", "trade_date", *cols],
                           filters=[("ticker", "=", ticker)])
    df = table.to_pandas().sort_values("trade_date").set_index("trade_date")
    return df[cols]


@lru_cache(maxsize=None)
def load_ticker_quarterly_frame(ticker: str) -> pd.DataFrame:
    """I/O: one ticker's fundamentals-branch input, deduped from the daily
    forward-filled panel to one row per distinct fundamental snapshot (same
    dedup reasoning as build_dataset/features.py's history-relative
    z-scores -- rolling the daily-forward-filled panel directly would be
    ~65x redundant). A new row is kept exactly when a filing changes any of
    FUND_COLS."""
    table = pq.read_table(DATASET_PATH, columns=["ticker", "trade_date", *QUARTERLY_FEATURES],
                           filters=[("ticker", "=", ticker)])
    df = table.to_pandas().sort_values("trade_date").set_index("trade_date")[list(QUARTERLY_FEATURES)]
    changed = (df != df.shift()).any(axis=1)
    changed.iloc[0] = True
    return df[changed]


def resample_branch_frame(daily_frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Downsample a per-ticker daily frame to `rule` resolution ('W' weekly,
    'ME' month-end) by taking each period's last daily row, then ffill any
    resulting gap (a period with zero trading days) -- ffill-only, matching
    CLAUDE.md's convention for technical columns; price-level columns are
    already bfilled upstream in ml_dataset.parquet so ffill alone is enough
    here too."""
    return daily_frame.resample(rule).last().ffill()


def window_tensor(frame: pd.DataFrame, as_of: pd.Timestamp, window: int,
                   features: Sequence[str]) -> np.ndarray:
    """Last `window` rows of `frame` at/before `as_of`, per-channel-normalized
    per FEATURE_NORM, returned as a [len(features), window] float64 array.
    Both individual in-window NaNs (a technical's own warm-up landing inside
    an otherwise-full window) and a too-short history (left-pad) are handled
    the same way per channel: "self" pads/fills with 1.0 (neutral
    price-relative), everything else with 0.0 -- same fill-value convention
    as rl_agent/data.py::window_tensor, applied along the TIME axis here
    instead of the asset-slot axis (no asset axis inside one branch).

    A completely empty `frame` (wrong ticker, bad file) still raises -- that's
    a caller bug, not a warm-up. But zero rows *before as_of specifically* is
    legitimate and gets the full left-pad treatment like any other too-short
    window: a resampled weekly/monthly branch has no row yet for an as_of
    that falls before that ticker's first week/month has closed, even though
    the ticker's daily history already exists."""
    if frame.empty:
        raise ValueError("frame has no rows at all")
    hist = frame.loc[:as_of].iloc[-window:]
    n_have = len(hist)
    anchor = hist.iloc[-1] if n_have else None
    out = np.empty((len(features), window), dtype=np.float64)
    for i, feat in enumerate(features):
        norm = FEATURE_NORM[feat]
        raw = hist[feat].to_numpy(dtype=np.float64) if n_have else np.empty(0, dtype=np.float64)
        if norm == "self":
            if n_have:
                denom = float(anchor[feat])
                denom = denom if (denom == denom and denom != 0.0) else 1.0
                vals = np.where(np.isnan(raw), denom, raw) / denom
            else:
                vals = raw
            pad_value = 1.0
        elif norm == "log1p":
            filled = np.nan_to_num(raw, nan=0.0)
            vals = np.sign(filled) * np.log1p(np.abs(filled))
            pad_value = 0.0
        else:
            vals = np.nan_to_num(raw, nan=0.0) * float(norm)
            pad_value = 0.0
        if n_have < window:
            padded = np.full(window, pad_value, dtype=np.float64)
            if n_have:
                padded[-n_have:] = vals  # n_have==0 handled by the `if` -- padded[-0:] is the whole array, not empty
            vals = padded
        out[i] = vals
    return out


def branch_windows_from_frames(daily_frame: pd.DataFrame, quarterly_frame: pd.DataFrame,
                                as_of: pd.Timestamp) -> dict:
    """Pure core of branch_windows(): given one ticker's already-loaded daily
    and quarterly frames, build all 4 branches' window tensors at `as_of`.
    Resamples weekly/monthly on every call -- fine for a one-off lookup, but
    NOT the path to use in a training loop calling this hundreds of times per
    step for the same handful of tickers (see build_frame_cache /
    branch_windows_from_precomputed, which resample once and reuse)."""
    weekly_frame = resample_branch_frame(daily_frame, "W")
    monthly_frame = resample_branch_frame(daily_frame, "ME")
    return branch_windows_from_precomputed(daily_frame, weekly_frame, monthly_frame, quarterly_frame, as_of)


def branch_windows_from_precomputed(daily_frame: pd.DataFrame, weekly_frame: pd.DataFrame,
                                     monthly_frame: pd.DataFrame, quarterly_frame: pd.DataFrame,
                                     as_of: pd.Timestamp) -> dict:
    """Same as branch_windows_from_frames but takes already-resampled
    weekly/monthly frames -- the hot-path version for batch assembly, where
    resampling per (ticker, as_of) call instead of once per ticker was a
    measured ~150x batch-assembly slowdown in the Stage 1A pilot (every
    position in the batch re-resampled the same handful of tickers' full
    daily history from scratch)."""
    return {
        "daily": window_tensor(daily_frame, as_of, DAILY_WINDOW, DAILY_FEATURES),
        "weekly": window_tensor(weekly_frame, as_of, WEEKLY_WINDOW, WEEKLY_FEATURES),
        "monthly": window_tensor(monthly_frame, as_of, MONTHLY_WINDOW, MONTHLY_FEATURES),
        "fundamentals": window_tensor(quarterly_frame, as_of, QUARTERLY_WINDOW, QUARTERLY_FEATURES),
    }


def branch_windows(ticker: str, as_of: pd.Timestamp) -> dict:
    """I/O + pure core: one ticker's 4 branch window tensors at `as_of`.
    load_ticker_daily_frame/load_ticker_quarterly_frame are lru_cached, so
    repeated calls across many `as_of` dates for the same ticker (e.g. many
    CPC batches in one training run) only hit the parquet file once per
    ticker. ponytail: caches every ticker's full history in memory for the
    life of the process (~515 tickers, a few thousand rows each) -- revisit
    with an eviction policy if that becomes a real memory problem."""
    daily_frame = load_ticker_daily_frame(ticker)
    quarterly_frame = load_ticker_quarterly_frame(ticker)
    return branch_windows_from_frames(daily_frame, quarterly_frame, as_of)


def build_frame_cache(tickers: Sequence[str]) -> dict:
    """{ticker: (daily_frame, weekly_frame, monthly_frame, quarterly_frame)}
    for every ticker in `tickers`, via the lru_cached loaders -- weekly/monthly
    resampling depends only on the ticker (not as_of), so it's done ONCE here
    rather than per window lookup (branch_windows_from_precomputed is the
    matching hot-path consumer). A plain dict snapshot so batch-assembly code
    (ssl_pretrain.py) can index it directly per (ticker, as_of) without
    re-hitting the cache's lock/hash on every window lookup."""
    cache = {}
    for t in tickers:
        daily_frame = load_ticker_daily_frame(t)
        cache[t] = (daily_frame, resample_branch_frame(daily_frame, "W"),
                    resample_branch_frame(daily_frame, "ME"), load_ticker_quarterly_frame(t))
    return cache
