"""
labels.py -- Phase 0 label construction for the Conviction Model
(docs/conviction_model/CONVICTION_MODEL_PLAN.md, Phase 0 and "Labels" section).

Builds the 6-output, per-horizon, CDI-relative conviction label: a risk-adjusted
excess return over CDI at each of 5 horizons (k=21,63,126,252,504 trading days)
plus one path-drawdown-severity measure. Deliberately NOT aggregated into a
single scalar -- every column stays independent all the way to the regressor
(see the plan's Labels section for why: a single aggregate collapses genuinely
different shapes of opportunity, e.g. a short-term-only move vs. a slow
multi-year compounder, into indistinguishable numbers).

I/O (load_prices_wide, load_cdi_daily_decimal) is kept separate from the pure
computation functions below it, so the computation can be unit-tested against
synthetic data without touching data/raw or data/processed (matches
src/h_series/tests's own convention of building synthetic prices_wide/bench
directly rather than reading real files in the fast test group).
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..h_series.spine import build_forward_targets
from .paths import CDI_PATH, DATASET_PATH

HORIZONS = (21, 63, 126, 252, 504)  # trading days: ~1/3/6/12/24 months
DRAWDOWN_HORIZON = 504              # the longest horizon -- see plan Labels, point 3
VOL_LOOKBACK_DAYS = 60              # trailing window for the risk-adjustment denominator


# --- I/O -----------------------------------------------------------------

def load_prices_wide() -> pd.DataFrame:
    """Wide ticker x trade_date panel of adj_close, read narrowly from
    ml_dataset.parquet -- the same shape and source h_series/spine.py's
    build_forward_targets expects. Mirrors h_series/features.py's
    _load_daily_prices, minus its BOVA11 bench -- this module uses CDI as
    the bench instead (load_cdi_daily_decimal, below), not BOVA11."""
    table = pq.read_table(DATASET_PATH, columns=["ticker", "trade_date", "adj_close"])
    prices = table.to_pandas()
    return prices.pivot(index="trade_date", columns="ticker", values="adj_close").sort_index()


def load_cdi_daily_decimal() -> pd.Series:
    """CDI as a daily decimal rate, indexed by reference_date.

    data/raw/macro/cdi.parquet stores `cdi` as a DAILY rate in PERCENT (e.g.
    0.0525 meaning 0.0525%/day, compounding to roughly 14% p.a.) -- confirmed
    by src/rl_agent/data.py::validate_cdi_daily_percent, which this function's
    /100 conversion matches exactly. This is deliberately NOT the /252
    convention src/build_dataset/features.py::compute_macro_features() uses
    for selic/ipca (dividing by 252 as if they were annual percentages) --
    selic's raw values are numerically identical in magnitude to cdi's
    (checked directly against data/raw/macro/selic.parquet), so that /252
    convention looks like a real, pre-existing unit mismatch elsewhere in the
    codebase, not something to copy here. Out of scope to fix in this module
    (compute_macro_features is a shared, foundational function many other
    things depend on) -- flagged so it isn't silently propagated into new code."""
    cdi = pd.read_parquet(CDI_PATH)[["reference_date", "cdi"]]
    cdi = cdi.sort_values("reference_date").set_index("reference_date")["cdi"]
    daily_decimal = cdi / 100.0
    assert (daily_decimal >= 0).all(), "negative CDI rate -- check units before proceeding"
    return daily_decimal


# --- pure computation ------------------------------------------------------

def build_cdi_cumulative_index(cdi_daily_decimal: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Cumulative CDI total-return index (starts at 1.0), reindexed onto
    `calendar` and forward/back-filled -- plays the same role as BOVA11's
    adj_close does as the `bench` argument to build_forward_targets, just
    compounded from a rate series instead of read directly as a price level."""
    daily = cdi_daily_decimal.reindex(calendar).ffill().bfill()
    return (1.0 + daily).cumprod()


def trailing_volatility(prices_wide: pd.DataFrame, window: int = VOL_LOOKBACK_DAYS) -> pd.DataFrame:
    """Trailing daily-log-return stdev per ticker, as of each calendar date --
    the risk-adjustment denominator (plan Labels, point 2). NaN until `window`
    days of a ticker's own history exist -- the same warm-up convention as
    every other rolling feature in this dataset (a leading prefix, not an
    interior hole).

    Masks non-positive adj_close to NaN before log -- mirrors
    build_dataset/features.py::compute_price_features's identical guard for
    the same vendor artifact (BolsAI's adj_close rounds to exactly 0.00 for a
    handful of deep-history microcaps once their cumulative split/dividend
    adjustment factor pushes the true price below its 2-decimal precision
    floor -- see adj_close_precision_degraded)."""
    log_ret = np.log(prices_wide.where(prices_wide > 0)).diff()
    return log_ret.rolling(window, min_periods=window).std()


def compute_risk_adjusted_excess_returns(prices_wide: pd.DataFrame, cdi_index: pd.Series,
                                          decision_dates: pd.DatetimeIndex, universe: pd.DataFrame,
                                          horizons=HORIZONS) -> pd.DataFrame:
    """One row per (decision_date, ticker) in `universe`, one column per
    horizon (`risk_adj_excess_return_k{k}`): the raw CDI-relative forward
    return (build_forward_targets, reused as-is) divided by trailing realized
    volatility as of decision_date. No aggregation across horizons -- columns
    stay separate all the way to the regressor (plan Labels, point 3)."""
    vol = trailing_volatility(prices_wide)
    out = universe.drop_duplicates(["decision_date", "ticker"]).reset_index(drop=True).copy()

    vol_at_decision = vol.reindex(index=pd.DatetimeIndex(decision_dates))
    vol_long = vol_at_decision.stack(future_stack=True).rename("trailing_vol").reset_index()
    vol_long.columns = ["decision_date", "ticker", "trailing_vol"]

    for k in horizons:
        raw = build_forward_targets(prices_wide, cdi_index, decision_dates, k, universe)
        raw = raw[["decision_date", "ticker", "fwd_rel_return"]]
        merged = out[["decision_date", "ticker"]].merge(raw, on=["decision_date", "ticker"], how="left")
        merged = merged.merge(vol_long, on=["decision_date", "ticker"], how="left")
        out[f"risk_adj_excess_return_k{k}"] = (
            merged["fwd_rel_return"].to_numpy() / merged["trailing_vol"].to_numpy()
        )

    return out


def compute_drawdown_severity(prices_wide: pd.DataFrame, cdi_index: pd.Series,
                               decision_dates: pd.DatetimeIndex, universe: pd.DataFrame,
                               k: int = DRAWDOWN_HORIZON) -> pd.DataFrame:
    """Max peak-to-trough decline in cumulative CDI-relative excess return,
    along the DAILY path from decision_date to decision_date+k trading days --
    not just the k-day endpoint return build_forward_targets computes. This is
    what distinguishes a position that goes up then round-trips from one that
    grinds up steadily to the same endpoint (plan Labels, point 3). Returns
    one row per (decision_date, ticker) in `universe`, column `drawdown_severity`."""
    calendar = prices_wide.index
    universe = universe.drop_duplicates(["decision_date", "ticker"])
    # Mask non-positive adj_close before log -- same vendor-rounding-artifact guard as
    # trailing_volatility() above (BolsAI's adj_close rounds to exactly 0.00 for a handful
    # of deep-history microcaps; see adj_close_precision_degraded). Without this, log(0)=-inf
    # poisons excess_path/running_max into +inf severities for those tickers.
    log_prices = np.log(prices_wide.where(prices_wide > 0))
    log_cdi = np.log(cdi_index)

    rows = []
    for decision_date, group in universe.groupby("decision_date"):
        start_pos = calendar.searchsorted(decision_date)
        end_pos = start_pos + k
        if start_pos >= len(calendar) or end_pos >= len(calendar):
            for ticker in group["ticker"]:
                rows.append((decision_date, ticker, np.nan))
            continue

        window = slice(start_pos, end_pos + 1)
        cdi_path = log_cdi.iloc[window].to_numpy()
        cdi_path = cdi_path - cdi_path[0]

        for ticker in group["ticker"]:
            if ticker not in log_prices.columns:
                rows.append((decision_date, ticker, np.nan))
                continue
            price_path = log_prices[ticker].iloc[window].to_numpy()
            if np.isnan(price_path[0]):
                rows.append((decision_date, ticker, np.nan))
                continue
            excess_path = (price_path - price_path[0]) - cdi_path
            running_max = np.fmax.accumulate(excess_path)  # fmax skips NaN pairwise
            drawdown = running_max - excess_path
            severity = float(np.nanmax(drawdown)) if not np.all(np.isnan(drawdown)) else np.nan
            rows.append((decision_date, ticker, severity))

    return pd.DataFrame(rows, columns=["decision_date", "ticker", "drawdown_severity"])


def build_conviction_labels(prices_wide: pd.DataFrame, cdi_index: pd.Series,
                             decision_dates: pd.DatetimeIndex, universe: pd.DataFrame,
                             horizons=HORIZONS, drawdown_horizon: int = DRAWDOWN_HORIZON) -> pd.DataFrame:
    """The full 6-output conviction label: one row per (decision_date, ticker)
    in `universe`, 5 risk-adjusted CDI-relative excess-return columns
    (one per horizon) + one drawdown_severity column. No aggregation into a
    single scalar anywhere in this pipeline (plan Labels, point 3) -- any
    single-number "conviction" is a downstream reduction computed later
    (reporting only), never trained as its own target.

    `cdi_index` is a caller-supplied argument, not loaded internally, so this
    function stays pure/synthetic-testable -- callers build it once via
    load_cdi_daily_decimal() + build_cdi_cumulative_index() and pass it in."""
    out = compute_risk_adjusted_excess_returns(prices_wide, cdi_index, decision_dates, universe, horizons)
    dd = compute_drawdown_severity(prices_wide, cdi_index, decision_dates, universe, drawdown_horizon)
    return out.merge(dd, on=["decision_date", "ticker"], how="left")
