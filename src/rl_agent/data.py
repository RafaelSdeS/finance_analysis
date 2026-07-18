"""
data.py — price data pipeline for the EIIE agent (docs/EIIE_AGENT_PLAN.md
"Global asset indexing & PVM dynamic<->global mapping" and "Observation /
state" sections).

Two coordinate systems meet here:
  - global space: 172 columns = 171 union tickers (permanent alphabetical
    index, 1..171) + cash (index 0). This is where price levels, y_t, and
    (in environment.py) all cost/reward math live.
  - slot space: a fixed 50-wide window the EIIE network actually sees each
    day, filled with that day's active top-50 members sorted by their
    permanent global index.

Observation prices are read from the full ml_dataset.parquet (not the
pre-built ml_dataset_top50_universe.parquet) so an asset's n-day lookback
history is available even on its first day of investability -- the
pre-built file drops rows outside a ticker's membership periods and would
leave that history missing. Universe membership (investability) still comes
from the point-in-time top50_universe_membership.parquet, restricted to the
2011-2026 window chosen for its complete fundamentals coverage (unused by
this price-only iteration 1, but load-bearing for later ones).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import DataConfig
from .paths import BOVA11_PATH, CDI_PATH, DATASET_PATH, MEMBERSHIP_PATH

CASH_GIDX = 0

# Per-channel normalization inside a window tensor. "self": divide the window by the
# channel's own value-at-t (paper eq. 18) -- correct only for price LEVELS, where a
# ratio-to-today is the meaningful representation. A float: multiply raw window values
# by that constant instead ("passthrough" at 1.0 for an already-stationary technical
# feature; a smaller constant rescales a wide-but-bounded feature like RSI's 0-100 range
# down near the other channels' O(1) scale). The two kinds also mask differently in
# window_tensor: "self" fills masked/missing with 1.0 (a neutral price-relative), a
# float-scaled channel fills with 0.0 (a neutral "no signal").
FEATURE_NORM = {
    "close": "self", "high": "self", "low": "self",
    "return_1m": 1.0, "return_3m": 1.0, "return_6m": 1.0,
    "price_vs_ma60": 1.0, "volatility_ratio_20_60": 1.0,
    "rsi_14": 0.01,
    "drawdown": 1.0, "volume_ratio_20d": 1.0,
    "pl_zhist_5y": "log1p", "pvp_zhist_5y": "log1p",
    "pl_zhist_5y_isnan": 1.0, "pvp_zhist_5y_isnan": 1.0,
    # Cross-sectional sector z-score (same-day peer group, cross_sectional.py) -- a
    # DIFFERENT computation from pl_zhist_5y above (own 5y history, not peers), and
    # already well-bounded (measured in the built dataset: p1/p99 ~ [-2.7, 2.8], max
    # ~5.6) unlike pl_zhist_5y's robust rolling z-score (which needed "log1p" to tame
    # occasional ~18k blowups from a near-zero rolling IQR) -- plain passthrough is
    # correct here, no squash needed. 0.2 brings its wider spread down near the other
    # channels' O(1) scale, same reasoning as rsi_14's 0.01. isnan mask matters: ~39%
    # of rows are NaN (missing pl that day, or a sector-of-one quarter), too much to
    # silently zero-fill without telling the network "no data" from "z-score of 0".
    "pl_zscore_sector": 0.2, "pl_zscore_sector_isnan": 1.0,
    # Cross-sectional excess return (return_Nm minus that day's cross-sectional mean,
    # cross_sectional.py) -- the one piece of information the encoder structurally can't
    # derive itself: each asset's conv stream only ever sees its OWN window (the "Identical
    # Independent Evaluators" design), so it has no path to compute "how am I doing vs.
    # everyone else today." Measured in the built dataset: low NaN (1.4%/3.4%, ordinary
    # warm-up, not the ~39% that justified an isnan mask for pl_zscore_sector above) and
    # already the same O(1) scale as return_1m/return_3m (p1/p99 ~ [-0.45,0.49]/[-0.87,0.82])
    # -- plain passthrough at 1.0, same treatment as the return_* channels it's derived from.
    "momentum_vs_market_1m": 1.0, "momentum_vs_market_3m": 1.0,
}


@dataclass(frozen=True)
class GlobalAssetIndex:
    """Permanent ticker <-> global-index mapping, index 0 reserved for cash.

    Alphabetical order is arbitrary but fixed: what matters is that it never
    depends on any file's row order, so a slot's ticker identity is stable
    day-to-day within a quarter and the mapping is fully reproducible from
    the membership table alone.
    """
    tickers: tuple  # sorted, length N_union; ticker_to_gidx[tickers[i]] == i + 1
    ticker_to_gidx: dict

    @property
    def n_global(self) -> int:
        return len(self.tickers) + 1  # + cash

    @classmethod
    def from_membership(cls, membership: pd.DataFrame) -> "GlobalAssetIndex":
        tickers = tuple(sorted(membership["ticker"].unique()))
        return cls(tickers=tickers, ticker_to_gidx={t: i + 1 for i, t in enumerate(tickers)})


def validate_cdi_daily_percent(cdi_values: np.ndarray) -> None:
    """Guard against a units mix-up in the CDI series (BCB series 12): the
    loader expects a DAILY rate in PERCENT (e.g. 0.0525 for ~14% p.a.), not
    an annualized percent or a fraction. Checks the implied annualized rate
    against CDI/SELIC's actual historical range in this dataset (verified
    2000-2026: ~2%-19% p.a.) with a generous 1%-60% band -- tight enough to
    catch an off-by-100 (fraction instead of percent) or an annualized value
    dropped in the daily field, loose enough to never false-fail on real data
    (docs/EIIE_AGENT_PLAN.md "Facts verified against the actual data").
    """
    assert np.all(cdi_values >= 0), "CDI must be non-negative (BCB target rate floor)"
    annualized = (1.0 + cdi_values / 100.0) ** 252 - 1.0
    assert np.all(annualized > 0.01) and np.all(annualized < 0.60), (
        f"implausible annualized CDI (range [{annualized.min():.2%}, {annualized.max():.2%}]) "
        "-- check units: expected a DAILY rate in PERCENT, not annualized-percent or a fraction"
    )


def _build_slot_calendar(calendar: pd.DatetimeIndex, membership: pd.DataFrame,
                          asset_index: GlobalAssetIndex, n_slots: int):
    """Per calendar day: which up-to-n_slots global indices are the active
    top-n_slots members that day (the point-in-time selection is done
    upstream, by build_top50_universe.py; this only assigns deterministic
    slots), sorted ascending by permanent global index so no slot maps to a
    fixed ticker across a universe rotation.

    Returns (slot_gidx[T, n_slots] int64, valid[T, n_slots] bool). Calendar
    days before the first membership period (pre-history lookback buffer
    only, never an experiment step) get all-invalid slots. A period with
    fewer than n_slots qualifiers pads with gidx=asset_index.n_global (a
    dummy index past the real 0..n_global-1 range, never 0/cash) and
    valid=False -- see pvm.py's write() for why that specific sentinel
    value matters.
    """
    periods = membership[["period_id", "start"]].drop_duplicates().sort_values("start").reset_index(drop=True)
    cal_df = pd.DataFrame({"trade_date": calendar})
    tagged = pd.merge_asof(cal_df, periods, left_on="trade_date", right_on="start", direction="backward")

    mem = membership.copy()
    mem["gidx"] = mem["ticker"].map(asset_index.ticker_to_gidx)
    mem = mem.sort_values(["period_id", "gidx"])
    period_members = mem.groupby("period_id")["gidx"].apply(list)

    period_ids = periods["period_id"].tolist()
    pid_to_row = {pid: i for i, pid in enumerate(period_ids)}
    # Padding slots (a period with fewer than n_slots qualifiers) default to
    # DUMMY_GIDX (n_global, one past the real 0..n_global-1 range) rather than
    # 0/cash -- so that even if a masked slot's network weight isn't exactly
    # zero, scattering it back into the PVM can never corrupt the cash column
    # (pvm.py's write() relies on this).
    lookup_gidx = np.full((len(period_ids), n_slots), asset_index.n_global, dtype=np.int64)
    lookup_valid = np.zeros((len(period_ids), n_slots), dtype=bool)
    for pid, members in period_members.items():
        if pid not in pid_to_row:
            continue
        row = pid_to_row[pid]
        k = min(len(members), n_slots)
        lookup_gidx[row, :k] = members[:k]
        lookup_valid[row, :k] = True

    pid_arr = tagged["period_id"].to_numpy()
    known = ~pd.isna(pid_arr)
    row_idx = np.zeros(len(calendar), dtype=np.int64)
    row_idx[known] = [pid_to_row[p] for p in pid_arr[known]]

    slot_gidx = lookup_gidx[row_idx]
    valid = lookup_valid[row_idx].copy()
    valid[~known] = False
    return slot_gidx, valid


def _load_bova11(calendar: pd.DatetimeIndex) -> np.ndarray:
    """Benchmark series only -- reindexed onto the same calendar, NEVER fed
    to the model. Sparse day-mismatches flat-filled for a usable eval index."""
    bova = pd.read_parquet(BOVA11_PATH, columns=["trade_date", "adj_close"])
    s = bova.set_index("trade_date")["adj_close"].sort_index()
    return s.reindex(calendar).ffill().bfill().to_numpy()


@dataclass
class PricePanel:
    """Dense price data on the union trading calendar, in global space
    (172 columns = cash + 171 union tickers), plus the per-day slot mapping
    into the network's fixed 50-wide input.
    """
    asset_index: GlobalAssetIndex
    dates: pd.DatetimeIndex
    close: np.ndarray       # (T, n_global)
    high: np.ndarray        # (T, n_global)
    low: np.ndarray         # (T, n_global)
    cdi_factor: np.ndarray  # (T,) -- 1 + cdi_t/100, cash's price-relative factor
    slot_gidx: np.ndarray   # (T, n_slots) int64
    valid: np.ndarray       # (T, n_slots) bool
    window: int
    start_idx: int          # first t with dates[t] >= experiment window_start
    end_idx: int             # last t with dates[t] <= experiment window_end
    bova11_close: Optional[np.ndarray] = None  # (T,), benchmark only
    extra: dict = field(default_factory=dict)  # name -> (T, n_global+1), any DataConfig.features beyond close/high/low

    @property
    def n_slots(self) -> int:
        return self.slot_gidx.shape[1]

    @property
    def n_global(self) -> int:
        """Real global-space width (cash + N_union tickers). close/high/low
        are allocated one column wider than this (see load_price_panel) so a
        padding slot's dummy sentinel index (== n_global, from
        _build_slot_calendar) is always a safe in-bounds gather in
        window_tensor -- its value is masked out immediately after, never
        exposed through this property or price_relative."""
        return self.asset_index.n_global

    def _channel(self, name: str) -> np.ndarray:
        if name in ("close", "high", "low"):
            return {"close": self.close, "high": self.high, "low": self.low}[name]
        return self.extra[name]

    def price_relative(self, t: int) -> np.ndarray:
        """y_t (paper eq. 1), global space: index 0 = cash's CDI factor,
        index i = v_{i,t}/v_{i,t-1}. Requires t >= 1."""
        if t < 1:
            raise ValueError("price_relative requires t >= 1 (needs t-1)")
        n = self.n_global
        y = self.close[t, :n] / self.close[t - 1, :n]
        y[CASH_GIDX] = self.cdi_factor[t]
        return y

    def window_tensor(self, t: int, features=("close", "high", "low")) -> np.ndarray:
        """X_t (paper eq. 18): shape (len(features), n_slots, window). Each
        active slot's channel history is normalized by its own price at t;
        masked/empty slots are filled flat (paper Sec. 3.3, 0-decay).
        Requires t >= window - 1."""
        if t < self.window - 1:
            raise ValueError(f"window_tensor requires t >= window-1 ({self.window - 1}), got {t}")
        gidx = self.slot_gidx[t]
        mask = self.valid[t]
        lo = t - self.window + 1
        out = np.ones((len(features), self.n_slots, self.window), dtype=np.float64)
        for f, name in enumerate(features):
            channel = self._channel(name)
            hist = channel[lo:t + 1, gidx].T        # (n_slots, window)
            norm = FEATURE_NORM[name]
            if norm == "self":
                # Masked slots' gidx can point at a real ticker's pre-listing/
                # padding-period rows (see _build_slot_calendar), which are
                # occasionally a genuine 0.0 far outside the experiment window
                # (e.g. some 2000-era prices) -- swap in a safe 1.0 divisor
                # before dividing rather than dividing then discarding, so a
                # masked slot never triggers a spurious div-by-zero warning.
                v_t = np.where(mask, channel[t, gidx], 1.0)[:, None]  # (n_slots, 1)
                out[f] = np.where(mask[:, None], hist / v_t, 1.0)
            elif norm == "log1p":
                # Sign-preserving log squash for extreme-tailed features (e.g.
                # pl_zhist_5y ±18k) -- maps ±18k → ±9.8, keeps identity near 0.
                hist_clean = np.nan_to_num(hist, nan=0.0)
                hist_squashed = np.sign(hist_clean) * np.log1p(np.abs(hist_clean))
                out[f] = np.where(mask[:, None], hist_squashed, 0.0)
            else:
                # A technical feature can still be NaN inside its own warm-up (e.g.
                # return_6m before 126 days of a ticker's own history) even on a day
                # its slot IS active -- nan_to_num before the mask, not instead of it,
                # so that warm-up (unmasked) and inactive (masked) both land at the
                # same neutral 0.0 rather than leaking a NaN into the network input.
                out[f] = np.where(mask[:, None], np.nan_to_num(hist, nan=0.0) * norm, 0.0)
        return out

    def price_relative_batch(self, t_idx: np.ndarray) -> np.ndarray:
        """Vectorized price_relative (eq. 1) over an array of periods --
        numerically identical to stacking price_relative(t) per t, but one
        batched numpy op instead of a Python loop (OSBL's per-step hot path).
        Requires all t_idx >= 1."""
        t_idx = np.asarray(t_idx)
        if np.any(t_idx < 1):
            raise ValueError("price_relative_batch requires all t >= 1 (needs t-1)")
        n = self.n_global
        y = self.close[t_idx, :n] / self.close[t_idx - 1, :n]
        y[:, CASH_GIDX] = self.cdi_factor[t_idx]
        return y

    def window_tensor_batch(self, t_idx: np.ndarray, features=("close", "high", "low")) -> np.ndarray:
        """Vectorized window_tensor (eq. 18) over an array of periods, shape
        (len(t_idx), len(features), n_slots, window) -- numerically identical
        to stacking window_tensor(t) per t, but built as one batched numpy
        gather instead of a Python loop over each period in the batch (OSBL's
        per-training-step hot path). Requires all t_idx >= window - 1."""
        t_idx = np.asarray(t_idx)
        min_t = self.window - 1
        if np.any(t_idx < min_t):
            raise ValueError(f"window_tensor_batch requires all t >= window-1 ({min_t})")
        B, W = len(t_idx), self.window
        gidx = self.slot_gidx[t_idx]     # (B, n_slots)
        mask = self.valid[t_idx]         # (B, n_slots)
        rows = t_idx[:, None] - W + 1 + np.arange(W)[None, :]                  # (B, W)
        rows_full = np.broadcast_to(rows[:, :, None], (B, W, self.n_slots))
        cols_full = np.broadcast_to(gidx[:, None, :], (B, W, self.n_slots))
        out = np.empty((B, len(features), self.n_slots, W), dtype=np.float64)
        for f, name in enumerate(features):
            channel = self._channel(name)
            hist = channel[rows_full, cols_full].transpose(0, 2, 1)            # (B, n_slots, W)
            norm = FEATURE_NORM[name]
            if norm == "self":
                raw_v_t = channel[t_idx[:, None], gidx]                        # (B, n_slots)
                v_t = np.where(mask, raw_v_t, 1.0)[:, :, None]                 # same masked-divisor
                out[:, f] = np.where(mask[:, :, None], hist / v_t, 1.0)        # guard as window_tensor
            elif norm == "log1p":
                # Sign-preserving log squash for extreme-tailed features (e.g.
                # pl_zhist_5y ±18k) -- maps ±18k → ±9.8, keeps identity near 0.
                hist_clean = np.nan_to_num(hist, nan=0.0)
                hist_squashed = np.sign(hist_clean) * np.log1p(np.abs(hist_clean))
                out[:, f] = np.where(mask[:, :, None], hist_squashed, 0.0)
            else:
                out[:, f] = np.where(mask[:, :, None], np.nan_to_num(hist, nan=0.0) * norm, 0.0)
        return out


def load_price_panel(data_cfg: DataConfig, n_slots: int = 50) -> PricePanel:
    """Build the full PricePanel for an experiment window from the on-disk
    dataset. n_slots is the network's fixed slot width (ModelConfig.n_assets
    -- passed explicitly rather than imported, to keep data.py decoupled
    from ModelConfig)."""
    window_start = pd.Timestamp(data_cfg.window_start)
    window_end = pd.Timestamp(data_cfg.window_end)

    membership = pd.read_parquet(MEMBERSHIP_PATH)
    asset_index = GlobalAssetIndex.from_membership(membership)

    # close/high/low are always loaded (paper's base channels, and close alone drives
    # price_relative's eq. 1 math); any other requested feature is loaded as an extra
    # dense column, keyed by its own name (matches the ml_dataset.parquet column name).
    # Features ending in _isnan are derived (not parquet columns): mark for special handling.
    extra_features = [f for f in data_cfg.features if f not in ("close", "high", "low")]
    real_extra_features = [f for f in extra_features if not f.endswith("_isnan")]
    mask_extra_features = [(f[:-6], f) for f in extra_features if f.endswith("_isnan")]  # (base, mask_name)

    table = pq.read_table(
        DATASET_PATH,
        columns=["ticker", "trade_date", "adj_close", "adj_high", "adj_low"] + real_extra_features,
        filters=[("ticker", "in", list(asset_index.tickers)), ("trade_date", "<=", window_end)],
    )
    prices = table.to_pandas()
    calendar = pd.DatetimeIndex(sorted(prices["trade_date"].unique()))
    prices["gidx"] = prices["ticker"].map(asset_index.ticker_to_gidx).astype(np.int64)

    def _dense(col: str, bfill: bool = True) -> np.ndarray:
        piv = prices.pivot(index="trade_date", columns="gidx", values=col)
        piv = piv.reindex(index=calendar, columns=range(1, asset_index.n_global))
        piv = piv.ffill()  # halts + post-delisting flat
        if bfill:
            piv = piv.bfill()  # pre-listing flat (paper Sec. 3.3) -- prices only, see below
        # +1 column: a dummy index (== n_global) padding slots can safely gather
        # from in window_tensor without an out-of-bounds error; always masked
        # out before use, never surfaced through PricePanel.n_global.
        arr = np.ones((len(calendar), asset_index.n_global + 1), dtype=np.float64)
        arr[:, 1:asset_index.n_global] = piv.to_numpy()
        return arr

    close = _dense("adj_close")
    high = _dense("adj_high")
    low = _dense("adj_low")
    # bfill=False for technicals: unlike a price (whose pre-listing days are always
    # masked out, so backfilling them is harmless), a technical feature's own NaN
    # warm-up (e.g. return_6m before 126 days of THIS ticker's history) can still fall
    # on a day the slot IS active -- bfilling would leak a later, now-defined value
    # backward into that unmasked training row. Left NaN here; window_tensor zeroes it.
    extra = {name: _dense(name, bfill=False) for name in real_extra_features}
    # Derive _isnan mask channels: 1.0 where the base feature was NaN, 0.0 where data exists.
    for base_name, mask_name in mask_extra_features:
        base_dense = extra[base_name] if base_name in extra else _dense(base_name, bfill=False)
        extra[mask_name] = np.isnan(base_dense).astype(np.float64)

    cdi_df = pd.read_parquet(CDI_PATH)
    validate_cdi_daily_percent(cdi_df["cdi"].to_numpy())
    cdi_s = cdi_df.set_index("reference_date")["cdi"].sort_index()
    cdi_factor = 1.0 + cdi_s.reindex(calendar).ffill().bfill().to_numpy() / 100.0

    slot_gidx, valid = _build_slot_calendar(calendar, membership, asset_index, n_slots)

    start_idx = int(np.searchsorted(calendar.values, np.datetime64(window_start)))
    end_idx = int(np.searchsorted(calendar.values, np.datetime64(window_end), side="right") - 1)

    in_window_counts = valid[start_idx:end_idx + 1].sum(axis=1)
    assert np.all(in_window_counts == n_slots), (
        f"expected every in-window trading day to have exactly {n_slots} active members "
        f"(docs/EIIE_AGENT_PLAN.md verified fact); got counts in "
        f"[{in_window_counts.min()}, {in_window_counts.max()}]"
    )

    return PricePanel(
        asset_index=asset_index,
        dates=calendar,
        close=close, high=high, low=low,
        cdi_factor=cdi_factor,
        slot_gidx=slot_gidx, valid=valid,
        window=data_cfg.window,
        start_idx=start_idx, end_idx=end_idx,
        bova11_close=_load_bova11(calendar),
        extra=extra,
    )
