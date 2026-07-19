"""
spine.py — H-series walk-forward decision grid (MEDIUM_HORIZON_RESEARCH_PLAN.md
Task H0): the monthly rebalance calendar, expanding-window OOS fold
boundaries, the point-in-time active universe per decision date, and the
forward relative-return target construction (raw + centered-uniform rank).

Everything here is dates and returns only -- no fundamentals, no sector
logic (that's features.py) -- so it's reusable by both the H0 baseline
backtest (which only needs the stitched-OOS start boundary) and H1/H2's
monthly panel (which needs the full target matrix).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stats import rank_normalize, winsorize_cross_sectional


@dataclass(frozen=True)
class FoldWindow:
    """One expanding-window fold: the OOS period is (train_end, oos_end] --
    everything up to and including train_end is available for fitting,
    nothing in the OOS period was seen. Mirrors build_dataset/manifest.py's
    FitWindow convention (fit_start=None there == "from the beginning" --
    matched here by every fold's train START implicitly being the
    dataset's first date; this is expanding, not rolling)."""
    fold_id: str
    train_end: pd.Timestamp
    oos_end: pd.Timestamp


def iter_expanding_folds(window_end, initial_train_end, step_months: int = 12) -> list:
    """Expanding-window OOS folds from initial_train_end to window_end,
    stepping step_months at a time. H1/H2 use this to refit per fold; H0's
    baselines (UCRP/BOVA11/min-variance/classical MV) have no fitted
    global parameter -- they re-estimate Sigma (and, for classical MV, mu)
    from a trailing lookback window AT EVERY rebalance, which is already
    causal by construction, so H0 only needs fold[0].train_end as the
    single stitched-OOS start boundary (see milestone_h0.py's module
    docstring for the full reasoning)."""
    window_end = pd.Timestamp(window_end)
    train_end = pd.Timestamp(initial_train_end)
    folds = []
    i = 0
    while train_end < window_end:
        oos_end = min(train_end + pd.DateOffset(months=step_months), window_end)
        folds.append(FoldWindow(fold_id=f"fold{i}", train_end=train_end, oos_end=oos_end))
        train_end = oos_end
        i += 1
    return folds


def hac_lag_for_horizon(k: int, sampling_days: int = 21) -> int:
    """Newey-West lag, in units of the monthly SAMPLING interval, for a
    k-trading-day-ahead target sampled every `sampling_days` days. Overlap
    persists for round(k/sampling_days)-1 extra sampling periods: k=21
    (one month) sampled monthly is non-overlapping (lag=0); k=63 (one
    quarter) sampled monthly overlaps its two neighboring decisions
    (lag=2)."""
    return max(0, round(k / sampling_days) - 1)


def monthly_decision_dates(trading_dates) -> pd.DatetimeIndex:
    """Last trading day of each calendar month present in trading_dates."""
    idx = pd.DatetimeIndex(sorted(pd.DatetimeIndex(trading_dates).unique()))
    s = pd.Series(idx, index=idx)
    last_per_month = s.groupby(s.dt.to_period("M")).max()
    return pd.DatetimeIndex(last_per_month.to_numpy())


def k_trading_days_later(calendar: pd.DatetimeIndex, dates, k: int) -> pd.DatetimeIndex:
    """The calendar date k TRADING days after each of `dates` (calendar
    must contain every date in `dates`). Past the end of `calendar`
    returns NaT -- the caller's forward return for that row is NaN, not a
    fabricated value."""
    calendar = pd.DatetimeIndex(calendar)
    dates = pd.DatetimeIndex(dates)
    pos = calendar.searchsorted(dates)
    target_pos = pos + k
    in_range = target_pos < len(calendar)
    safe_pos = np.clip(target_pos, 0, len(calendar) - 1)
    out = np.where(in_range, calendar[safe_pos].to_numpy(), np.datetime64("NaT"))
    return pd.DatetimeIndex(out)


def active_universe_by_date(membership: pd.DataFrame, decision_dates) -> pd.DataFrame:
    """Long-format (decision_date, ticker) -- the point-in-time top-50
    membership as of each monthly decision date. membership periods are
    half-open [start, end): a date exactly on `end` has already rolled to
    the NEXT period (matches src/rl_agent/data.py's _build_slot_calendar
    convention)."""
    decision_dates = pd.DatetimeIndex(decision_dates)
    periods = membership[["period_id", "start", "end"]].drop_duplicates().sort_values("start")
    cal_df = pd.DataFrame({"decision_date": decision_dates}).sort_values("decision_date")
    tagged = pd.merge_asof(cal_df, periods, left_on="decision_date", right_on="start", direction="backward")
    tagged = tagged[tagged["decision_date"] < tagged["end"]]
    out = tagged.merge(membership[["period_id", "ticker"]], on="period_id", how="left")
    return out[["decision_date", "ticker"]].reset_index(drop=True)


def build_forward_targets(prices_wide: pd.DataFrame, bench: pd.Series,
                           decision_dates, k: int, universe: pd.DataFrame) -> pd.DataFrame:
    """Forward k-trading-day BOVA11-relative return per (decision_date,
    ticker), restricted to that date's active universe (`universe`, from
    active_universe_by_date) -- n_universe_members is that restricted
    membership count (a reporting/diagnostic figure); the N actually used
    inside rank_normalize's denominator is the count of non-NaN returns
    within that universe, which can be smaller (e.g. a halt on the target
    date), by rank_normalize's own contract.

    prices_wide: index=trade_date, columns=ticker, values=adj_close.
    bench: index=trade_date, values=BOVA11 adj_close.
    Returns columns: decision_date, ticker, fwd_rel_return, target_rank,
    fwd_rel_return_winsorized, n_universe_members.
    """
    decision_dates = pd.DatetimeIndex(decision_dates)
    calendar = prices_wide.index
    target_dates = k_trading_days_later(calendar, decision_dates, k)

    p0 = prices_wide.reindex(decision_dates)
    p1 = prices_wide.reindex(target_dates)
    p1.index = decision_dates  # realign onto decision_dates for elementwise divide despite the NaT-safe reindex
    fwd_ret = p1.to_numpy() / p0.to_numpy() - 1.0

    b0 = bench.reindex(decision_dates).to_numpy()
    b1 = bench.reindex(target_dates).to_numpy()
    bench_ret = b1 / b0 - 1.0

    rel = fwd_ret - bench_ret[:, None]
    wide = pd.DataFrame(rel, index=decision_dates, columns=prices_wide.columns)
    wide.index.name = "decision_date"
    wide.columns.name = "ticker"
    long = wide.stack(future_stack=True).rename("fwd_rel_return").reset_index()  # future_stack never drops NA rows

    out = universe.merge(long, on=["decision_date", "ticker"], how="left")
    out["n_universe_members"] = out.groupby("decision_date")["ticker"].transform("nunique")
    out["target_rank"] = rank_normalize(out["fwd_rel_return"], out["decision_date"])
    out["fwd_rel_return_winsorized"] = winsorize_cross_sectional(out["fwd_rel_return"], out["decision_date"])
    return out
