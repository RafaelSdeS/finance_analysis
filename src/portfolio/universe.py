"""
universe.py -- thin wrapper over the existing point-in-time liquid-universe
helper (src/build_dataset/build_top50_universe.py). That module already
implements trailing-window, no-lookahead, quarterly-rebalanced membership
(verified end-to-end on the real dataset 2026-07-24 -- see
docs/PORTFOLIO_IMPLEMENTATION_PLAN.md §2.1); this module does not
recompute any of that, it only exposes it the way the backtest needs.
"""

import pandas as pd

from src.build_dataset.build_top50_universe import build_top50_membership


def liquid_universe(df: pd.DataFrame, top_n: int = 50, rebalance_freq: str = "Q") -> pd.DataFrame:
    """Point-in-time membership table: one row per (ticker, period) the
    ticker qualified for, ranked by trailing 252-trading-day traded_amount
    as of that period's rebalance date. `df` needs ticker/trade_date/traded_amount.
    """
    return build_top50_membership(df, top_n=top_n, rebalance_freq=rebalance_freq)


def rebalance_dates(membership: pd.DataFrame) -> pd.DatetimeIndex:
    """The membership table's period-start dates ARE the rebalance calendar
    -- reuse them, don't recompute a separate date grid (proposal §7 note)."""
    return pd.DatetimeIndex(sorted(membership["start"].unique()))


def universe_at(membership: pd.DataFrame, date: pd.Timestamp) -> set:
    """Tickers qualified for the locked period covering `date` (start <= date < end).
    Used at exact rebalance dates by the backtest; the half-open range also makes it
    safe to query at any date, not just a rebalance date, if that's ever needed."""
    hit = membership[(membership["start"] <= date) & (date < membership["end"])]
    return set(hit["ticker"])
