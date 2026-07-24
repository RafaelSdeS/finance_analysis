"""
backtest.py -- quarterly-rebalanced portfolio backtest harness (proposal
§2.3/Phase 2a). Buy-and-hold between rebalances: weights DRIFT with prices
until the next rebalance re-targets them. This is deliberate, not an
approximation -- it's what lets a no-trade band's "winners run" behavior
actually show up in the equity curve; daily-renormalizing to constant
weights would silently erase exactly the behavior this pipeline exists to
test (proposal §4.3).

Universe churn (a name can exit the point-in-time liquid universe between
rebalances) is handled by unioning the previous holdings with the current
universe at every rebalance: an exited name's target weight is implicitly
0, so it's force-liquidated and that trade's cost/turnover IS counted --
proposal §5 pothole P6, not left as an unmodeled gap.
"""

import numpy as np
import pandas as pd

from src.portfolio import universe as universe_mod

ONE_WAY_COST_DEFAULT = 0.0003  # 0.03% B3 fee floor, one-way (proposal §5/§6 fix)


def equal_weight_fn(date, universe: set, state: dict) -> dict:
    """Reference/baseline strategy (proposal Phase 2a): equal-weight the
    current liquid universe every rebalance, ignoring state entirely."""
    n = len(universe)
    if n == 0:
        return {}
    w = 1.0 / n
    return {ticker: w for ticker in universe}


def _snap(date: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Snap `date` to itself if present, else the nearest earlier date in `index`."""
    if date in index:
        return date
    pos = index.get_indexer([date], method="ffill")[0]
    if pos == -1:
        raise ValueError(f"no date in index at or before {date}")
    return index[pos]


def run_backtest(prices: pd.DataFrame, cdi: pd.DataFrame, membership: pd.DataFrame,
                  weights_fn, one_way_cost: float = ONE_WAY_COST_DEFAULT,
                  initial_value: float = 1.0) -> tuple:
    """
    prices: long DataFrame[ticker, trade_date, adj_close] -- extra tickers
        beyond what `membership`/`weights_fn` ever select are simply unused.
    cdi: DataFrame[trade_date, cdi] -- market-wide, one row per date, %/day
        (manifest.COLUMN_UNITS convention).
    membership: point-in-time universe table (universe.liquid_universe()).
    weights_fn(date, universe: set[str], state: dict) -> dict[ticker, weight]
        Weights need not sum to 1 -- the residual (1 - sum) is cash. `state`
        carries {"date", "port_value", "prev_weights"} (last period's
        realized, drifted weights) for strategies that need it -- e.g. a
        no-op/no-trade test strategy, or a later optimizer's w_{t-1}.
    Returns (equity_curve: Series indexed by trade_date, rebalance_log: DataFrame).
    Rebalance-log `turnover` = sum(|Δw_i|) over equities only (cash excluded,
    matching "the B3 fee applies to equity trades, not moving into cash") --
    a full one-way book replacement reads as turnover=2 (100% sold + 100%
    bought), the convention metrics.turnover_stats expects.
    """
    price_wide = prices.pivot(index="trade_date", columns="ticker", values="adj_close")
    price_wide = price_wide.ffill()  # forward-only: never leaks a future price into a gap
    all_dates = price_wide.index

    cdi_rate = cdi.set_index("trade_date")["cdi"].reindex(all_dates).ffill() / 100
    cum_log_cdi = np.log1p(cdi_rate).cumsum()

    reb_dates = universe_mod.rebalance_dates(membership)
    reb_dates = pd.DatetimeIndex([_snap(d, all_dates) for d in reb_dates if d <= all_dates[-1]])

    prev_shares: dict = {}
    prev_cash = initial_value
    log_rows = []
    curve = {}

    for k, t in enumerate(reb_dates):
        price_t = price_wide.loc[t]
        equity_value_t = sum(
            sh * price_t[tkr] for tkr, sh in prev_shares.items()
            if tkr in price_t.index and pd.notna(price_t[tkr])
        )
        port_value = prev_cash + equity_value_t
        prev_w = (
            {tkr: (sh * price_t[tkr]) / port_value for tkr, sh in prev_shares.items()
             if tkr in price_t.index and pd.notna(price_t[tkr])}
            if port_value > 0 else {}
        )

        current_universe = universe_mod.universe_at(membership, t)
        raw_new_w = weights_fn(t, current_universe, {
            "date": t, "port_value": port_value, "prev_weights": prev_w,
        })
        # drop anything weights_fn wanted but has no valid price at t -- its
        # capital correctly falls through to cash below, not silently lost.
        new_w = {
            tkr: w for tkr, w in raw_new_w.items()
            if w and tkr in price_t.index and pd.notna(price_t[tkr]) and price_t[tkr] > 0
        }

        full_names = set(prev_w) | set(new_w)
        delta = {tkr: new_w.get(tkr, 0.0) - prev_w.get(tkr, 0.0) for tkr in full_names}
        turnover = sum(abs(v) for v in delta.values())
        cost = one_way_cost * turnover

        capital = port_value * (1 - cost)
        new_shares = {tkr: (capital * w) / price_t[tkr] for tkr, w in new_w.items()}
        cash_after = capital * (1 - sum(new_w.values()))

        log_rows.append({
            "date": t, "turnover": turnover, "cost": cost,
            "n_holdings": len(new_shares), "cash_weight": 1 - sum(new_w.values()),
            "port_value_pre_cost": port_value,
        })

        next_t = reb_dates[k + 1] if k + 1 < len(reb_dates) else all_dates[-1]
        period_dates = all_dates[(all_dates >= t) & (all_dates <= next_t)]

        held_tickers = list(new_shares.keys())
        if held_tickers:
            equity_path = price_wide.loc[period_dates, held_tickers].mul(
                pd.Series(new_shares), axis=1
            ).sum(axis=1)
        else:
            equity_path = pd.Series(0.0, index=period_dates)

        cash_factor = np.exp(cum_log_cdi.loc[period_dates] - cum_log_cdi.loc[t])
        total_path = equity_path + cash_after * cash_factor
        curve.update(total_path.to_dict())
        # (period boundaries overlap by one day with the next iteration's `t`
        # -- both compute the identical mark-to-market value at that instant,
        # so the overwrite in `curve` is a no-op, not a discontinuity)

        prev_shares = new_shares
        prev_cash = cash_after * np.exp(cum_log_cdi.loc[next_t] - cum_log_cdi.loc[t])

    equity_curve = pd.Series(curve).sort_index()
    rebalance_log = pd.DataFrame(log_rows)
    return equity_curve, rebalance_log


def buy_and_hold_curve(prices: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """`prices` is a single-asset adj_close series indexed by trade_date.
    Trivial buy-and-hold from its first available date -- no rebalancing, no cost."""
    prices = prices.dropna()
    return initial_value * prices / prices.iloc[0]


def cdi_curve(cdi: pd.DataFrame, initial_value: float = 1.0) -> pd.Series:
    """100% CDI accrual curve. `cdi` needs trade_date/cdi (%/day)."""
    cdi_rate = cdi.set_index("trade_date")["cdi"].sort_index() / 100
    return initial_value * np.exp(np.log1p(cdi_rate).cumsum())
