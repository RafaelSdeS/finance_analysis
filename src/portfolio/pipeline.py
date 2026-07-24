"""
pipeline.py -- wires Stage A (alpha) + the risk model (Sigma) + Stage B (the
convex optimizer) into a single weights_fn the backtest harness can drive
(proposal Phase 2.7): the full predict-then-optimize loop, end to end.
"""

import pandas as pd

from src.portfolio.optimizer import solve
from src.portfolio.risk import add_cash_row_col, shrinkage_cov

CASH_KEY = "cash"


def make_full_weights_fn(alpha_by_date: dict, price_wide: pd.DataFrame,
                          sigma_window: int = 252, c1=0.0003, c2: float = 0.0,
                          lam: float = 1.0, w_max: float = 1.0):
    """
    alpha_by_date: {date: {ticker: alpha}} -- e.g. alpha.walk_forward_predict()'s
        output grouped by date. `alpha_i` is already an excess-return-over-CDI
        estimate (the label's own definition), so alpha_cash=0 is the
        consistent baseline -- no separate CDI-carry term needed here.
    price_wide: DataFrame[date, ticker] of adj_close, NOT forward-filled --
        a real gap in a ticker's trading history should show up as NaN and
        get that ticker safely excluded from this quarter's investable set
        (below) rather than silently deflating its estimated variance.
    c1: one-way per-asset cost passed straight through to optimizer.solve().
    Returns a weights_fn(date, universe, state) -> dict[ticker, weight]
        compatible with backtest.run_backtest(). Falls back to equal-weight
        on any date without an alpha prediction yet (early history) or
        without enough tickers with a full trailing sigma_window (very
        early history, before the risk model has enough data either).
    """
    def weights_fn(date, universe, state):
        universe = sorted(universe)
        n = len(universe)
        if n == 0:
            return {}
        if date not in alpha_by_date:
            return {t: 1.0 / n for t in universe}

        returns_window = price_wide.loc[:date, universe].pct_change().iloc[1:].tail(sigma_window)
        sigma = shrinkage_cov(returns_window)
        investable = list(sigma.columns)  # only tickers with a full, gap-free trailing window
        if not investable:
            return {t: 1.0 / n for t in universe}

        preds = alpha_by_date[date]
        alpha_series = pd.Series({**{t: preds.get(t, 0.0) for t in investable}, CASH_KEY: 0.0})
        sigma = add_cash_row_col(sigma, cash_key=CASH_KEY)

        prev_w = dict(state["prev_weights"])
        prev_w[CASH_KEY] = 1.0 - sum(prev_w.values())
        w_prev = pd.Series(prev_w)

        w = solve(alpha_series, sigma, w_prev, c1=c1, c2=c2, lam=lam, w_max=w_max, cash_key=CASH_KEY)
        return {t: float(w[t]) for t in investable if w.get(t, 0) > 1e-8}

    return weights_fn
