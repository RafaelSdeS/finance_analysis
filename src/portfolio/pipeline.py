"""
pipeline.py -- wires Stage A (alpha) + the risk model (Sigma) + Stage B (the
convex optimizer) into a single weights_fn the backtest harness can drive
(proposal Phase 2.7): the full predict-then-optimize loop, end to end.
"""

import pandas as pd

from src.portfolio.alpha import shrink_alpha
from src.portfolio.optimizer import solve
from src.portfolio.risk import add_cash_row_col, shrinkage_cov

CASH_KEY = "cash"


def scaled_sigma(price_wide: pd.DataFrame, date: pd.Timestamp, universe: list,
                  sigma_window: int, horizon_td: int) -> pd.DataFrame:
    """The risk model as make_full_weights_fn actually uses it: bounded-ffill
    (see make_full_weights_fn's docstring for why), shrinkage_cov on daily
    returns, then scaled by `horizon_td` to match alpha's horizon. Exposed
    separately so the scaling itself is directly testable, not just
    indirectly through a full weights_fn call."""
    returns_window = (
        price_wide.loc[:date, universe].ffill(limit=5)
        .pct_change(fill_method=None).iloc[1:].tail(sigma_window)
    )
    return shrinkage_cov(returns_window) * horizon_td


def make_full_weights_fn(alpha_by_date: dict, price_wide: pd.DataFrame,
                          sigma_window: int = 252, horizon_td: int = 252,
                          c1=0.0003, c2: float = 0.0, lam: float = 1.0, w_max: float = 1.0,
                          exposure_by_date: dict | None = None, shrink_factor: float = 0.0):
    """
    alpha_by_date: {date: {ticker: alpha}} -- e.g. alpha.walk_forward_predict()'s
        output grouped by date. `alpha_i` is already an excess-return-over-CDI
        estimate (the label's own definition), so alpha_cash=0 is the
        consistent baseline -- no separate CDI-carry term needed here.
    price_wide: DataFrame[date, ticker] of adj_close. Bounded-forward-filled
        (5 trading days) before computing returns for the risk model --
        found empirically (2026-07-24) that a single real, market-wide data
        gap (e.g. 2026-05-22, missing for 49/50 top-50 tickers simultaneously
        despite the date being a real trading day for others) otherwise
        disqualifies almost the entire universe from shrinkage_cov's strict
        "any NaN drops the column" rule for every rebalance whose trailing
        window includes that date -- forcing mass liquidation and immediate
        repurchase that has nothing to do with alpha or genuine risk. A 5-day
        cap still excludes a ticker that's genuinely stopped trading (as
        opposed to one narrow vendor gap), unlike an unbounded ffill.
    horizon_td: the alpha label's own forward horizon (trading days) -- MUST
        match whatever `horizon_td` built the labels the model was trained
        on. `shrinkage_cov` estimates a DAILY covariance from daily returns;
        `alpha_i` is a cumulative return over `horizon_td` days -- feeding
        the optimizer a daily Sigma against an H-day alpha is a real units
        mismatch (found empirically 2026-07-24: daily variance ~0.0003-0.0008
        vs. alpha on the order of several % to tens of %, ~100-300x apart),
        which makes the risk-aversion term negligible and the optimizer
        chase whichever ticker has the highest apparent (noisy) alpha every
        quarter -- high turnover, worse Sharpe/drawdown than equal-weight,
        the opposite of the proposal's "L1 no-trade band" thesis. Scaling
        Sigma by `horizon_td` (the standard i.i.d.-returns approximation,
        Var(H-day return) ~= H * Var(daily return)) puts it back on the same
        basis as alpha.
    c1: one-way per-asset cost passed straight through to optimizer.solve().
    exposure_by_date: optional {date: equity_cap} from contrarian.equity_exposure()
        -- the Layer-2 "cannons/violins" overlay. Caps total equity weight per
        rebalance (residual -> CDI); None = always 100% equity (overlay off).
    shrink_factor: alpha.shrink_alpha() factor in [0, 1] applied to the raw
        predictions before optimization (plan Phase 1.3) -- 0 (default)
        leaves alpha untouched. Caps how much a single noisy quarterly
        estimate can dominate the allocation (Michaud error-max fix).
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

        sigma = scaled_sigma(price_wide, date, universe, sigma_window, horizon_td)
        investable = list(sigma.columns)  # only tickers with a full, gap-free trailing window
        if not investable:
            return {t: 1.0 / n for t in universe}

        preds = alpha_by_date[date]
        alpha_series = pd.Series({**{t: preds.get(t, 0.0) for t in investable}, CASH_KEY: 0.0})
        alpha_series = shrink_alpha(alpha_series, shrink_factor)
        sigma = add_cash_row_col(sigma, cash_key=CASH_KEY)

        prev_w = dict(state["prev_weights"])
        prev_w[CASH_KEY] = 1.0 - sum(prev_w.values())
        w_prev = pd.Series(prev_w)

        max_equity = 1.0 if exposure_by_date is None else exposure_by_date.get(date, 1.0)
        w = solve(alpha_series, sigma, w_prev, c1=c1, c2=c2, lam=lam, w_max=w_max,
                  cash_key=CASH_KEY, max_equity=max_equity)
        return {t: float(w[t]) for t in investable if w.get(t, 0) > 1e-8}

    return weights_fn
