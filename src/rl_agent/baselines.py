"""
baselines.py — the required baseline suite (docs/EIIE_AGENT_PLAN.md
"Evaluation"), all running through the exact same environment.run_backtest
loop, costs, and dates as the trained agent -- "the RL agent should never
be evaluated in isolation."

Six of the seven are ordinary weight_fn's plugged into run_backtest. BOVA11
is the exception: it isn't a portfolio built from our universe at all, so
it bypasses run_backtest and is evaluated directly from its own price
series (holding a single already-diversified ETF has no rebalancing
decisions in this framework, hence no transaction-cost mechanics to apply).
"""

from typing import Optional

import numpy as np

from .data import CASH_GIDX, PricePanel
from .environment import BacktestResult, run_backtest

BASELINE_NAMES = (
    "ubah", "ucrp", "best_stock", "random_portfolio",
    "random_rebalancing", "constant_cash", "bova11",
)


def _active_gidx(t: int, panel: PricePanel) -> np.ndarray:
    mask = panel.valid[t]
    return panel.slot_gidx[t][mask]


def constant_cash_weight_fn(t, w_prev, w_drift, panel):
    """100% cash, always -- the pure-CDI baseline."""
    w = np.zeros(panel.n_global)
    w[CASH_GIDX] = 1.0
    return w


def ucrp_weight_fn(t, w_prev, w_drift, panel):
    """Uniform Constant Rebalanced Portfolio: equal weight across cash +
    every currently-active asset, rebalanced back to uniform every period."""
    active = _active_gidx(t, panel)
    w = np.zeros(panel.n_global)
    share = 1.0 / (len(active) + 1)
    w[CASH_GIDX] = share
    w[active] = share
    return w


def make_ubah_weight_fn(start_idx: int):
    """Uniform Buy And Hold: the same uniform allocation as UCRP, bought
    exactly once at the start of the backtest, then never rebalanced again
    -- weights simply drift with prices thereafter (matching real-world
    buy-and-hold: a name doesn't get sold just because it later drops out
    of the top-50). start_idx must be the BACKTEST's first day, not
    panel.start_idx -- checking the latter meant the buy never fired on
    eval splits and UBAH sat 100% in cash."""
    def fn(t, w_prev, w_drift, panel):
        if t == start_idx:
            return ucrp_weight_fn(t, w_prev, w_drift, panel)
        return w_drift
    return fn


def make_random_portfolio_weight_fn(seed: int, start_idx: int):
    """A single Dirichlet-drawn allocation (over cash + today's active
    assets), bought once and held -- a static baseline, distinct from
    random_rebalancing's daily churn."""
    def fn(t, w_prev, w_drift, panel):
        if t == start_idx:
            active = _active_gidx(t, panel)
            rng = np.random.default_rng(seed)
            draw = rng.dirichlet(np.ones(len(active) + 1))
            w = np.zeros(panel.n_global)
            w[CASH_GIDX] = draw[0]
            w[active] = draw[1:]
            return w
        return w_drift
    return fn


def make_random_rebalancing_weight_fn(seed: int):
    """A fresh Dirichlet draw every period -- exercises transaction costs
    the way an uninformed but active trader would."""
    rng = np.random.default_rng(seed)

    def fn(t, w_prev, w_drift, panel):
        active = _active_gidx(t, panel)
        draw = rng.dirichlet(np.ones(len(active) + 1))
        w = np.zeros(panel.n_global)
        w[CASH_GIDX] = draw[0]
        w[active] = draw[1:]
        return w
    return fn


def _best_stock_gidx(panel: PricePanel, start_idx: int, end_idx: int) -> int:
    """The single best-performing asset among those active on day 1 of the
    backtest, in hindsight over the full window -- paper's "Best Stock"
    benchmark, restricted to the assets actually investable at the start
    (matching the paper's fixed-preselection assumption)."""
    active = _active_gidx(start_idx, panel)
    total_return = panel.close[end_idx, active] / panel.close[start_idx, active]
    return int(active[np.argmax(total_return)])


def make_best_stock_weight_fn(panel: PricePanel, start_idx: int, end_idx: int):
    best_gidx = _best_stock_gidx(panel, start_idx, end_idx)

    def fn(t, w_prev, w_drift, panel):
        if t == start_idx:
            w = np.zeros(panel.n_global)
            w[best_gidx] = 1.0
            return w
        return w_drift
    return fn


def _bova11_result(panel: PricePanel, start_idx: int, end_idx: int) -> BacktestResult:
    """BOVA11 (IBOV proxy ETF), evaluated directly from its own adj_close
    series -- bypasses run_backtest entirely since it isn't a portfolio
    built from our universe. mu=1/turnover=0 throughout: passively holding
    one already-diversified ETF has no rebalancing decisions to cost."""
    prices = panel.bova11_close[start_idx - 1: end_idx + 1]
    log_returns = np.diff(np.log(prices))
    T = len(log_returns)
    portfolio_value = np.concatenate([[1.0], np.exp(np.cumsum(log_returns))])
    return BacktestResult(
        dates=panel.dates[start_idx:end_idx + 1],
        portfolio_value=portfolio_value,
        log_returns=log_returns,
        mu=np.ones(T),
        turnover=np.zeros(T),
        cost=np.zeros(T),
        weights=np.zeros((T, panel.n_global)),
    )


def run_baseline(name: str, panel: PricePanel, c_sell: float, c_buy: float,
                  seed: int = 42, start_idx: Optional[int] = None, end_idx: Optional[int] = None,
                  mu_tol: float = 1e-10, mu_max_iter: int = 100) -> BacktestResult:
    """Run any of the BASELINE_NAMES over [start_idx, end_idx] with the same
    costs and dates as the agent's own backtest."""
    start_idx = panel.start_idx if start_idx is None else start_idx
    end_idx = panel.end_idx if end_idx is None else end_idx

    if name == "bova11":
        return _bova11_result(panel, start_idx, end_idx)

    weight_fns = {
        "constant_cash": constant_cash_weight_fn,
        "ucrp": ucrp_weight_fn,
        "ubah": make_ubah_weight_fn(start_idx),
        "best_stock": make_best_stock_weight_fn(panel, start_idx, end_idx),
        "random_portfolio": make_random_portfolio_weight_fn(seed, start_idx),
        "random_rebalancing": make_random_rebalancing_weight_fn(seed),
    }
    if name not in weight_fns:
        raise ValueError(f"unknown baseline: {name!r} (available: {sorted(weight_fns) + ['bova11']})")
    return run_backtest(panel, weight_fns[name], c_sell, c_buy, start_idx, end_idx, mu_tol, mu_max_iter)
