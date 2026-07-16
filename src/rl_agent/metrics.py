"""
metrics.py — performance metrics + block-bootstrap confidence intervals
for a BacktestResult (docs/EIIE_AGENT_PLAN.md "Evaluation" section).

Every function takes plain numpy arrays (not BacktestResult directly) so
they're independently testable and reusable outside the backtest loop.
Per-period figures use the paper's *simple* periodic return rho_t = e^{r_t}-1
(eq. 9), recovered from the log return r_t environment.py stores -- Sharpe
(eq. 28) is explicitly defined on rho_t, not r_t.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .environment import BacktestResult

TRADING_DAYS_PER_YEAR = 252


def simple_returns(log_returns: np.ndarray) -> np.ndarray:
    return np.exp(log_returns) - 1.0


def total_return(result: BacktestResult) -> float:
    return float(result.portfolio_value[-1] - 1.0)


def final_apv(result: BacktestResult) -> float:
    return float(result.portfolio_value[-1])


def annualized_return(result: BacktestResult, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Trading-day compounding: (fAPV)^(252/T) - 1."""
    T = len(result.log_returns)
    return float(result.portfolio_value[-1] ** (periods_per_year / T) - 1.0)


def cagr(result: BacktestResult) -> float:
    """Calendar-time compounding: (fAPV)^(1/years_elapsed) - 1, distinct from
    annualized_return's trading-day basis (both requested by the plan)."""
    years = (result.dates[-1] - result.dates[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float(result.portfolio_value[-1] ** (1.0 / years) - 1.0)


def volatility(result: BacktestResult, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return float(result.log_returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(result: BacktestResult, risk_free_returns: np.ndarray,
                  periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """eq. 28: S = E[rho_t - rho_F] / std(rho_t - rho_F)."""
    excess = simple_returns(result.log_returns) - risk_free_returns
    std = excess.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(result: BacktestResult, risk_free_returns: np.ndarray,
                   periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    excess = simple_returns(result.log_returns) - risk_free_returns
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std(ddof=1) == 0:
        return float("nan")
    return float(excess.mean() / downside.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(portfolio_value: np.ndarray) -> float:
    """eq. 29, standard running-peak formulation: max_t (peak_t - v_t) / peak_t."""
    running_max = np.maximum.accumulate(portfolio_value)
    drawdown = (running_max - portfolio_value) / running_max
    return float(drawdown.max())


def calmar_ratio(result: BacktestResult, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    mdd = max_drawdown(result.portfolio_value)
    if mdd == 0:
        return float("nan")
    return float(annualized_return(result, periods_per_year) / mdd)


def historical_var(returns: np.ndarray, level: float = 0.95) -> float:
    """Historical VaR at `level` confidence, reported as a positive loss."""
    return float(-np.quantile(returns, 1.0 - level))


def historical_cvar(returns: np.ndarray, level: float = 0.95) -> float:
    """Expected loss beyond the VaR threshold, reported as a positive loss."""
    threshold = np.quantile(returns, 1.0 - level)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return float(-threshold)
    return float(-tail.mean())


def mean_daily_turnover(result: BacktestResult) -> float:
    return float(result.turnover.mean())


def annualized_turnover(result: BacktestResult, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return float(result.turnover.mean() * periods_per_year)


def transaction_cost_drag(result: BacktestResult) -> float:
    """Total multiplicative wealth lost to costs over the whole backtest:
    1 - prod(mu_t), i.e. what fAPV would have been divided by if costs were
    zero (mu == 1 every period)."""
    return float(1.0 - np.prod(result.mu))


def win_rate(returns: np.ndarray) -> float:
    return float((returns > 0).mean())


def information_ratio(result: BacktestResult, benchmark_returns: np.ndarray,
                       periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    active = simple_returns(result.log_returns) - benchmark_returns
    std = active.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(active.mean() / std * np.sqrt(periods_per_year))


def block_bootstrap_ci(returns: np.ndarray, stat_fn: Callable[[np.ndarray], float],
                        n_bootstrap: int = 1000, block_size: int = 20,
                        level: float = 0.95, seed: Optional[int] = None) -> tuple:
    """Moving-block bootstrap CI for any scalar statistic of a returns
    series -- financial return series are autocorrelated, so an i.i.d.
    resample would understate uncertainty; resampling contiguous blocks
    preserves short-range dependence. Returns (point_estimate, lo, hi)."""
    rng = np.random.default_rng(seed)
    T = len(returns)
    n_blocks = int(np.ceil(T / block_size))
    stats = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        starts = rng.integers(0, max(T - block_size, 1) + 1, size=n_blocks)
        sample = np.concatenate([returns[s:s + block_size] for s in starts])[:T]
        stats[b] = stat_fn(sample)

    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return float(stat_fn(returns)), float(lo), float(hi)


@dataclass
class MetricsSummary:
    total_return: float
    annualized_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    var: float
    cvar: float
    mean_daily_turnover: float
    annualized_turnover: float
    transaction_cost_drag: float
    win_rate: float
    information_ratio: float
    final_apv: float
    total_return_ci: tuple  # (point, lo, hi)
    sharpe_ci: tuple


def summarize(result: BacktestResult, risk_free_returns: np.ndarray, benchmark_returns: np.ndarray,
              var_level: float = 0.95, bootstrap_n: int = 1000, bootstrap_block: int = 20,
              periods_per_year: int = TRADING_DAYS_PER_YEAR, seed: Optional[int] = None) -> MetricsSummary:
    """One-stop metrics + bootstrap CIs for an experiment report, computed
    identically for the agent and every baseline (docs/EIIE_AGENT_PLAN.md
    "Evaluation": the agent is never evaluated in isolation)."""
    returns = simple_returns(result.log_returns)

    def _total_return_stat(r: np.ndarray) -> float:
        return float(np.prod(1.0 + r) - 1.0)

    def _sharpe_stat(r: np.ndarray) -> float:
        rf = risk_free_returns[: len(r)]
        excess = r - rf
        std = excess.std(ddof=1)
        return float(excess.mean() / std * np.sqrt(periods_per_year)) if std != 0 else float("nan")

    return MetricsSummary(
        total_return=total_return(result),
        annualized_return=annualized_return(result, periods_per_year),
        cagr=cagr(result),
        volatility=volatility(result, periods_per_year),
        sharpe=sharpe_ratio(result, risk_free_returns, periods_per_year),
        sortino=sortino_ratio(result, risk_free_returns, periods_per_year),
        calmar=calmar_ratio(result, periods_per_year),
        max_drawdown=max_drawdown(result.portfolio_value),
        var=historical_var(returns, var_level),
        cvar=historical_cvar(returns, var_level),
        mean_daily_turnover=mean_daily_turnover(result),
        annualized_turnover=annualized_turnover(result, periods_per_year),
        transaction_cost_drag=transaction_cost_drag(result),
        win_rate=win_rate(returns),
        information_ratio=information_ratio(result, benchmark_returns, periods_per_year),
        final_apv=final_apv(result),
        total_return_ci=block_bootstrap_ci(returns, _total_return_stat, bootstrap_n, bootstrap_block, var_level, seed),
        sharpe_ci=block_bootstrap_ci(returns, _sharpe_stat, bootstrap_n, bootstrap_block, var_level, seed),
    )
