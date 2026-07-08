"""
Portfolio performance metrics.

Shared by the training callback (validation Sharpe for early stopping)
and the evaluation/backtesting module. All functions take a 1-D array of
daily log returns unless stated otherwise.
"""

import numpy as np

TRADING_DAYS = 252
EPS = 1e-12


def sharpe_ratio(log_returns: np.ndarray) -> float:
    """Annualized Sharpe ratio from daily log returns (risk-free rate 0)."""
    r = np.asarray(log_returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    return float(r.mean() / (r.std() + EPS) * np.sqrt(TRADING_DAYS))


def sortino_ratio(log_returns: np.ndarray) -> float:
    """Annualized Sortino ratio: penalizes downside deviation only."""
    r = np.asarray(log_returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    downside_std = downside.std() if len(downside) > 0 else EPS
    return float(r.mean() / (downside_std + EPS) * np.sqrt(TRADING_DAYS))


def max_drawdown(portfolio_values: np.ndarray) -> float:
    """Maximum peak-to-trough decline, as a positive fraction (0.25 = -25%)."""
    v = np.asarray(portfolio_values, dtype=np.float64)
    if len(v) < 2:
        return 0.0
    peaks = np.maximum.accumulate(v)
    drawdowns = (peaks - v) / peaks
    return float(drawdowns.max())


def cumulative_return(portfolio_values: np.ndarray) -> float:
    """Total return over the period: (V_final - V_0) / V_0."""
    v = np.asarray(portfolio_values, dtype=np.float64)
    if len(v) < 2:
        return 0.0
    return float(v[-1] / v[0] - 1.0)


def annualized_return(portfolio_values: np.ndarray) -> float:
    """Geometric annualized return from a daily value series."""
    v = np.asarray(portfolio_values, dtype=np.float64)
    if len(v) < 2 or v[0] <= 0:
        return 0.0
    years = (len(v) - 1) / TRADING_DAYS
    return float((v[-1] / v[0]) ** (1.0 / max(years, EPS)) - 1.0)


def win_rate(log_returns: np.ndarray) -> float:
    """Fraction of days with positive return."""
    r = np.asarray(log_returns, dtype=np.float64)
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def effective_n_positions(weights: np.ndarray) -> float:
    """Effective number of positions (1/HHI) averaged over the period."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2 or len(w) == 0:
        return 0.0
    hhi = (w ** 2).sum(axis=1)
    return float(np.mean(1.0 / np.maximum(hhi, EPS)))


def compute_all(log_returns: np.ndarray, portfolio_values: np.ndarray, weights: np.ndarray | None = None,
                daily_costs: np.ndarray | None = None, cost_bps: float | None = None) -> dict:
    """All metrics in one dict (for metrics.json / comparison tables).

    Args:
        log_returns: daily log returns
        portfolio_values: daily portfolio values (length = len(log_returns) + 1, including initial capital)
        weights: optional [days, assets] weight matrix
        daily_costs: optional [days] transaction cost per day (cost on rebalance day, 0 elsewhere)
        cost_bps: optional cost basis points; if given with weights, override turnover calculation
    """
    metrics = {
        "cumulative_return": cumulative_return(portfolio_values),
        "annualized_return": annualized_return(portfolio_values),
        "sharpe": sharpe_ratio(log_returns),
        "sortino": sortino_ratio(log_returns),
        "max_drawdown": max_drawdown(portfolio_values),
        "win_rate": win_rate(log_returns),
        "n_days": int(len(log_returns)),
    }

    # Gross Sharpe and cost drag (if daily_costs provided)
    if daily_costs is not None:
        daily_costs = np.asarray(daily_costs, dtype=np.float64)
        # Additive approximation: gross log return ≈ net log return + cost
        gross_log_rets = log_returns + daily_costs
        metrics["gross_sharpe"] = sharpe_ratio(gross_log_rets)
        metrics["annualized_cost_drag"] = float(daily_costs.mean() * TRADING_DAYS)

    if weights is not None:
        metrics["effective_n"] = effective_n_positions(weights)
        # Concentration: mean of daily max weight (0.004 for equal weight, >0.01 for conviction)
        metrics["max_weight"] = float(weights.max(axis=1).mean())

        # Turnover: override if cost_bps given (more accurate than diff-based)
        if cost_bps is not None and daily_costs is not None:
            # Recover one-way turnover from cost: cost = turnover * (bps / 1e4)
            daily_costs_clean = np.asarray(daily_costs, dtype=np.float64)
            # Cost on rebalance days; derive turnover
            rebalance_days = np.where(daily_costs_clean > 0)[0]
            if len(rebalance_days) > 0:
                # Average turnover per rebalance day
                rebalance_turnover = daily_costs_clean[rebalance_days] / (cost_bps / 1e4)
                # Daily average (turnover concentrated on rebalance days)
                metrics["avg_daily_turnover"] = float(rebalance_turnover.mean() / 1)  # per-day on rebalance
            else:
                metrics["avg_daily_turnover"] = 0.0
        else:
            # Fallback: turnover from weight diffs (less accurate on drifted weights, but works)
            w = np.asarray(weights, dtype=np.float64)
            daily_turnover = np.abs(np.diff(w, axis=0)).sum(axis=1)
            metrics["avg_daily_turnover"] = float(daily_turnover.mean())

    return metrics
