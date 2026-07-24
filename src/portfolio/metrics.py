"""
metrics.py -- backtest evaluation (proposal §8): annualized return, Sharpe,
deflated Sharpe (Bailey & Lopez de Prado), max drawdown, turnover/holding-period
stats, regime-sliced performance (SELIC median split + named crisis windows).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS_PER_YEAR = 252

# Approximate, commonly-referenced windows (proposal §1.1's own examples) --
# not sourced from a per-ticker crisis-dating table, just fixed calendar ranges.
CRISIS_WINDOWS = {
    "gfc_2008": ("2008-09-01", "2009-03-31"),
    "recession_2015_16": ("2015-01-01", "2016-12-31"),
    "covid_2020": ("2020-02-01", "2020-06-30"),
}


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """returns: simple per-period returns (not log)."""
    n = len(returns)
    if n == 0:
        return float("nan")
    gross = (1 + returns).prod()
    return gross ** (periods_per_year / n) - 1


def sharpe_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    std = returns.std(ddof=1)
    if not std or np.isnan(std):
        return float("nan")
    return (returns.mean() / std) * np.sqrt(periods_per_year)


def max_drawdown(cum_values: pd.Series) -> float:
    """cum_values: a cumulative equity curve (levels, not returns)."""
    running_max = cum_values.cummax()
    return (cum_values / running_max - 1).min()


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int = 1, sr_benchmark: float = 0.0) -> float:
    """Probabilistic/Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): the
    probability the TRUE Sharpe exceeds `sr_benchmark`, corrected for non-normal
    returns (skew/kurtosis) and -- if n_trials > 1 -- for the multiple-testing
    inflation of having tried that many configurations. A probability in
    [0, 1], not a ratio; >0.95 is the usual bar. n_trials=1 (default) reduces
    this to the plain Probabilistic Sharpe Ratio (PSR) against `sr_benchmark`.
    """
    n = len(returns)
    sr_hat = returns.mean() / returns.std(ddof=1)
    skew = returns.skew()
    kurt = returns.kurtosis() + 3  # pandas reports EXCESS kurtosis; formula wants the raw 4th moment

    base_term = 1 - skew * sr_hat + ((kurt - 1) / 4) * sr_hat ** 2

    if n_trials > 1:
        euler_mascheroni = 0.5772156649015329
        var_sr = base_term / (n - 1)
        sr0 = np.sqrt(var_sr) * (
            (1 - euler_mascheroni) * norm.ppf(1 - 1 / n_trials)
            + euler_mascheroni * norm.ppf(1 - 1 / (n_trials * np.e))
        )
    else:
        sr0 = sr_benchmark

    denom = np.sqrt(base_term)
    if not denom or np.isnan(denom):
        return float("nan")
    z_score = (sr_hat - sr0) * np.sqrt(n - 1) / denom
    return norm.cdf(z_score)


def turnover_stats(rebalance_log: pd.DataFrame, rebalances_per_year: int = 4) -> dict:
    """rebalance_log needs a `turnover` column defined as sum(|Δw_i|) over
    equities per rebalance (backtest.run_backtest's convention -- a full
    one-way book replacement reads as turnover=2, since it sums both the
    100% sold and the 100% bought)."""
    turnover = rebalance_log["turnover"]
    annual_turnover = turnover.mean() * rebalances_per_year
    avg_holding_years = (2 / annual_turnover) if annual_turnover > 0 else float("inf")
    return {
        "annual_turnover": annual_turnover,
        "avg_holding_period_years": avg_holding_years,
        "no_trade_fraction": float((turnover < 1e-9).mean()),
    }


def _slice_summary(returns: pd.Series) -> dict:
    if len(returns) == 0:
        return {"n_days": 0, "annualized_return": float("nan"), "sharpe": float("nan")}
    return {
        "n_days": len(returns),
        "annualized_return": annualized_return(returns),
        "sharpe": sharpe_ratio(returns),
    }


def regime_slice(returns: pd.Series, selic_daily: pd.Series) -> dict:
    """returns and selic_daily both indexed by date. Splits by SELIC median
    over the backtest span (`selic_daily` in the dataset's raw %/day units --
    the split point is invariant to that unit choice), plus the named crisis
    windows above."""
    aligned_selic = selic_daily.reindex(returns.index).ffill()
    median = aligned_selic.median()
    result = {
        "high_selic": _slice_summary(returns[aligned_selic > median]),
        "low_selic": _slice_summary(returns[aligned_selic <= median]),
    }
    for name, (start, end) in CRISIS_WINDOWS.items():
        window = returns[(returns.index >= start) & (returns.index <= end)]
        result[name] = _slice_summary(window)
    return result


def full_report(equity_curve: pd.Series, rebalance_log: pd.DataFrame,
                 selic_daily: pd.Series = None, n_trials: int = 1) -> dict:
    """The full §8 metric panel for one strategy's equity curve."""
    returns = equity_curve.pct_change().dropna()
    report = {
        "annualized_return": annualized_return(returns),
        "sharpe": sharpe_ratio(returns),
        "deflated_sharpe": deflated_sharpe_ratio(returns, n_trials=n_trials),
        "max_drawdown": max_drawdown(equity_curve),
        **turnover_stats(rebalance_log),
    }
    if selic_daily is not None:
        report["regime_slices"] = regime_slice(returns, selic_daily)
    return report
