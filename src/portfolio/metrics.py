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


_DEGENERATE_STD = 1e-9  # below genuine daily-return noise (~1e-4); catches float64
                         # accumulation noise (~1e-16) on a self-vs-self diff (e.g.
                         # CDI curve minus its own CDI series), not real near-zero vol


def sharpe_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    std = returns.std(ddof=1)
    if not std or np.isnan(std) or std < _DEGENERATE_STD:
        return float("nan")
    return (returns.mean() / std) * np.sqrt(periods_per_year)


def excess_over_cdi_sharpe(returns: pd.Series, cdi_daily: pd.Series,
                            periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Sharpe of (returns - CDI) -- plain sharpe_ratio() has no risk-free term
    (confirmed 2026-07-25: it's why 100% CDI itself prints Sharpe ~42, not
    ~0), so it answers "is this distinguishable from zero", not "does this
    beat cash". This is the metric that answers the second question.
    `cdi_daily`: %/trading-day (manifest.COLUMN_UNITS convention, e.g. 0.0534
    = 0.0534%/day), same series backtest.run_backtest uses for cash accrual."""
    cdi_ret = cdi_daily.reindex(returns.index).ffill() / 100
    return sharpe_ratio((returns - cdi_ret).dropna(), periods_per_year)


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series,
                       periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Sharpe of (returns - benchmark) -- e.g. vs the equal-weight floor.
    Isolates whatever the extra machinery (alpha model + optimizer) adds on
    top of shared market beta, which annualized_return/sharpe_ratio alone
    conflate with it."""
    active = (returns - benchmark_returns.reindex(returns.index)).dropna()
    return sharpe_ratio(active, periods_per_year)


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


def active_return_report(returns: pd.Series, benchmark_returns: pd.Series,
                          selic_daily: pd.Series) -> dict:
    """Regime breakdown of (returns - benchmark), e.g. pipeline minus
    equal-weight -- same regime_slice() slicing, applied to the ACTIVE return
    instead of the raw one, so "which regime does the extra machinery help
    or hurt in" is a direct table read, not a mental subtraction across two
    separate regime tables (plan Phase 0.3)."""
    active = (returns - benchmark_returns.reindex(returns.index)).dropna()
    return regime_slice(active, selic_daily)


def full_report(equity_curve: pd.Series, rebalance_log: pd.DataFrame,
                 selic_daily: pd.Series = None, cdi_daily: pd.Series = None,
                 benchmark_returns: pd.Series = None, n_trials: int = 1,
                 rebalances_per_year: int = 4) -> dict:
    """The full §8 metric panel for one strategy's equity curve.
    `rebalances_per_year` MUST match the actual rebalance cadence the log
    came from (4 for quarterly, 12 for monthly, ...) -- turnover_stats()
    just multiplies mean per-event turnover by this number, so passing the
    wrong value silently mis-annualizes turnover/holding-period by that
    ratio without affecting anything else in the report (found empirically
    2026-07-25: every caller here used to rely on the 4-quarters-a-year
    default even when the backtest itself ran on a monthly membership
    calendar)."""
    returns = equity_curve.pct_change().dropna()
    report = {
        "annualized_return": annualized_return(returns),
        "sharpe": sharpe_ratio(returns),
        "deflated_sharpe": deflated_sharpe_ratio(returns, n_trials=n_trials),
        "max_drawdown": max_drawdown(equity_curve),
        **turnover_stats(rebalance_log, rebalances_per_year=rebalances_per_year),
    }
    if cdi_daily is not None:
        report["excess_cdi_sharpe"] = excess_over_cdi_sharpe(returns, cdi_daily)
    if benchmark_returns is not None:
        report["information_ratio"] = information_ratio(returns, benchmark_returns)
    if selic_daily is not None:
        report["regime_slices"] = regime_slice(returns, selic_daily)
    return report


def print_regime_slices(slices: dict) -> None:
    """The per-regime table body shared by print_report and any standalone
    regime table (e.g. active_return_report's output)."""
    for slice_name, s in slices.items():
        print(f"    {slice_name:<20}n={s['n_days']:>5}   "
              f"ann.ret={s['annualized_return']:>8.2%}   sharpe={s['sharpe']:>8.3f}")


def print_report(name: str, report: dict) -> None:
    """Human-readable replacement for pprint.pprint(full_report(...)) -- the
    raw dict prints numpy's `np.float64(...)` reprs and alphabetizes an
    otherwise logically-grouped set of metrics, which is tedious to scan on
    every backtest run. Same numbers, laid out as a fixed table instead."""
    holding = report["avg_holding_period_years"]
    holding_str = "never trades" if np.isinf(holding) else f"{holding:.2f} years"
    print(f"\n=== {name} ===")
    print(f"  {'Annualized return':<22}{report['annualized_return']:>10.2%}")
    print(f"  {'Sharpe ratio':<22}{report['sharpe']:>10.3f}")
    if "excess_cdi_sharpe" in report:
        print(f"  {'Excess-CDI Sharpe':<22}{report['excess_cdi_sharpe']:>10.3f}")
    if "information_ratio" in report:
        print(f"  {'Info ratio vs EW':<22}{report['information_ratio']:>10.3f}")
    print(f"  {'Deflated Sharpe':<22}{report['deflated_sharpe']:>10.3f}")
    print(f"  {'Max drawdown':<22}{report['max_drawdown']:>10.2%}")
    print(f"  {'Annual turnover':<22}{report['annual_turnover']:>9.2f}x")
    print(f"  {'Avg holding period':<22}{holding_str:>10}")
    print(f"  {'No-trade fraction':<22}{report['no_trade_fraction']:>10.1%}")

    slices = report.get("regime_slices")
    if slices:
        print("  Regime breakdown:")
        print_regime_slices(slices)
