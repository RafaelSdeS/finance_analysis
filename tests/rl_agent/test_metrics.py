"""
Test: metrics.py's performance metrics and block-bootstrap CIs, checked
against hand-computed values (docs/EIIE_AGENT_PLAN.md Phase 4). Synthetic
data only.

Run from project root:
    python tests/rl_agent/test_metrics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.rl_agent import metrics as m  # noqa: E402
from src.rl_agent.environment import BacktestResult  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402


def _result(log_returns, mu=None, turnover=None, start="2020-01-01"):
    log_returns = np.asarray(log_returns, dtype=float)
    T = len(log_returns)
    pv = np.empty(T + 1)
    pv[0] = 1.0
    for i in range(T):
        pv[i + 1] = pv[i] * np.exp(log_returns[i])
    return BacktestResult(
        dates=pd.bdate_range(start, periods=T),
        portfolio_value=pv,
        log_returns=log_returns,
        mu=np.ones(T) if mu is None else np.asarray(mu, dtype=float),
        turnover=np.zeros(T) if turnover is None else np.asarray(turnover, dtype=float),
        cost=np.zeros(T),
        weights=np.zeros((T, 1)),
    )


def test_return_metrics(passed, failed):
    # 10 days of exactly 1% simple daily return each -> total return = 1.01^10 - 1
    r = np.full(10, np.log(1.01))
    result = _result(r)

    expected_total = 1.01 ** 10 - 1
    ok = np.isclose(m.total_return(result), expected_total)
    print_check("total_return: matches 1.01^10 - 1 by hand", ok, f"got {m.total_return(result):.6f}")
    passed, failed = passed + ok, failed + (not ok)

    ok = np.isclose(m.final_apv(result), 1.01 ** 10)
    print_check("final_apv: matches 1.01^10", ok)
    passed, failed = passed + ok, failed + (not ok)

    expected_ann = (1.01 ** 10) ** (252 / 10) - 1
    ok = np.isclose(m.annualized_return(result), expected_ann)
    print_check("annualized_return: trading-day compounding matches (fAPV)^(252/T)-1", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_volatility(passed, failed):
    r = np.array([0.01, -0.01, 0.01, -0.01, 0.01])
    result = _result(r)
    expected = r.std(ddof=1) * np.sqrt(252)
    ok = np.isclose(m.volatility(result), expected)
    print_check("volatility: log-return std * sqrt(252) matches by hand", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_max_drawdown(passed, failed):
    # pv path: 1 -> 1.2 (peak) -> 0.9 (trough, -25% from peak) -> 1.0
    log_r = np.log([1.2, 0.75, 1.0 / 0.9])
    result = _result(log_r)
    ok = np.isclose(result.portfolio_value[2], 0.9, atol=1e-9)
    print_check("max_drawdown setup: trough value is 0.9 as constructed", ok, str(result.portfolio_value))
    passed, failed = passed + ok, failed + (not ok)

    mdd = m.max_drawdown(result.portfolio_value)
    ok = np.isclose(mdd, 0.25, atol=1e-9)
    print_check("max_drawdown: (1.2 - 0.9) / 1.2 = 0.25 by hand", ok, f"got {mdd:.6f}")
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_sharpe_sortino(passed, failed):
    # log returns for simple returns [0.02, -0.01, 0.03, -0.02, 0.01]
    simple = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    log_r = np.log(1 + simple)
    result = _result(log_r)
    rf = np.zeros(5)

    excess = simple - rf
    expected_sharpe = excess.mean() / excess.std(ddof=1) * np.sqrt(252)
    ok = np.isclose(m.sharpe_ratio(result, rf), expected_sharpe)
    print_check("sharpe_ratio: matches hand-computed E[rho-rf]/std(rho-rf)*sqrt(252)", ok)
    passed, failed = passed + ok, failed + (not ok)

    downside = excess[excess < 0]
    expected_sortino = excess.mean() / downside.std(ddof=1) * np.sqrt(252)
    ok = np.isclose(m.sortino_ratio(result, rf), expected_sortino)
    print_check("sortino_ratio: matches hand-computed downside-deviation formula", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_var_cvar(passed, failed):
    # 11 points so (n-1)*(1-level) = 10*0.10 = 1.0 exactly -> quantile lands
    # precisely on sorted[1], no interpolation ambiguity to hand-compute around.
    returns = np.array([-0.06, -0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10])
    var95 = m.historical_var(returns, level=0.90)
    ok = np.isclose(var95, 0.05, atol=1e-9)
    print_check("historical_var: 90% VaR on 11 sorted obs = -sorted[1] = 0.05", ok, f"got {var95}")
    passed, failed = passed + ok, failed + (not ok)

    cvar95 = m.historical_cvar(returns, level=0.90)
    ok = cvar95 >= var95 - 1e-9
    print_check("historical_cvar: CVaR >= VaR (tail mean at least as bad as the threshold)", ok,
                f"var={var95}, cvar={cvar95}")
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_turnover_and_cost_drag(passed, failed):
    result = _result(np.zeros(4), mu=[1.0, 0.99, 1.0, 0.98], turnover=[0.0, 0.2, 0.0, 0.3])
    ok = np.isclose(m.mean_daily_turnover(result), 0.5 / 4)
    print_check("mean_daily_turnover: matches mean of the turnover array", ok)
    passed, failed = passed + ok, failed + (not ok)

    expected_drag = 1.0 - (1.0 * 0.99 * 1.0 * 0.98)
    ok = np.isclose(m.transaction_cost_drag(result), expected_drag)
    print_check("transaction_cost_drag: 1 - prod(mu) matches by hand", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_win_rate_and_information_ratio(passed, failed):
    simple = np.array([0.01, -0.01, 0.02, -0.005, 0.0])
    log_r = np.log(1 + simple)
    result = _result(log_r)
    ok = np.isclose(m.win_rate(simple), 2 / 5)
    print_check("win_rate: fraction of strictly positive returns", ok, f"got {m.win_rate(simple)}")
    passed, failed = passed + ok, failed + (not ok)

    bench = np.zeros(5)
    expected_ir = simple.mean() / simple.std(ddof=1) * np.sqrt(252)
    ok = np.isclose(m.information_ratio(result, bench), expected_ir)
    print_check("information_ratio: matches hand-computed active-return formula vs a zero benchmark", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_bootstrap_ci(passed, failed):
    rng_returns = np.random.default_rng(0).normal(0.0005, 0.01, size=500)

    def total_ret(r):
        return float(np.prod(1 + r) - 1)

    point, lo, hi = m.block_bootstrap_ci(rng_returns, total_ret, n_bootstrap=200, block_size=10, seed=42)
    ok = lo <= point <= hi
    print_check("block_bootstrap_ci: point estimate falls within its own CI", ok, f"[{lo:.4f}, {point:.4f}, {hi:.4f}]")
    passed, failed = passed + ok, failed + (not ok)

    point2, lo2, hi2 = m.block_bootstrap_ci(rng_returns, total_ret, n_bootstrap=200, block_size=10, seed=42)
    ok = (lo, point, hi) == (lo2, point2, hi2)
    print_check("block_bootstrap_ci: deterministic given a fixed seed", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_summarize_smoke(passed, failed):
    rng = np.random.default_rng(3)
    log_r = np.log(1 + rng.normal(0.0003, 0.01, size=100))
    mu = np.clip(1 - np.abs(rng.normal(0.0005, 0.0003, size=100)), 0.9, 1.0)
    turnover = np.abs(rng.normal(0.05, 0.02, size=100))
    result = _result(log_r, mu=mu, turnover=turnover)
    rf = np.full(100, 0.0001)
    bench = rng.normal(0.0002, 0.008, size=100)

    summary = m.summarize(result, rf, bench, bootstrap_n=50, seed=7)
    fields = [summary.total_return, summary.annualized_return, summary.cagr, summary.volatility,
              summary.sharpe, summary.sortino, summary.calmar, summary.max_drawdown, summary.var,
              summary.cvar, summary.mean_daily_turnover, summary.annualized_turnover,
              summary.transaction_cost_drag, summary.win_rate, summary.information_ratio, summary.final_apv]
    ok = all(np.isfinite(f) for f in fields)
    print_check("summarize: every metric is finite on a realistic synthetic series", ok,
                str([round(f, 4) for f in fields]))
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def main():
    print_header("test_metrics")
    passed = failed = 0

    passed, failed = test_return_metrics(passed, failed)
    passed, failed = test_volatility(passed, failed)
    passed, failed = test_max_drawdown(passed, failed)
    passed, failed = test_sharpe_sortino(passed, failed)
    passed, failed = test_var_cvar(passed, failed)
    passed, failed = test_turnover_and_cost_drag(passed, failed)
    passed, failed = test_win_rate_and_information_ratio(passed, failed)
    passed, failed = test_bootstrap_ci(passed, failed)
    passed, failed = test_summarize_smoke(passed, failed)

    print_section_end(passed, failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
