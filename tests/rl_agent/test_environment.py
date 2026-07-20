"""
Test: environment.py's mu solver (eq. 14, Theorem 1), weight drift (eq. 7),
and the shared backtest loop (docs/eiie_agent/EIIE_AGENT_PLAN.md Phase 4). Synthetic
data only.

Run from project root:
    python tests/rl_agent/test_environment.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.rl_agent.data import CASH_GIDX, GlobalAssetIndex, PricePanel  # noqa: E402
from src.rl_agent.environment import (  # noqa: E402
    drift_weights,
    run_backtest,
    solve_mu,
    solve_mu_torch,
)
from test_utils import print_check, print_header, print_section_end  # noqa: E402

C_SELL = C_BUY = 0.0003


def _f(mu, w_prime, w_target, c_sell, c_buy):
    """Independent re-implementation of eq. 14's RHS, f(mu) -- used to
    verify solve_mu against a root found by bisection, not by re-running
    solve_mu's own iteration (that would be circular)."""
    w0_prime, w0_target = w_prime[CASH_GIDX], w_target[CASH_GIDX]
    non_cash_prime = np.delete(w_prime, CASH_GIDX)
    non_cash_target = np.delete(w_target, CASH_GIDX)
    denom = 1.0 - c_buy * w0_target
    const = 1.0 - c_buy * w0_prime
    csum_coef = c_sell + c_buy - c_sell * c_buy
    pos = np.clip(non_cash_prime - mu * non_cash_target, 0.0, None)
    return (const - csum_coef * pos.sum()) / denom


def _bisect_root(w_prime, w_target, c_sell, c_buy, tol=1e-12):
    """mu - f(mu) is monotonically decreasing on [0, 1] (f is increasing,
    Lemma A.1) with a unique root -- find it independently via bisection."""
    lo, hi = 0.0, 1.0
    g = lambda mu: mu - _f(mu, w_prime, w_target, c_sell, c_buy)
    g_lo, g_hi = g(lo), g(hi)
    assert g_lo <= 0 <= g_hi or g_hi <= 0 <= g_lo, "no sign change -- bisection precondition violated"
    for _ in range(100):
        mid = (lo + hi) / 2
        if (g(mid) > 0) == (g_lo > 0):
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def test_drift_weights(passed, failed):
    y_t = np.array([1.0, 2.0, 0.5])
    w_prev = np.array([0.2, 0.3, 0.5])
    w_drift = drift_weights(y_t, w_prev)
    expected_unnorm = np.array([0.2, 0.6, 0.25])
    expected = expected_unnorm / expected_unnorm.sum()

    ok = np.allclose(w_drift, expected)
    print_check("drift_weights: matches eq. 7 by hand", ok, str(w_drift))
    passed, failed = passed + ok, failed + (not ok)

    ok = np.isclose(w_drift.sum(), 1.0)
    print_check("drift_weights: output sums to 1", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_drift_weights_nan_column_safe(passed, failed):
    """A global-space column with no price data yet (NaN in y_t) must not
    poison the drift-weight normalization as long as its own weight is 0 --
    0*NaN=NaN would otherwise propagate into every other column's result."""
    y_t = np.array([1.0, 2.0, np.nan])
    w_prev = np.array([0.4, 0.6, 0.0])
    w_drift = drift_weights(y_t, w_prev)
    expected_unnorm = np.array([0.4, 1.2, 0.0])
    expected = expected_unnorm / expected_unnorm.sum()

    ok = np.allclose(w_drift, expected) and not np.any(np.isnan(w_drift))
    print_check("drift_weights: NaN column with zero weight doesn't poison the result", ok, str(w_drift))
    passed, failed = passed + ok, failed + (not ok)

    y_t_torch = torch.tensor(y_t, dtype=torch.float32).unsqueeze(0)
    w_prev_torch = torch.tensor(w_prev, dtype=torch.float32).unsqueeze(0)
    from src.rl_agent.environment import drift_weights_torch
    w_drift_torch = drift_weights_torch(y_t_torch, w_prev_torch).squeeze(0).numpy()
    ok = np.allclose(w_drift_torch, expected) and not np.any(np.isnan(w_drift_torch))
    print_check("drift_weights_torch: same NaN-safety", ok, str(w_drift_torch))
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_solve_mu_zero_trade(passed, failed):
    w = np.array([0.3, 0.3, 0.4])
    mu = solve_mu(w, w, C_SELL, C_BUY)
    ok = np.isclose(mu, 1.0, atol=1e-8)
    print_check("solve_mu: zero-trade (w_prime == w_target) => mu = 1", ok, f"got {mu}")
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_solve_mu_vs_bisection(passed, failed):
    rng = np.random.default_rng(0)
    all_ok = True
    for _ in range(20):
        raw1, raw2 = rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(4))
        mu_solver = solve_mu(raw1, raw2, C_SELL, C_BUY)
        mu_bisect = _bisect_root(raw1, raw2, C_SELL, C_BUY)
        if not np.isclose(mu_solver, mu_bisect, atol=1e-6):
            all_ok = False
    print_check("solve_mu: matches an independently-derived bisection root on 20 random cases", all_ok)
    passed, failed = passed + all_ok, failed + (not all_ok)
    return passed, failed


def test_solve_mu_bounds(passed, failed):
    rng = np.random.default_rng(1)
    in_bounds = True
    for _ in range(50):
        w1, w2 = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))
        mu = solve_mu(w1, w2, C_SELL, C_BUY)
        if not (0.0 < mu <= 1.0 + 1e-9):
            in_bounds = False
    print_check("solve_mu: mu in (0, 1] across 50 random cases (Theorem 1)", in_bounds)
    passed, failed = passed + in_bounds, failed + (not in_bounds)
    return passed, failed


def test_differentiable_mu_matches_solver(passed, failed):
    rng = np.random.default_rng(2)
    w1 = rng.dirichlet(np.ones(4), size=16)
    w2 = rng.dirichlet(np.ones(4), size=16)
    mu_np = np.array([solve_mu(w1[i], w2[i], C_SELL, C_BUY) for i in range(16)])
    mu_torch = solve_mu_torch(torch.tensor(w1), torch.tensor(w2), C_SELL, C_BUY, k=50).numpy()

    ok = np.allclose(mu_np, mu_torch, atol=1e-4)
    print_check("solve_mu_torch (k=50) matches the converged numpy solver", ok,
                f"max abs diff = {np.abs(mu_np - mu_torch).max():.2e}")
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_differentiable_mu_gradient(passed, failed):
    """Finite-difference gradient check: d(mu)/d(w_target) from autograd
    must match a numeric finite-difference estimate."""
    torch.manual_seed(0)
    w_prime = torch.tensor([[0.3, 0.3, 0.4]], dtype=torch.float64)
    w_target = torch.tensor([[0.5, 0.2, 0.3]], dtype=torch.float64, requires_grad=True)

    mu = solve_mu_torch(w_prime, w_target, C_SELL, C_BUY, k=1)
    mu.sum().backward()
    analytic_grad = w_target.grad.clone().numpy()[0]

    eps = 1e-6
    numeric_grad = np.zeros(3)
    for i in range(3):
        w_plus = w_target.detach().clone()
        w_plus[0, i] += eps
        w_minus = w_target.detach().clone()
        w_minus[0, i] -= eps
        mu_plus = solve_mu_torch(w_prime, w_plus, C_SELL, C_BUY, k=1).item()
        mu_minus = solve_mu_torch(w_prime, w_minus, C_SELL, C_BUY, k=1).item()
        numeric_grad[i] = (mu_plus - mu_minus) / (2 * eps)

    ok = np.allclose(analytic_grad, numeric_grad, atol=1e-4)
    print_check("solve_mu_torch: autograd gradient matches finite-difference", ok,
                f"analytic={analytic_grad}, numeric={numeric_grad}")
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def _toy_panel(close_a_path, T=5):
    """3-column global space: cash, asset A (path given), asset B (flat)."""
    asset_index = GlobalAssetIndex(tickers=("AAA", "BBB"), ticker_to_gidx={"AAA": 1, "BBB": 2})
    dates = pd.bdate_range("2020-01-01", periods=T)
    close = np.column_stack([np.ones(T), close_a_path, np.full(T, 10.0)])
    return PricePanel(
        asset_index=asset_index, dates=dates, close=close, high=close.copy(), low=close.copy(),
        cdi_factor=np.ones(T),  # zero risk-free rate, isolates cost effects
        slot_gidx=np.array([[1, 2]] * T), valid=np.array([[True, True]] * T),
        window=2, start_idx=1, end_idx=T - 1,
    )


def _toy_panel_with_phantom(close_a_path, T=5):
    """Like _toy_panel, but with a 4th global column (CCC) that's never in
    slot_gidx (never holdable, e.g. a ticker whose IPO postdates this
    truncated window) and has all-NaN prices throughout. Isolates the
    run_backtest growth dot-product's NaN guard specifically (drift_weights'
    own guard is tested separately above and would mask this if the panel
    only had 3 columns)."""
    asset_index = GlobalAssetIndex(tickers=("AAA", "BBB", "CCC"),
                                    ticker_to_gidx={"AAA": 1, "BBB": 2, "CCC": 3})
    dates = pd.bdate_range("2020-01-01", periods=T)
    close = np.column_stack([np.ones(T), close_a_path, np.full(T, 10.0), np.full(T, np.nan)])
    return PricePanel(
        asset_index=asset_index, dates=dates, close=close, high=close.copy(), low=close.copy(),
        cdi_factor=np.ones(T),
        slot_gidx=np.array([[1, 2]] * T), valid=np.array([[True, True]] * T),  # CCC (gidx 3) never referenced
        window=2, start_idx=1, end_idx=T - 1,
    )


def test_backtest_nan_column_safe(passed, failed):
    """A never-holdable global column with all-NaN prices (e.g. a ticker
    whose IPO postdates a truncated window_end) must not poison the
    backtest's growth dot-product, exactly the same 0*NaN=NaN hazard as
    drift_weights, but in run_backtest's OWN growth calc (a separate,
    previously unguarded site)."""
    panel = _toy_panel_with_phantom(close_a_path=np.full(5, 10.0))

    def hold(t, w_prev, w_drift, panel):
        return w_drift  # never trade; w_drift is already global-space width (cash, AAA, BBB, CCC=0)

    result = run_backtest(panel, hold, C_SELL, C_BUY)
    ok = bool(np.all(np.isfinite(result.portfolio_value))) and np.allclose(result.portfolio_value, 1.0)
    print_check("backtest: an all-NaN never-held column doesn't poison portfolio value", ok,
                str(result.portfolio_value))
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_backtest_no_trade_zero_cost(passed, failed):
    """Constant prices + weight_fn holds the drifted weights unchanged
    (no rebalancing) => zero turnover, mu == 1, portfolio value constant."""
    panel = _toy_panel(close_a_path=np.full(5, 10.0))  # A constant too -> y_t == 1 every day

    def hold(t, w_prev, w_drift, panel):
        return w_drift  # never trade

    result = run_backtest(panel, hold, C_SELL, C_BUY)
    ok = np.allclose(result.portfolio_value, 1.0)
    print_check("backtest: constant prices + no trading => portfolio value stays exactly 1.0",
                ok, str(result.portfolio_value))
    passed, failed = passed + ok, failed + (not ok)

    ok = np.allclose(result.mu, 1.0) and np.allclose(result.turnover, 0.0)
    print_check("backtest: no trade => mu = 1, turnover = 0 every day", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_backtest_costs_reduce_value(passed, failed):
    """Constant prices (zero underlying growth) but weight_fn forces a trade
    every day by flip-flopping allocation => costs must strictly reduce
    portfolio value below 1.0."""
    panel = _toy_panel(close_a_path=np.full(5, 10.0))

    def flip_flop(t, w_prev, w_drift, panel):
        if t % 2 == 0:
            return np.array([0.1, 0.8, 0.1])
        return np.array([0.1, 0.1, 0.8])

    result = run_backtest(panel, flip_flop, C_SELL, C_BUY)
    ok = result.portfolio_value[-1] < 1.0
    print_check("backtest: forced trading under zero growth strictly reduces portfolio value",
                ok, f"final={result.portfolio_value[-1]:.6f}")
    passed, failed = passed + ok, failed + (not ok)

    ok = bool(np.all(result.mu < 1.0)) and bool(np.all(result.turnover > 0))
    print_check("backtest: every forced-trade day has mu < 1 and turnover > 0", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_backtest_buy_and_hold_growth(passed, failed):
    """Asset A is flat through day 1, then doubles once, then stays flat;
    weight_fn buys 100% A on day 1 (start_idx) and never rebalances again.
    The agent only owns A starting the period AFTER it decides to buy (eq.
    19: a_t = w_t is applied to the NEXT period's price move), so the
    capture of the doubling -- and its APV -- must reflect exactly that
    one-period lag, net of the initial trade's cost (mu[0], at t=start_idx)."""
    close_a = np.array([10.0, 10.0, 20.0, 20.0, 20.0])
    panel = _toy_panel(close_a_path=close_a)

    def buy_a_once(t, w_prev, w_drift, panel):
        if t == panel.start_idx:
            return np.array([0.0, 1.0, 0.0])
        return w_drift  # hold thereafter

    result = run_backtest(panel, buy_a_once, C_SELL, C_BUY)
    ok = np.isclose(result.portfolio_value[-1], 2.0 * result.mu[0], atol=1e-6)
    print_check("backtest: buy-and-hold APV matches the one real price move, net of the initial trade cost",
                ok, f"final={result.portfolio_value[-1]:.6f}, mu[0]={result.mu[0]:.6f}")
    passed, failed = passed + ok, failed + (not ok)

    ok = len(result.dates) == len(result.log_returns) == len(result.mu) == 4
    ok = ok and len(result.portfolio_value) == 5
    print_check("backtest: array shapes consistent (T dates/returns, T+1 portfolio_value)", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def main():
    print_header("test_environment")
    passed = failed = 0

    passed, failed = test_drift_weights(passed, failed)
    passed, failed = test_drift_weights_nan_column_safe(passed, failed)
    passed, failed = test_solve_mu_zero_trade(passed, failed)
    passed, failed = test_solve_mu_vs_bisection(passed, failed)
    passed, failed = test_solve_mu_bounds(passed, failed)
    passed, failed = test_differentiable_mu_matches_solver(passed, failed)
    passed, failed = test_differentiable_mu_gradient(passed, failed)
    passed, failed = test_backtest_nan_column_safe(passed, failed)
    passed, failed = test_backtest_no_trade_zero_cost(passed, failed)
    passed, failed = test_backtest_costs_reduce_value(passed, failed)
    passed, failed = test_backtest_buy_and_hold_growth(passed, failed)

    print_section_end(passed, failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
