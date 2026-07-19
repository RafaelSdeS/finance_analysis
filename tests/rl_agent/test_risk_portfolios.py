"""
Test: risk_portfolios.py's covariance estimator, min-variance/risk-parity
optimizers, vol-target overlay, and the full weight_fn harness integration
(RISK_MANDATE_IMPL_PLAN.md Sec 3.6). Synthetic, deterministic data only.

Run from project root:
    python tests/rl_agent/test_risk_portfolios.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.rl_agent.config import RiskConfig  # noqa: E402
from src.rl_agent.data import GlobalAssetIndex, PricePanel  # noqa: E402
from src.rl_agent.environment import run_backtest  # noqa: E402
from src.rl_agent.risk_portfolios import (  # noqa: E402
    _solve_with_fallback, eligible_mask, estimate_cov, make_risk_weight_fn,
    min_variance_weights, risk_parity_weights, trailing_returns, vol_target_overlay,
)
from test_utils import print_check, print_header, print_section_end  # noqa: E402

C_SELL = C_BUY = 0.0003


def _cov_2asset(sigma1=0.10, sigma2=0.20, rho=0.3) -> np.ndarray:
    cov12 = rho * sigma1 * sigma2
    return np.array([[sigma1 ** 2, cov12], [cov12, sigma2 ** 2]])


def _cov_diag(sigmas) -> np.ndarray:
    return np.diag(np.array(sigmas) ** 2)


def test_min_variance_2asset(passed, failed):
    cov = _cov_2asset()
    w, ok = min_variance_weights(cov)
    expected = np.array([0.8947368421, 0.1052631579])
    ok_val = ok and np.allclose(w, expected, atol=1e-5)
    print_check("min_variance 2-asset matches closed-form (0.894737, 0.105263)", ok_val, str(w))
    return passed + ok_val, failed + (not ok_val)


def test_erc_2asset(passed, failed):
    cov = _cov_2asset()
    w, ok = risk_parity_weights(cov)
    expected = np.array([2.0 / 3.0, 1.0 / 3.0])
    ok_val = ok and np.allclose(w, expected, atol=1e-5)
    print_check("risk_parity 2-asset matches closed-form (2/3, 1/3), any rho", ok_val, str(w))
    return passed + ok_val, failed + (not ok_val)


def test_min_variance_diag(passed, failed):
    cov = _cov_diag([0.1, 0.2, 0.4])
    w, ok = min_variance_weights(cov)
    expected = np.array([16, 4, 1]) / 21.0
    ok_val = ok and np.allclose(w, expected, atol=1e-5)
    print_check("min_variance diagonal-Sigma matches inverse-variance weights (16/21, 4/21, 1/21)",
                ok_val, str(w))
    return passed + ok_val, failed + (not ok_val)


def test_erc_diag(passed, failed):
    cov = _cov_diag([0.1, 0.2, 0.4])
    w, ok = risk_parity_weights(cov)
    expected = np.array([4, 2, 1]) / 7.0
    ok_val = ok and np.allclose(w, expected, atol=1e-5)
    print_check("risk_parity diagonal-Sigma matches inverse-vol weights (4/7, 2/7, 1/7)", ok_val, str(w))
    return passed + ok_val, failed + (not ok_val)


def test_erc_property_random(passed, failed):
    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, 50))
    cov = A @ A.T / 50 + 0.01 * np.eye(50)
    w, ok = risk_parity_weights(cov)
    rc = w * (cov @ w)
    ok_val = ok and np.all(w > 0) and np.allclose(rc, rc.mean(), rtol=1e-4)
    print_check("risk_parity: all 50 risk contributions equalized, all weights positive", ok_val,
                f"converged={ok}, RC range=[{rc.min():.6g}, {rc.max():.6g}]")
    return passed + ok_val, failed + (not ok_val)


def test_vol_target_overlay(passed, failed):
    cov = _cov_2asset()
    w_eq = np.array([0.6, 0.4])
    sigma_p = float(np.sqrt(w_eq @ cov @ w_eq))

    f, sp = vol_target_overlay(w_eq, cov, 0.5 * sigma_p)
    ok1 = np.isclose(f, 0.5, atol=1e-9) and np.isclose(sp, sigma_p)
    print_check("vol_target_overlay: half the realized vol -> f=0.5 exactly", ok1, f"f={f}")

    f2, _ = vol_target_overlay(w_eq, cov, 2.0 * sigma_p)
    ok2 = np.isclose(f2, 1.0)
    print_check("vol_target_overlay: target above realized vol -> f=1.0 (no cash top-up)", ok2, f"f={f2}")

    passed, failed = passed + ok1 + ok2, failed + (not ok1) + (not ok2)
    return passed, failed


def test_max_weight_cap(passed, failed):
    cov = _cov_2asset()
    w, ok = min_variance_weights(cov, max_weight=0.6)
    ok_val = ok and np.allclose(w, [0.6, 0.4], atol=1e-5)
    print_check("min_variance: binding max_weight=0.6 cap redistributes remainder", ok_val, str(w))
    return passed + ok_val, failed + (not ok_val)


def test_degenerate_cov(passed, failed):
    cov = np.diag([0.04, 0.09, 0.16, 0.25, 0.25]).astype(float)
    cov[3, 4] = cov[4, 3] = 0.25  # asset 4 == asset 5 exactly: rank-deficient

    w_mv, _ = _solve_with_fallback("min_variance", cov, RiskConfig(), None)
    w_rp, _ = _solve_with_fallback("risk_parity", cov, RiskConfig(), None)

    ok = (np.all(np.isfinite(w_mv)) and np.isclose(w_mv.sum(), 1.0) and np.all(w_mv >= -1e-9)
          and np.all(np.isfinite(w_rp)) and np.isclose(w_rp.sum(), 1.0) and np.all(w_rp >= -1e-9))
    print_check("degenerate (rank-deficient) Sigma: both solvers survive, finite simplex output",
                ok, f"mv={w_mv}, rp={w_rp}")
    return passed + ok, failed + (not ok)


def test_estimator_recovery(passed, failed):
    rng = np.random.default_rng(1)
    true_sigmas = np.array([0.01, 0.015, 0.02, 0.025])  # daily
    returns = rng.normal(0.0, true_sigmas, size=(2000, 4))
    cov = estimate_cov(returns, RiskConfig(cov_estimator="ledoit_wolf"))
    w, ok = min_variance_weights(cov)

    inv_var = 1.0 / true_sigmas ** 2
    expected = inv_var / inv_var.sum()
    ok_val = ok and np.allclose(w, expected, atol=0.05)
    print_check("LedoitWolf-estimated cov recovers analytic diag min-variance weights (atol=0.05)",
                ok_val, f"got {w}, expected {expected}")
    return passed + ok_val, failed + (not ok_val)


def _entrant_panel():
    """cash + AAA, BBB (established, real history throughout) + CCC (flat/
    backfilled through day 5, real trading from day 6). 14 days; backtest
    window is [6, 13], lookback=5, rebalance_every=2 -- CCC crosses the
    min_history_frac=0.8 seasoning threshold exactly at its 3rd rebalance
    (t=10), never before."""
    T = 14
    rng = np.random.default_rng(42)
    aaa = 10.0 * np.cumprod(1.0 + rng.normal(0.001, 0.01, T))
    bbb = 10.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.008, T))
    ccc = np.array([10, 10, 10, 10, 10, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14], dtype=float)

    asset_index = GlobalAssetIndex(tickers=("AAA", "BBB", "CCC"),
                                    ticker_to_gidx={"AAA": 1, "BBB": 2, "CCC": 3})
    dates = pd.bdate_range("2020-01-01", periods=T)
    close = np.column_stack([np.ones(T), aaa, bbb, ccc])
    return PricePanel(
        asset_index=asset_index, dates=dates, close=close, high=close.copy(), low=close.copy(),
        cdi_factor=np.full(T, 1.0002),
        slot_gidx=np.array([[1, 2, 3]] * T), valid=np.array([[True, True, True]] * T),
        window=2, start_idx=6, end_idx=13,
    )


def test_eligible_mask_entrant_seasoning(passed, failed):
    panel = _entrant_panel()
    cfg = RiskConfig(lookback=5, min_history_frac=0.8, rebalance_every=2)
    gidx = np.array([1, 2, 3])

    elig_t6 = eligible_mask(panel, 6, cfg.lookback, gidx, cfg.min_history_frac)
    elig_t8 = eligible_mask(panel, 8, cfg.lookback, gidx, cfg.min_history_frac)
    elig_t10 = eligible_mask(panel, 10, cfg.lookback, gidx, cfg.min_history_frac)

    ok = (not elig_t6[2]) and (not elig_t8[2]) and elig_t10[2] and all(elig_t6[:2]) and all(elig_t10[:2])
    print_check("eligible_mask: entrant CCC excluded until real_frac crosses threshold at t=10",
                ok, f"t6={elig_t6}, t8={elig_t8}, t10={elig_t10}")
    return passed + ok, failed + (not ok)


def test_harness_smoke(passed, failed):
    panel = _entrant_panel()
    cfg = RiskConfig(lookback=5, min_history_frac=0.8, rebalance_every=2)
    ccc_gidx = 3

    for policy in ("min_variance", "risk_parity", "min_variance_voltarget", "risk_parity_voltarget"):
        weight_fn = make_risk_weight_fn(policy, cfg, panel, panel.start_idx)
        result = run_backtest(panel, weight_fn, C_SELL, C_BUY, panel.start_idx, panel.end_idx)

        ok_simplex = np.allclose(result.weights.sum(axis=1), 1.0, atol=1e-8)
        print_check(f"{policy}: every day's weights sum to 1", ok_simplex)
        passed, failed = passed + ok_simplex, failed + (not ok_simplex)

        ok_finite = bool(np.all(np.isfinite(result.portfolio_value)))
        print_check(f"{policy}: portfolio value finite throughout", ok_finite)
        passed, failed = passed + ok_finite, failed + (not ok_finite)

        # non-rebalance days (offset 1 from start_idx, since rebalance_every=2) must have zero turnover
        non_rebalance_turnover = result.turnover[1::2]
        ok_norebal = np.allclose(non_rebalance_turnover, 0.0, atol=1e-10)
        print_check(f"{policy}: zero turnover on non-rebalance days", ok_norebal,
                    str(non_rebalance_turnover))
        passed, failed = passed + ok_norebal, failed + (not ok_norebal)

        # CCC (entrant) must hold exactly 0 weight until it seasons at t=10 (index 4 in this window)
        ccc_weights_before_seasoning = result.weights[:4, ccc_gidx]  # t=6,7,8,9
        ok_entrant = np.allclose(ccc_weights_before_seasoning, 0.0, atol=1e-10)
        print_check(f"{policy}: entrant CCC held at 0 until seasoned", ok_entrant,
                    str(ccc_weights_before_seasoning))
        passed, failed = passed + ok_entrant, failed + (not ok_entrant)

    return passed, failed


def test_determinism(passed, failed):
    panel = _entrant_panel()
    cfg = RiskConfig(lookback=5, min_history_frac=0.8, rebalance_every=2)

    for policy in ("min_variance", "risk_parity"):
        r1 = run_backtest(panel, make_risk_weight_fn(policy, cfg, panel, panel.start_idx),
                           C_SELL, C_BUY, panel.start_idx, panel.end_idx)
        r2 = run_backtest(panel, make_risk_weight_fn(policy, cfg, panel, panel.start_idx),
                           C_SELL, C_BUY, panel.start_idx, panel.end_idx)
        ok = np.allclose(r1.portfolio_value, r2.portfolio_value, atol=1e-12) and \
            np.allclose(r1.weights, r2.weights, atol=1e-12)
        print_check(f"{policy}: bit-identical across two runs (no RNG in these policies)", ok)
        passed, failed = passed + ok, failed + (not ok)

    return passed, failed


def test_unknown_policy_raises(passed, failed):
    try:
        make_risk_weight_fn("not_a_real_policy", RiskConfig(), _entrant_panel(), 6)
        ok = False
    except ValueError:
        ok = True
    print_check("make_risk_weight_fn: unknown policy name raises ValueError", ok)
    return passed + ok, failed + (not ok)


def test_trailing_returns_lookahead_guard(passed, failed):
    panel = _entrant_panel()
    try:
        trailing_returns(panel, t=3, lookback=5, gidx=np.array([1, 2]))
        ok = False
    except ValueError:
        ok = True
    print_check("trailing_returns: t < lookback raises ValueError (no silent lookahead)", ok)
    return passed + ok, failed + (not ok)


def main():
    print_header("test_risk_portfolios")
    passed = failed = 0

    passed, failed = test_min_variance_2asset(passed, failed)
    passed, failed = test_erc_2asset(passed, failed)
    passed, failed = test_min_variance_diag(passed, failed)
    passed, failed = test_erc_diag(passed, failed)
    passed, failed = test_erc_property_random(passed, failed)
    passed, failed = test_vol_target_overlay(passed, failed)
    passed, failed = test_max_weight_cap(passed, failed)
    passed, failed = test_degenerate_cov(passed, failed)
    passed, failed = test_estimator_recovery(passed, failed)
    passed, failed = test_eligible_mask_entrant_seasoning(passed, failed)
    passed, failed = test_harness_smoke(passed, failed)
    passed, failed = test_determinism(passed, failed)
    passed, failed = test_unknown_policy_raises(passed, failed)
    passed, failed = test_trailing_returns_lookahead_guard(passed, failed)

    print_section_end(passed, failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
