"""
Test: baselines.py's 7 baseline strategies, all through the shared
environment (docs/eiie_agent/EIIE_AGENT_PLAN.md Phase 5). Synthetic data only.

Run from project root:
    python tests/rl_agent/test_baselines.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.rl_agent.baselines import BASELINE_NAMES, run_baseline  # noqa: E402
from src.rl_agent.data import GlobalAssetIndex, PricePanel  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402

C_SELL = C_BUY = 0.0003


def _panel():
    """cash + A (fastest grower, the hindsight best-stock), B (moderate),
    C (flat). 8 days; experiment window is days [2, 7] (2 days lookback)."""
    asset_index = GlobalAssetIndex(tickers=("AAA", "BBB", "CCC"),
                                    ticker_to_gidx={"AAA": 1, "BBB": 2, "CCC": 3})
    dates = pd.bdate_range("2020-01-01", periods=8)
    close_a = np.array([10, 10, 11, 12, 14, 17, 21, 26], dtype=float)
    close_b = np.array([10, 10, 10.5, 11, 11.5, 12, 12.5, 13], dtype=float)
    close_c = np.full(8, 10.0)
    close = np.column_stack([np.ones(8), close_a, close_b, close_c])
    bova11 = np.array([100, 100, 101, 102, 104, 103, 105, 108], dtype=float)
    return PricePanel(
        asset_index=asset_index, dates=dates, close=close, high=close.copy(), low=close.copy(),
        cdi_factor=np.full(8, 1.0002),
        slot_gidx=np.array([[1, 2, 3]] * 8), valid=np.array([[True, True, True]] * 8),
        window=2, start_idx=2, end_idx=7,
        bova11_close=bova11,
    )


def test_registry(passed, failed):
    ok = set(BASELINE_NAMES) == {"ubah", "ucrp", "best_stock", "random_portfolio",
                                  "random_rebalancing", "constant_cash", "bova11"}
    print_check("BASELINE_NAMES contains all 7 required baselines", ok, str(BASELINE_NAMES))
    passed, failed = passed + ok, failed + (not ok)

    try:
        run_baseline("not_a_real_baseline", _panel(), C_SELL, C_BUY)
        ok = False
    except ValueError:
        ok = True
    print_check("run_baseline: unknown name raises ValueError", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_constant_cash(passed, failed):
    panel = _panel()
    result = run_baseline("constant_cash", panel, C_SELL, C_BUY)

    ok = np.allclose(result.weights, np.array([1.0, 0.0, 0.0, 0.0]))
    print_check("constant_cash: 100% cash every day", ok)
    passed, failed = passed + ok, failed + (not ok)

    expected_pv = np.concatenate([[1.0], np.cumprod(panel.cdi_factor[panel.start_idx:panel.end_idx + 1])])
    ok = np.allclose(result.portfolio_value, expected_pv)
    print_check("constant_cash: portfolio value tracks pure CDI compounding exactly", ok,
                f"got {result.portfolio_value}, expected {expected_pv}")
    passed, failed = passed + ok, failed + (not ok)

    ok = np.allclose(result.mu, 1.0)
    print_check("constant_cash: never trades, mu = 1 always", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_ucrp(passed, failed):
    panel = _panel()
    result = run_baseline("ucrp", panel, C_SELL, C_BUY)
    ok = np.allclose(result.weights, 0.25)
    print_check("ucrp: equal weight (1/4) across cash + 3 active assets, every day", ok,
                str(result.weights[0]))
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_ubah(passed, failed):
    panel = _panel()
    result = run_baseline("ubah", panel, C_SELL, C_BUY)

    ok = np.allclose(result.weights[0], 0.25)
    print_check("ubah: initial allocation matches UCRP's uniform weights", ok)
    passed, failed = passed + ok, failed + (not ok)

    ok = np.allclose(result.mu[1:], 1.0)
    print_check("ubah: no further trading after the initial buy -- mu = 1 for all later days", ok,
                str(result.mu))
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_best_stock(passed, failed):
    panel = _panel()
    result = run_baseline("best_stock", panel, C_SELL, C_BUY)
    # asset A (global idx 1) grows fastest over [start_idx, end_idx] -- must be the one bought
    ok = np.isclose(result.weights[0, 1], 1.0) and np.isclose(result.weights[0].sum(), 1.0)
    print_check("best_stock: picks asset A (the hindsight-best performer) at 100%", ok,
                str(result.weights[0]))
    passed, failed = passed + ok, failed + (not ok)

    ok = np.allclose(result.mu[1:], 1.0)
    print_check("best_stock: buy-once, hold thereafter (mu = 1 after the initial trade)", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_random_determinism_and_turnover(passed, failed):
    panel = _panel()
    r1 = run_baseline("random_portfolio", panel, C_SELL, C_BUY, seed=123)
    r2 = run_baseline("random_portfolio", panel, C_SELL, C_BUY, seed=123)
    ok = np.allclose(r1.portfolio_value, r2.portfolio_value)
    print_check("random_portfolio: deterministic given the same seed", ok)
    passed, failed = passed + ok, failed + (not ok)

    rp = run_baseline("random_portfolio", panel, C_SELL, C_BUY, seed=7)
    rr = run_baseline("random_rebalancing", panel, C_SELL, C_BUY, seed=7)
    ok = rr.turnover.mean() > rp.turnover.mean()
    print_check("random_rebalancing has higher average turnover than static random_portfolio",
                ok, f"rebalancing={rr.turnover.mean():.4f}, static={rp.turnover.mean():.4f}")
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_bova11(passed, failed):
    panel = _panel()
    result = run_baseline("bova11", panel, C_SELL, C_BUY)

    prices = panel.bova11_close[panel.start_idx - 1: panel.end_idx + 1]
    expected_pv = np.concatenate([[1.0], np.cumprod(prices[1:] / prices[:-1])])
    ok = np.allclose(result.portfolio_value, expected_pv)
    print_check("bova11: portfolio value matches its own adj_close ratio series exactly", ok,
                f"got {result.portfolio_value}, expected {expected_pv}")
    passed, failed = passed + ok, failed + (not ok)

    ok = np.allclose(result.mu, 1.0) and np.allclose(result.turnover, 0.0)
    print_check("bova11: no transaction-cost mechanics (single passive ETF hold)", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def main():
    print_header("test_baselines")
    passed = failed = 0

    passed, failed = test_registry(passed, failed)
    passed, failed = test_constant_cash(passed, failed)
    passed, failed = test_ucrp(passed, failed)
    passed, failed = test_ubah(passed, failed)
    passed, failed = test_best_stock(passed, failed)
    passed, failed = test_random_determinism_and_turnover(passed, failed)
    passed, failed = test_bova11(passed, failed)

    print_section_end(passed, failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
