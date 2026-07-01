#!/usr/bin/env python3
"""
Metric functions vs hand-computed values.

Run from project root: python tests/agent/test_backtest_metrics.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.metrics import (
    compute_all,
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


def main() -> None:
    # --- Sharpe: constant positive returns → huge Sharpe; zero mean → ~0 ---
    r = np.array([0.01, -0.01, 0.01, -0.01])
    assert abs(sharpe_ratio(r)) < 1e-9, "zero-mean returns should give Sharpe ~0"

    r = np.array([0.01, 0.02, 0.015, 0.005])
    expected = r.mean() / r.std() * np.sqrt(252)
    assert approx(sharpe_ratio(r), expected), "Sharpe formula mismatch"
    print("✓ sharpe_ratio")

    # --- Sortino: no negative days → very large (downside std ~ 0) ---
    r_up = np.array([0.01, 0.02, 0.01])
    assert sortino_ratio(r_up) > sharpe_ratio(r_up), "Sortino should exceed Sharpe with no downside"
    print("✓ sortino_ratio")

    # --- Max drawdown: 100 → 120 → 60 → 90 gives (120-60)/120 = 50% ---
    v = np.array([100.0, 120.0, 60.0, 90.0])
    assert approx(max_drawdown(v), 0.5), f"expected 0.5, got {max_drawdown(v)}"
    # Monotonic increase → 0 drawdown
    assert approx(max_drawdown(np.array([1.0, 2.0, 3.0])), 0.0)
    print("✓ max_drawdown")

    # --- Cumulative return: 100 → 150 = +50% ---
    assert approx(cumulative_return(np.array([100.0, 130.0, 150.0])), 0.5)
    print("✓ cumulative_return")

    # --- Win rate: 3 of 4 positive ---
    assert approx(win_rate(np.array([0.01, 0.02, -0.01, 0.03])), 0.75)
    print("✓ win_rate")

    # --- compute_all returns every key ---
    result = compute_all(np.array([0.01, -0.005, 0.02]), np.array([100.0, 101.0, 100.5, 102.5]))
    expected_keys = {"cumulative_return", "annualized_return", "sharpe", "sortino",
                     "max_drawdown", "win_rate", "n_days"}
    assert set(result) == expected_keys, f"keys mismatch: {set(result)}"
    assert all(np.isfinite(v) for v in result.values()), "non-finite metric"
    print("✓ compute_all")

    # --- Edge cases: empty / single-element inputs don't crash ---
    assert sharpe_ratio(np.array([])) == 0.0
    assert max_drawdown(np.array([100.0])) == 0.0
    print("✓ edge cases (empty/single)")

    print("\nALL METRICS TESTS PASSED ✓")


if __name__ == "__main__":
    main()
