#!/usr/bin/env python3
"""
Metric functions vs hand-computed values.

Run from project root: python tests/agent/test_backtest_metrics.py (pytest-compatible)
or: pytest tests/agent/test_backtest_metrics.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

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


def test_sharpe_zero_mean() -> None:
    r = np.array([0.01, -0.01, 0.01, -0.01])
    assert abs(sharpe_ratio(r)) < 1e-9, "zero-mean returns should give Sharpe ~0"


def test_sharpe_formula() -> None:
    r = np.array([0.01, 0.02, 0.015, 0.005])
    expected = r.mean() / r.std() * np.sqrt(252)
    assert approx(sharpe_ratio(r), expected), "Sharpe formula mismatch"


def test_sortino_exceeds_sharpe_no_downside() -> None:
    r_up = np.array([0.01, 0.02, 0.01])
    assert sortino_ratio(r_up) > sharpe_ratio(r_up), "Sortino should exceed Sharpe with no downside"


def test_max_drawdown() -> None:
    v = np.array([100.0, 120.0, 60.0, 90.0])
    assert approx(max_drawdown(v), 0.5), f"expected 0.5, got {max_drawdown(v)}"
    assert approx(max_drawdown(np.array([1.0, 2.0, 3.0])), 0.0)


def test_cumulative_return() -> None:
    assert approx(cumulative_return(np.array([100.0, 130.0, 150.0])), 0.5)


def test_win_rate() -> None:
    assert approx(win_rate(np.array([0.01, 0.02, -0.01, 0.03])), 0.75)


def test_compute_all_keys_and_finite() -> None:
    result = compute_all(np.array([0.01, -0.005, 0.02]), np.array([100.0, 101.0, 100.5, 102.5]))
    expected_keys = {"cumulative_return", "annualized_return", "sharpe", "sortino",
                     "max_drawdown", "win_rate", "n_days"}
    assert set(result) == expected_keys, f"keys mismatch: {set(result)}"
    assert all(np.isfinite(v) for v in result.values()), "non-finite metric"


def test_edge_cases_empty_and_single() -> None:
    assert sharpe_ratio(np.array([])) == 0.0
    assert max_drawdown(np.array([100.0])) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
