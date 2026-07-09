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
    deflated_sharpe_ratio,
    effective_n_positions,
    expected_max_sharpe,
    max_drawdown,
    probabilistic_sharpe_ratio,
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


def test_effective_n_positions_uniform() -> None:
    w = np.zeros((5, 10))
    w[:, :] = 0.1  # uniform: 10 tickers × 0.1 each
    assert approx(effective_n_positions(w), 10.0, tol=1e-3), "uniform weights over 10 should give effective_n≈10"


def test_effective_n_positions_concentrated() -> None:
    w = np.zeros((5, 10))
    w[:, 0] = 1.0  # all mass on one ticker
    assert approx(effective_n_positions(w), 1.0, tol=1e-9), "single position should give effective_n=1"


def test_effective_n_positions_half_half() -> None:
    w = np.zeros((5, 10))
    w[:, 0] = 0.5
    w[:, 1] = 0.5  # two equal positions
    assert approx(effective_n_positions(w), 2.0, tol=1e-9), "two equal positions should give effective_n=2"


def test_effective_n_positions_edge_cases() -> None:
    assert effective_n_positions(np.array([])) == 0.0
    # Single ticker with all mass → HHI = 1.0, effective_n = 1
    w_single = np.ones((5, 1))  # each row: [1.0] (100% in one ticker)
    assert approx(effective_n_positions(w_single), 1.0, tol=1e-9)


def test_edge_cases_empty_and_single() -> None:
    assert sharpe_ratio(np.array([])) == 0.0
    assert max_drawdown(np.array([100.0])) == 0.0


def test_psr_high_sharpe_near_one() -> None:
    rng = np.random.default_rng(0)
    r = 0.01 + 0.001 * rng.standard_normal(500)  # strong, low-noise positive drift
    assert probabilistic_sharpe_ratio(r) > 0.99, "high, low-noise Sharpe should give PSR near 1"


def test_psr_zero_sharpe_near_half() -> None:
    r = np.array([0.01, -0.01] * 100)  # exactly zero-mean
    assert approx(probabilistic_sharpe_ratio(r, benchmark_sr=0.0), 0.5, tol=1e-6)


def test_expected_max_sharpe_increases_with_trials() -> None:
    assert expected_max_sharpe(50, 0.5) > expected_max_sharpe(5, 0.5)
    assert expected_max_sharpe(1, 0.5) == 0.0, "single trial has no 'max of N' effect"
    assert expected_max_sharpe(10, 0.0) == 0.0, "zero cross-trial variance has no effect"


def test_deflated_sharpe_ratio_penalizes_more_trials() -> None:
    rng = np.random.default_rng(1)
    r = 0.005 + 0.01 * rng.standard_normal(300)
    few_trials = np.array([0.01, 0.02, 0.015])
    many_trials = np.concatenate([few_trials, rng.normal(0.015, 0.01, size=50)])
    assert deflated_sharpe_ratio(r, many_trials) < deflated_sharpe_ratio(r, few_trials), \
        "more trials with the same spread should deflate the Sharpe further"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
