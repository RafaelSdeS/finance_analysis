#!/usr/bin/env python3
"""
attribution.decompose_returns: exact per-day arithmetic + sanity endpoints.

Run from project root: python tests/agent/test_attribution.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.attribution import decompose_returns


def main() -> None:
    rng = np.random.default_rng(0)
    n = 100
    selic_log = rng.normal(0.0004, 0.0001, n)
    ew_log = rng.normal(0.0003, 0.01, n)

    # --- Test 1: 100% cash always -> agent return == carry exactly, others zero ---
    w_cash = np.ones(n)
    agent_log = selic_log.copy()  # 100% cash means agent earns exactly SELIC
    result = decompose_returns(agent_log, w_cash, selic_log, ew_log)
    assert abs(result["carry_cumulative"] - result["agent_cumulative"]) < 1e-9
    assert abs(result["market_exposure_cumulative"]) < 1e-9
    assert abs(result["selection_residual_cumulative"]) < 1e-9
    print(f"✓ 100% cash: carry={result['carry_cumulative']:.4f} == agent={result['agent_cumulative']:.4f}, "
          f"market_exposure≈0, selection≈0")

    # --- Test 2: 0% cash, agent tracks EW exactly -> market_exposure == agent, others zero ---
    w_cash = np.zeros(n)
    agent_log = ew_log.copy()
    result = decompose_returns(agent_log, w_cash, selic_log, ew_log)
    assert abs(result["market_exposure_cumulative"] - result["agent_cumulative"]) < 1e-9
    assert abs(result["carry_cumulative"]) < 1e-9
    assert abs(result["selection_residual_cumulative"]) < 1e-9
    print(f"✓ 0% cash tracking EW: market_exposure={result['market_exposure_cumulative']:.4f} "
          f"== agent={result['agent_cumulative']:.4f}, carry≈0, selection≈0")

    # --- Test 3: 0% cash, agent DEVIATES from EW -> selection_residual captures it ---
    w_cash = np.zeros(n)
    alpha_daily = 0.001  # constant daily outperformance vs EW
    agent_simple = np.expm1(ew_log) + alpha_daily
    agent_log = np.log1p(agent_simple)
    result = decompose_returns(agent_log, w_cash, selic_log, ew_log)
    assert result["selection_residual_cumulative"] > 0, "positive alpha should show up as positive selection"
    print(f"✓ 0% cash with +{alpha_daily}/day alpha: selection_residual="
          f"{result['selection_residual_cumulative']:.4f} (positive, as expected)")

    # --- Test 4: per-day arithmetic is EXACT (the core invariant) ---
    w_cash = rng.uniform(0, 1, n)
    agent_log = rng.normal(0, 0.01, n)
    result = decompose_returns(agent_log, w_cash, selic_log, ew_log)
    agent_simple = np.expm1(agent_log)
    reconstructed = (result["carry_daily"] + result["market_exposure_daily"]
                     + result["selection_residual_daily"])
    assert np.allclose(reconstructed, agent_simple, atol=1e-10), "per-day decomposition must be exact"
    print("✓ per-day decomposition (carry + market_exposure + selection_residual) == agent return, exactly")

    print("\nALL ATTRIBUTION TESTS PASSED ✓")


if __name__ == "__main__":
    main()
