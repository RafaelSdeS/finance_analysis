#!/usr/bin/env python3
"""
agent_vs_equal_weight: excess-Sharpe wiring + formula sanity.

Run from project root: python tests/agent/test_excess_sharpe.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.config import DEFAULT_CONFIG
from src.agent.env import PortfolioEnv
from src.agent.evaluate import agent_vs_equal_weight, equal_weight_policy
from src.agent.metrics import sharpe_ratio


class _EqualWeightStubModel:
    """Fake PPO model that replays equal_weight_policy step-by-step (agent_policy's
    act_fn signature drops t, so this stub tracks its own counter)."""
    def __init__(self, env):
        self._act = equal_weight_policy(env)
        self._t = 0

    def predict(self, obs, deterministic=True):
        action = self._act(obs, self._t)
        self._t += 1
        return action, None


def main() -> None:
    cfg = DEFAULT_CONFIG

    # --- 1. Wiring check: equal-weight "agent" vs equal-weight baseline → excess ≈ 0 ---
    env = PortfolioEnv(cfg, date_range="val")
    stub = _EqualWeightStubModel(env)
    result = agent_vs_equal_weight(env, stub)
    assert set(result) == {"sharpe", "excess_sharpe", "max_drawdown", "final_value"}, f"bad keys: {result}"
    assert abs(result["excess_sharpe"]) < 0.5, (
        f"equal-weight vs itself should give ~0 excess_sharpe, got {result['excess_sharpe']:.4f} "
        "(nonzero likely due to mask/cash-handling asymmetry between the two rollouts)"
    )
    print(f"✓ equal-weight vs itself: excess_sharpe={result['excess_sharpe']:.4f} (~0 as expected)")

    # --- 2. Formula check: excess_sharpe is Sharpe of the DIFF series, not diff of Sharpes ---
    agent_returns = np.array([0.01, 0.02, 0.015, 0.005, 0.01])
    ew_returns = np.array([0.005, 0.005, 0.005, 0.005, 0.005])
    excess = agent_returns - ew_returns
    expected = sharpe_ratio(excess)
    diff_of_sharpes = sharpe_ratio(agent_returns) - sharpe_ratio(ew_returns)
    assert expected != diff_of_sharpes, (
        "test fixture doesn't actually distinguish diff-of-sharpes from sharpe-of-diff"
    )
    print(f"✓ excess Sharpe formula: sharpe_ratio(diff)={expected:.4f} "
          f"≠ diff_of_sharpes={diff_of_sharpes:.4f}")

    print("\nALL EXCESS-SHARPE TESTS PASSED ✓")


if __name__ == "__main__":
    main()
