"""
DIAGNOSIS_PLAN.md Phase 3.2: does the risk_aversion (lambda) mean-variance penalty dominate
the reward signal? TODO.md 3.2: "if penalty > 50% of |reward| on average, sweep
lambda in {0, 1, 5}."

Rolls a policy through PortfolioEnv and sums info["reward_raw_excess"] / info["reward_penalty"]
(exact per-step decomposition, see env.py step()) across the whole split.

Usage:
    python tools/reward_decomposition_audit.py [--model artifacts/models/agent_best.zip] [--split train]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root, for `src.*` imports

from src.agent.config import DEFAULT_CONFIG
from src.agent.env import PortfolioEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward decomposition audit (DIAGNOSIS_PLAN Phase 3.2)")
    parser.add_argument("--model", type=Path, default=DEFAULT_CONFIG.model_dir / "agent_best.zip")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train",
                         help="train is what PPO actually optimizes against; default")
    args = parser.parse_args()

    env = PortfolioEnv(DEFAULT_CONFIG, date_range=args.split)
    model = PPO.load(args.model, device="cpu")

    obs, _ = env.reset()
    raw_excess_total, penalty_total, reward_total = 0.0, 0.0, 0.0
    n_steps = 0
    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, info = env.step(action)
        raw_excess_total += info["reward_raw_excess"]
        penalty_total += info["reward_penalty"]
        reward_total += reward
        n_steps += 1

    print(f"\nReward decomposition over {n_steps} decisions ({args.split} split, {args.model.name}):")
    print(f"  sum(reward_raw_excess): {raw_excess_total:+.4f}")
    print(f"  sum(reward_penalty):    {penalty_total:+.4f}")
    print(f"  sum(reward) [= raw - penalty]: {reward_total:+.4f}")
    print(f"  identity check: {abs((raw_excess_total - penalty_total) - reward_total):.2e} (should be ~0)")

    mean_abs_raw = abs(raw_excess_total) / n_steps
    mean_abs_penalty = abs(penalty_total) / n_steps
    mean_abs_reward = abs(reward_total) / n_steps if reward_total != 0 else np.nan
    ratio_vs_raw = mean_abs_penalty / max(mean_abs_raw, 1e-12)
    print(f"\n  mean |raw_excess| per decision: {mean_abs_raw:.4f}")
    print(f"  mean |penalty| per decision:    {mean_abs_penalty:.4f}")
    print(f"  penalty / |raw_excess|:          {ratio_vs_raw:.1%}")
    if reward_total != 0:
        print(f"  penalty / |total reward| (TODO 3.2's stated criterion): {mean_abs_penalty / mean_abs_reward:.1%}")

    if ratio_vs_raw > 0.5:
        print("\n  -> penalty is a MAJOR component (>50% of raw excess) -- sweep risk_aversion per TODO 3.2")
    else:
        print("\n  -> penalty is a MINOR component (<50% of raw excess) -- no evidence lambda needs sweeping")


if __name__ == "__main__":
    main()
