#!/usr/bin/env python3
"""
PortfolioEnv validation: shapes, masking invariants, reward sanity, speed.

Run from project root: python tests/agent/test_env_basic.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.config import DEFAULT_CONFIG
from src.agent.env import PortfolioEnv


def main() -> None:
    cfg = DEFAULT_CONFIG
    n_tickers_expected = 279
    n_features = len(cfg.state_features)
    obs_dim_expected = n_tickers_expected * n_features + n_tickers_expected

    print("=" * 60)
    print("TEST: PortfolioEnv basic invariants")
    print("=" * 60)

    # --- 1. Construction & reset (all three splits) ---
    for split in ("train", "val", "test"):
        env = PortfolioEnv(cfg, date_range=split)
        obs, info = env.reset(seed=cfg.seed)
        assert obs.shape == (obs_dim_expected,), f"{split}: obs shape {obs.shape}"
        assert np.isfinite(obs).all(), f"{split}: obs has non-finite values"
        print(f"✓ {split}: reset OK, obs_dim={obs.shape[0]}, days={len(env.dates)}")

    # --- 2. Random policy episode on train (masking + reward invariants) ---
    env = PortfolioEnv(cfg, date_range="train")
    obs, _ = env.reset(seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    n_checked = 0
    t0 = time.time()
    for _ in range(1000):
        action = rng.normal(size=env.action_space.shape).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        w = info["weights"]
        active = env.mask[env._t - 1]  # mask at decision time
        assert np.isfinite(reward), f"non-finite reward at step {n_checked}"
        assert abs(w.sum() - 1.0) < 1e-9, f"weights sum {w.sum()} != 1"
        assert (w >= 0).all(), "negative weight found"
        assert (w[~active] == 0).all(), "INACTIVE TICKER GOT NONZERO WEIGHT"
        assert info["portfolio_value"] > 0, "portfolio value went non-positive"
        assert np.isfinite(obs).all(), "obs has non-finite values"

        n_checked += 1
        if terminated:
            break
    elapsed = time.time() - t0
    sps = n_checked / elapsed

    print(f"✓ random policy: {n_checked} steps, all invariants held")
    print(f"✓ final portfolio value: R$ {info['portfolio_value']:,.2f} "
          f"(start R$ {cfg.initial_capital:,.2f})")
    print(f"✓ speed: {sps:,.0f} steps/sec "
          f"({'OK for 1M timesteps' if sps > 300 else 'WARNING: SLOW'})")

    # --- 3. Universe grows over time (masking is real) ---
    env2 = PortfolioEnv(cfg, date_range="train")
    first_active = int(env2.mask[0].sum())
    last_active = int(env2.mask[-1].sum())
    print(f"✓ active tickers: {first_active} (2000) → {last_active} ({env2.dates[-1].year})")
    assert first_active < last_active, "expected universe to grow over train period"

    # --- 4. Determinism: same seed + same actions → same trajectory ---
    vals = []
    for _ in range(2):
        env3 = PortfolioEnv(cfg, date_range="val")
        env3.reset(seed=123)
        r = np.random.default_rng(123)
        for _ in range(50):
            _, _, _, _, inf = env3.step(r.normal(size=env3.action_space.shape).astype(np.float32))
        vals.append(inf["portfolio_value"])
    assert vals[0] == vals[1], f"non-deterministic: {vals}"
    print(f"✓ deterministic under fixed seed")

    print("\n" + "=" * 60)
    print("ALL ENV TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
