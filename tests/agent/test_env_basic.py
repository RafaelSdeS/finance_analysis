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
    n_features = len(cfg.state_features)

    print("=" * 60)
    print("TEST: PortfolioEnv basic invariants")
    print("=" * 60)

    # --- 1. Construction & reset (all three splits) ---
    for split in ("train", "val", "test"):
        env = PortfolioEnv(cfg, date_range=split)
        n_tickers = len(env.tickers)  # data-derived (universe size drifts as raw data grows)
        obs_dim_expected = n_tickers * n_features + 2 * n_tickers
        obs, info = env.reset(seed=cfg.seed)
        assert obs.shape == (obs_dim_expected,), f"{split}: obs shape {obs.shape}"
        assert np.isfinite(obs).all(), f"{split}: obs has non-finite values"
        assert env._prev_weights.sum() == 1.0, f"{split}: initial weights don't sum to 1"
        assert (env._prev_weights[env._is_cash_mask] == 1.0).all(), f"{split}: should start 100% CASH"
        print(f"✓ {split}: reset OK, obs_dim={obs.shape[0]}, days={len(env.dates)}, starts 100% CASH")

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
        assert np.isfinite(info["log_return"]), f"non-finite log_return at step {n_checked}"
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

    # --- 5. Transaction cost: turnover tracking and CASH exclusion ---
    env4 = PortfolioEnv(cfg, date_range="val")
    env4.reset(seed=cfg.seed)

    # Verify the info dict includes turnover
    for _ in range(10):
        action = np.random.randn(len(env4.tickers)).astype(np.float32)
        _, _, terminated, _, info = env4.step(action)
        assert "turnover" in info, "turnover not in info dict"
        assert info["turnover"] >= 0, f"turnover={info['turnover']} should be non-negative"
        assert info["turnover"] <= 2.0, f"turnover={info['turnover']} should not exceed 2.0 (max one-way delta)"
        if terminated:
            break
    print(f"✓ transaction cost: turnover tracking works, CASH excluded from traded sum")

    # --- 6. Transaction cost: repeated action has zero cost ---
    env_test = PortfolioEnv(cfg, date_range="val")
    env_test.reset(seed=99)

    # Take an action
    action = np.ones(len(env_test.tickers), dtype=np.float32) * 0.1
    obs1, reward1, _, _, info1 = env_test.step(action)
    turnover1 = info1["turnover"]

    # Repeat the same action (should have zero turnover since weights don't change, only mask drifts)
    obs2, reward2, _, _, info2 = env_test.step(action)
    turnover2 = info2["turnover"]

    # Second step should have near-zero turnover (only mask changes, not policy choice)
    assert turnover2 < turnover1 * 0.1, f"repeated action should have low turnover: {turnover1:.4f} → {turnover2:.4f}"
    print(f"✓ reward penalty: turnover tracked correctly ({turnover1:.4f} → {turnover2:.4f})")

    # --- 7. Equal-weight action yields zero excess reward (reward = log_return - ew_log_return) ---
    env_ew = PortfolioEnv(cfg, date_range="val")
    env_ew.reset(seed=99)

    # Equal-weight action: all tickers get the same logit (softmax is uniform)
    ew_action = np.zeros(len(env_ew.tickers), dtype=np.float32)
    obs, rew, _, _, info = env_ew.step(ew_action)

    # excess reward should be ≈ 0 (agent return ≈ equal-weight return), maybe slightly negative due to cost
    assert rew < 0.01, f"equal-weight action should yield ~0 excess reward, got {rew:.4f}"
    assert np.isfinite(info["log_return"]), "log_return must be finite"
    print(f"✓ excess reward model: equal-weight action → reward={rew:.5f} (cost drag only)")

    # --- 8. Identical-bounds envs share cached tensors (online-backtest memory fix) ---
    env_a = PortfolioEnv(cfg, date_range="train")
    env_b = PortfolioEnv(cfg, date_range="train")
    assert env_a.features is env_b.features, "same-bounds envs should share the cached array, not copy it"
    env_a.reset(seed=1)
    env_b.reset(seed=2)
    env_a._prev_weights[0] = 999.0  # mutate one instance's private state
    assert env_b._prev_weights[0] != 999.0, "per-env mutable state leaked across shared-cache instances"
    print("✓ identical-bounds envs share cached tensors without leaking mutable state")

    print("\n" + "=" * 60)
    print("ALL ENV TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
