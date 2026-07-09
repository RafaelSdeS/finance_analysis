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
        # Capture mask BEFORE step (when decision is made)
        mask_at_decision = env.mask[env._t].copy()
        obs, reward, terminated, truncated, info = env.step(action)

        w = info["weights"]
        # Weights should be valid against the mask at decision time, not current mask
        assert np.isfinite(reward), f"non-finite reward at step {n_checked}"
        assert np.isfinite(info["log_return"]), f"non-finite log_return at step {n_checked}"
        assert abs(w.sum() - 1.0) < 1e-9, f"weights sum {w.sum()} != 1"
        assert (w >= 0).all(), "negative weight found"
        assert (w[~mask_at_decision] == 0).all(), "INACTIVE TICKER GOT NONZERO WEIGHT"
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

    # --- 7. First deployment decision (from the 100%-CASH reset state) is a real
    #        timing call, scored against "what if I'd stayed in cash" -- NOT
    #        reward-neutral. This is intentional (see test 7b's contrast: a
    #        REPEATED decision that changes nothing IS reward-neutral). ---
    env_deploy = PortfolioEnv(cfg, date_range="val")
    env_deploy.reset(seed=99)
    ew_action = np.zeros(len(env_deploy.tickers), dtype=np.float32)  # uniform incl. CASH
    _, rew_deploy, _, _, info_deploy = env_deploy.step(ew_action)
    assert np.isfinite(rew_deploy) and np.isfinite(info_deploy["log_return"])
    print(f"✓ first deployment decision (cash→~equity) → reward={rew_deploy:.4f} "
          f"(a real timing call vs the cash starting point, not forced to ~0)")

    # --- 7b. Cash-aware reward, previous-weight benchmark (M3.1, corrected):
    #         a decision that changes NOTHING (repeats the prior allocation) must
    #         net ~0 reward regardless of cash level -- no free lunch for static
    #         positioning. Deterministic via mocked returns (private copies, not
    #         the shared lru_cache) so the assertion doesn't depend on what the
    #         real historical window happened to do. ---
    env_static = PortfolioEnv(cfg, date_range="val")
    env_static._simple_rets = env_static._simple_rets.copy()
    env_static._ew_log_returns = env_static._ew_log_returns.copy()
    env_static.reset(seed=1)
    cash_idx = np.where(env_static._is_cash_mask)[0][0]
    cash_action = np.full(len(env_static.tickers), -100.0, dtype=np.float32)
    cash_action[cash_idx] = 100.0  # near-argmax on CASH

    N = cfg.rebalance_interval_days

    def mock_scenario(env: PortfolioEnv, market_ret: float, cash_ret: float = 0.0005) -> None:
        """Force every ticker's return over the NEXT decision window to `market_ret`
        (so any equity allocation realizes exactly that return, matching the
        benchmark's EW term by construction) except CASH, forced to `cash_ret`."""
        idx = np.arange(env._t + 1, min(env._t + N + 1, len(env._ew_log_returns)))
        env._simple_rets[idx, :] = market_ret
        env._simple_rets[idx, cash_idx] = cash_ret
        env._ew_log_returns[idx] = np.log1p(market_ret)

    env_static.step(cash_action)  # decision 1: cash (reset)->cash, i.e. no real change
    mock_scenario(env_static, market_ret=-0.05)  # force a severe mocked crash
    _, rew_static, _, _, _ = env_static.step(cash_action)  # decision 2: repeat 100% CASH
    assert abs(rew_static) < 0.02 * cfg.reward_scale, (
        f"repeating an unchanged 100% CASH decision should net ~0 reward even during a "
        f"mocked crash (no NEW timing information), got {rew_static:.4f}"
    )
    print(f"✓ static repeated CASH decision during a mocked crash → reward={rew_static:.5f} "
          f"(no free lunch for an already-held position, even a lucky one)")

    # --- 7c. A NEW, well-timed move into cash right before a mocked crash earns
    #         clearly positive reward, and a NEW, well-timed move BACK into equity
    #         right before a mocked rally (after that crash) also earns clearly
    #         positive reward -- the previous-weight benchmark can still teach
    #         genuine defensive timing in both directions (the whole point of the
    #         fix). Each transition here is a real decision (prev != new), unlike
    #         test 7b's repeated, unchanged decision. ---
    env_timing = PortfolioEnv(cfg, date_range="val")
    env_timing._simple_rets = env_timing._simple_rets.copy()
    env_timing._ew_log_returns = env_timing._ew_log_returns.copy()
    env_timing.reset(seed=1)
    equity_action = np.zeros(len(env_timing.tickers), dtype=np.float32)
    equity_action[cash_idx] = -100.0  # exclude cash, uniform over active stocks
    env_timing.step(equity_action)  # establish a mostly-equity starting position

    mock_scenario(env_timing, market_ret=-0.05)  # mocked crash ahead
    _, rew_crash, _, _, _ = env_timing.step(cash_action)  # NEW decision: equity → cash
    assert rew_crash > 0, f"moving into cash right before a mocked crash should be rewarded, got {rew_crash:.4f}"
    print(f"✓ well-timed equity→cash move before a mocked crash → reward={rew_crash:+.4f} (rewarded)")

    mock_scenario(env_timing, market_ret=0.05)  # mocked rally ahead
    _, rew_reentry, _, _, _ = env_timing.step(equity_action)  # NEW decision: cash → equity
    assert rew_reentry > 0, f"re-entering equity right before a mocked rally should be rewarded, got {rew_reentry:.4f}"
    print(f"✓ well-timed cash→equity re-entry before a mocked rally → reward={rew_reentry:+.4f} (rewarded)")

    # --- 7d. The mirror image: a NEW move into cash right before a mocked rally
    #         (a bad call) earns clearly negative reward. ---
    env_bad = PortfolioEnv(cfg, date_range="val")
    env_bad._simple_rets = env_bad._simple_rets.copy()
    env_bad._ew_log_returns = env_bad._ew_log_returns.copy()
    env_bad.reset(seed=1)
    env_bad.step(equity_action)  # establish a mostly-equity starting position

    mock_scenario(env_bad, market_ret=0.05)  # mocked rally ahead
    _, rew_bad, _, _, _ = env_bad.step(cash_action)  # NEW decision: equity → cash (bad call)
    assert rew_bad < 0, f"moving into cash right before a mocked rally should be penalized, got {rew_bad:.4f}"
    print(f"✓ badly-timed equity→cash move before a mocked rally → reward={rew_bad:+.4f} (penalized)")

    # --- 8. Identical-bounds envs share cached tensors (online-backtest memory fix) ---
    env_a = PortfolioEnv(cfg, date_range="train")
    env_b = PortfolioEnv(cfg, date_range="train")
    assert env_a.features is env_b.features, "same-bounds envs should share the cached array, not copy it"
    env_a.reset(seed=1)
    env_b.reset(seed=2)
    env_a._prev_weights[0] = 999.0  # mutate one instance's private state
    assert env_b._prev_weights[0] != 999.0, "per-env mutable state leaked across shared-cache instances"
    print("✓ identical-bounds envs share cached tensors without leaking mutable state")

    # --- 9. Max position weight cap + redistribution (not forced to CASH) ---
    env_cap = PortfolioEnv(cfg, date_range="val")
    env_cap.reset(seed=42)
    cap = cfg.max_position_weight

    # Create an extreme one-hot action (should be capped and overflow redistributed to other stocks)
    extreme_action = np.ones(len(env_cap.tickers), dtype=np.float32) * 100.0
    extreme_action[0] = 1000.0  # make first stock massively attractive
    obs, _, _, _, info = env_cap.step(extreme_action)

    w = info["weights"]
    stock_mask = ~env_cap._is_cash_mask
    stock_weights = w[stock_mask]  # exclude CASH
    max_weight = stock_weights.max()
    nonzero_stocks = (stock_weights > 1e-12).sum()
    cash_weight = w[-1]

    assert max_weight <= cap + 1e-9, f"max_weight {max_weight} exceeds cap {cap}"
    assert abs(w.sum() - 1.0) < 1e-9, f"weights don't sum to 1: {w.sum()}"
    # Key fix: extreme one-hot should diversify to many stocks, NOT force 90% into CASH
    # If this fails, the old (buggy) behavior forced overflow into CASH
    assert nonzero_stocks > 10, f"overflow should redistribute to {nonzero_stocks} stocks, not CASH"
    assert cash_weight < 0.5, f"CASH weight {cash_weight:.2%} should not absorb the capped overflow"
    print(f"✓ max position cap + redistribution: one-hot → {nonzero_stocks} stocks, CASH={cash_weight:.1%} (was 90% before fix)")

    print("\n" + "=" * 60)
    print("ALL ENV TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
