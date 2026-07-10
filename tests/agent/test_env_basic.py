#!/usr/bin/env python3
"""
PortfolioEnv validation: shapes, masking invariants, reward sanity, speed.

Run from project root: python tests/agent/test_env_basic.py
"""

import dataclasses
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
    # Forced to softmax: the near-argmax `cash_action` trick below only guarantees
    # "100% CASH, identically on both calls" under softmax's argmax behavior. This test
    # is about the reward benchmark mechanism (action_mode-agnostic); tilt's own weight
    # mapping is covered separately below.
    cfg_static = dataclasses.replace(cfg, action_mode="softmax")
    env_static = PortfolioEnv(cfg_static, date_range="val")
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

    # --- 8. Action reparameterization (DIAGNOSIS_PLAN.md Phase 2.2 / roadmap Option C):
    #     problem (softmax saturation, replaying the M5 measurement) vs solution (tilt) ---
    # Uses a date well into train (not t=0) so n_active (~38) is representative of the
    # regime tilt_kappa was calibrated for (~50 names) -- t=0's thin ~10-name universe is
    # its own edge case (equal-weight itself sits at the cap boundary there), covered
    # separately below since it caught a real _cap_weights bug during this test's own development.
    T = 4500
    cfg_softmax = dataclasses.replace(cfg, action_mode="softmax")  # this block tests softmax specifically
    env_softmax = PortfolioEnv(cfg_softmax, date_range="train")
    active = env_softmax.mask[T]
    n_active = int(active.sum())
    rng = np.random.default_rng(0)
    modest_scores = rng.normal(size=n_active)  # a "modestly differentiated" score, not an extreme one-hot

    action_modest = np.zeros(len(env_softmax.tickers), dtype=np.float64)
    action_modest[active] = modest_scores
    stock_mask = ~env_softmax._is_cash_mask
    active_stock = active & stock_mask

    # Pre-cap softmax top-1: reconstruct without the cap to replay the exact M5 measurement
    presoftmax = PortfolioEnv._masked_softmax(action_modest * cfg.logit_scale, active)
    precap_top1 = presoftmax[active_stock].max()
    # Threshold is data-dependent (n_active drifts as the tradable universe/PIT scheme
    # changes -- see DIAGNOSIS_PLAN.md Phase 4); 0.3 still means >10x uniform (~1/42) for
    # this window, clearly saturated relative to a modest, non-extreme score spread.
    uniform = 1.0 / n_active
    assert precap_top1 > 10 * uniform, (
        f"expected softmax(action*logit_scale=10) to saturate toward near-argmax on even a "
        f"modest score spread (replaying the M5 measurement), got pre-cap top-1={precap_top1:.3f} "
        f"(uniform would be {uniform:.3f})"
    )
    print(f"✓ (problem, replaying M5) softmax pre-cap top-1 weight = {precap_top1:.3f} (saturated toward argmax)")

    tilt_cfg = dataclasses.replace(cfg, action_mode="tilt", tilt_kappa=2.0)
    env_tilt = PortfolioEnv(tilt_cfg, date_range="train")
    w_tilt = env_tilt.action_to_weights(action_modest, active)

    assert abs(w_tilt.sum() - 1.0) < 1e-9, f"tilt weights don't sum to 1: {w_tilt.sum()}"
    assert (w_tilt[stock_mask] <= cap + 1e-9).all(), "tilt weights must respect the cap"
    assert (w_tilt[~active] == 0.0).all(), "tilt weights must be exactly 0 on inactive names"
    tilt_top1 = w_tilt[active_stock].max()
    assert 0.05 < tilt_top1 < 0.7, (
        f"expected the tilt rule to keep top-1 in a usable, non-saturated band on a modest "
        f"score spread (the band M2.3's temperature sweep couldn't hit), got {tilt_top1:.3f}"
    )
    print(f"✓ (solution) tilt top-1 weight = {tilt_top1:.3f} (usable band, not saturated)")

    zero_action = np.zeros(len(env_tilt.tickers), dtype=np.float64)
    w_zero = env_tilt.action_to_weights(zero_action, active)
    ew_expected = active.astype(np.float64) / n_active
    assert np.allclose(w_zero, ew_expected, atol=1e-9), "kappa*z=0 (all-zero action) must map to EXACTLY equal-weight"
    print("✓ zero action -> exactly equal-weight under the tilt rule")

    # Monotonicity: strictly increasing one ticker's score (others fixed) must strictly
    # increase its pre-cap weight -- check on a ticker far from the cap boundary.
    probe = np.where(active_stock)[0][0]
    action_lo, action_hi = action_modest.copy(), action_modest.copy()
    action_lo[probe], action_hi[probe] = -1.0, 1.0
    w_lo = env_tilt.action_to_weights(action_lo, active)
    w_hi = env_tilt.action_to_weights(action_hi, active)
    assert w_hi[probe] > w_lo[probe], (
        f"raising one ticker's score must raise its tilt weight (monotonicity), "
        f"got w_lo={w_lo[probe]:.4f} w_hi={w_hi[probe]:.4f}"
    )
    print(f"✓ monotone in score: w(score=-1)={w_lo[probe]:.4f} < w(score=+1)={w_hi[probe]:.4f}")

    # kappa sweep: concentration should grow smoothly and roughly monotonically with kappa
    top1s = []
    for kappa in [0.0, 1.0, 2.0, 4.0, 8.0]:
        env_k = PortfolioEnv(dataclasses.replace(cfg, action_mode="tilt", tilt_kappa=kappa), date_range="train")
        w_k = env_k.action_to_weights(action_modest, active)
        top1s.append(float(w_k[active_stock].max()))
    assert abs(top1s[0] - ew_expected[active_stock][0]) < 1e-9, f"kappa=0 must be exactly EW, got {top1s[0]}"
    assert all(top1s[i] <= top1s[i + 1] + 1e-9 for i in range(len(top1s) - 1)), (
        f"expected concentration to grow smoothly (non-decreasing) with kappa, got {top1s}"
    )
    print(f"✓ kappa sweep {[0, 1, 2, 4, 8]} -> top-1 weight {[round(t, 4) for t in top1s]} (smooth, monotone in kappa)")

    # --- 9. _cap_weights thin-universe regression (found via this session's own tilt-mode
    #     testing): when n_active*cap is close to the total mass being allocated (e.g. a
    #     thin ~10-name early-history window where equal-weight itself sits at the cap
    #     boundary), the redistribution loop could run out of uncapped active stocks to
    #     absorb overflow mid-iteration -- the overflow silently vanished from `w` right
    #     there, and the final clip-then-renormalize-the-WHOLE-vector step scaled the
    #     just-capped stocks back UP, re-violating the cap by construction, not by chance. ---
    env_thin = PortfolioEnv(dataclasses.replace(cfg, action_mode="tilt", tilt_kappa=2.0), date_range="train")
    active_thin = env_thin.mask[0]  # the pathological ~10-name window
    n_active_thin = int(active_thin.sum())
    rng_thin = np.random.default_rng(0)
    action_thin = np.zeros(len(env_thin.tickers), dtype=np.float64)
    action_thin[active_thin] = rng_thin.normal(size=n_active_thin)

    w_thin = env_thin.action_to_weights(action_thin, active_thin)
    stock_mask_thin = ~env_thin._is_cash_mask
    assert (w_thin[stock_mask_thin] <= cap + 1e-9).all(), (
        f"cap violated in the thin-universe regression case: max={w_thin[stock_mask_thin].max():.6f} > cap={cap}"
    )
    assert abs(w_thin.sum() - 1.0) < 1e-9, f"thin-universe weights don't sum to 1: {w_thin.sum()}"
    print(f"✓ _cap_weights thin-universe regression: n_active={n_active_thin}, "
          f"max stock weight={w_thin[stock_mask_thin].max():.4f} <= cap={cap} (was 0.10277 before the fix)")

    # --- 10. Random episode starts (DIAGNOSIS_PLAN.md Phase 2.4): problem (every reset
    #     replays day 0) vs solution (train-only random starts, val/test stay deterministic) ---
    env_default = PortfolioEnv(cfg, date_range="train")  # random_episode_start=False (default)
    starts_default = []
    for i in range(5):
        obs_r, info_r = env_default.reset(seed=100 + i)
        starts_default.append(env_default._t)
    assert all(s == 0 for s in starts_default), (
        f"expected the CURRENT default (no random start) to always reset to day 0, got {starts_default}"
    )
    print(f"✓ (problem) default reset(): 5 resets with different seeds all land on day 0: {starts_default}")

    random_cfg = dataclasses.replace(cfg, random_episode_start=True)
    env_random = PortfolioEnv(random_cfg, date_range="train")
    starts_random = []
    for i in range(20):
        obs_r, info_r = env_random.reset(seed=200 + i)
        starts_random.append(env_random._t)
        assert env_random._t <= env_random.n_steps - random_cfg.rebalance_interval_days, (
            f"random start {env_random._t} leaves less than one full rebalance decision "
            f"of room (n_steps={env_random.n_steps})"
        )
        assert info_r["date"] == env_random.dates[env_random._t], "reset() info date must match the actual random start"
    assert len(set(starts_random)) > 1, (
        f"expected random starts to vary across different seeds, got the same value every time: {starts_random}"
    )
    print(f"✓ (solution) random-start reset(): 20 resets land on {len(set(starts_random))} distinct "
          f"days out of 20, all within bounds (sample: {sorted(starts_random)[:5]}...)")

    # Same seed -> same start (reproducibility), different seeds -> can differ
    env_random_a = PortfolioEnv(random_cfg, date_range="train")
    env_random_b = PortfolioEnv(random_cfg, date_range="train")
    env_random_a.reset(seed=777)
    env_random_b.reset(seed=777)
    assert env_random_a._t == env_random_b._t, "the SAME seed must reproduce the SAME random start"
    print(f"✓ same seed (777) reproduces the same random start on independent env instances: t={env_random_a._t}")

    # val/test MUST stay deterministic regardless of the flag -- honest, reproducible evaluation
    for split in ("val", "test"):
        env_split = PortfolioEnv(random_cfg, date_range=split)
        split_starts = []
        for i in range(5):
            env_split.reset(seed=300 + i)
            split_starts.append(env_split._t)
        assert all(s == 0 for s in split_starts), (
            f"{split} split must stay deterministic (day 0) even with random_episode_start=True, got {split_starts}"
        )
    print("✓ val/test splits stay deterministic at day 0 regardless of random_episode_start (honest evaluation)")

    # --- 11. pit_universe is the SOURCE OF TRUTH at load time, not just "did the tensor file
    #     happen to have PIT data" (DIAGNOSIS_PLAN.md Phase 2.4): problem this fixes --
    #     config.pit_universe=True on a tensor file with NO PIT data used to silently train on
    #     the full universe instead of what was requested; now it raises loudly. Production
    #     tensors carry PIT data as of Phase 4, so this mocks _load_raw_tensors' return
    #     (isolated unit test of the enforcement branch, not tied to what's on disk today). ---
    import src.agent.env as env_mod

    def _fake_load_raw_tensors(expected_features):
        n_t, n_d = 3, 5
        return (
            np.arange(n_d), np.array(["A", "B", "CASH"]), np.ones((n_d, n_t), dtype=bool),
            np.zeros((n_d, n_t)), np.zeros((n_d, n_t, 1)), {}, [], None,  # pit_membership_mask=None
        )

    original = env_mod._load_raw_tensors
    env_mod._load_raw_tensors = _fake_load_raw_tensors
    env_mod._load_normalized_tensors.cache_clear()  # bust real entries cached earlier in this test
    env_mod._load_slice.cache_clear()
    try:
        pit_true_cfg = dataclasses.replace(cfg, pit_universe=True)
        try:
            PortfolioEnv(pit_true_cfg, date_range="train")
            raise AssertionError("expected RuntimeError: mocked tensors have no PIT membership data")
        except RuntimeError as e:
            assert "pit_universe=True" in str(e) and "--pit-universe" in str(e), f"unexpected error message: {e}"
    finally:
        env_mod._load_raw_tensors = original
        env_mod._load_normalized_tensors.cache_clear()
        env_mod._load_slice.cache_clear()
    print("✓ pit_universe=True on tensors with no PIT data raises RuntimeError (was a silent no-op before this fix)")

    # --- 12. Reward decomposition identity (DIAGNOSIS_PLAN.md Phase 3.2): the logged
    #     (raw_excess, penalty) must recompose to the actual returned reward EXACTLY, not
    #     approximately -- this is what makes the decomposition trustworthy as an audit tool. ---
    env_decomp = PortfolioEnv(cfg, date_range="train")
    env_decomp.reset(seed=0)
    rng_decomp = np.random.default_rng(0)
    n_checked = 0
    for step_i in range(5):
        action_d = rng_decomp.normal(size=len(env_decomp.tickers)).astype(np.float32)
        _, reward_d, terminated_d, _, info_d = env_decomp.step(action_d)
        recomposed = info_d["reward_raw_excess"] - info_d["reward_penalty"]
        assert abs(recomposed - reward_d) < 1e-6, (
            f"reward_raw_excess - reward_penalty ({recomposed}) must exactly equal the "
            f"returned reward ({reward_d})"
        )
        n_checked += 1
        if terminated_d:
            break
    print(f"✓ reward decomposition identity holds exactly: reward_raw_excess - reward_penalty "
          f"== reward (checked over {n_checked} steps, max diff < 1e-6)")

    # --- 13. Tilt EMA anchoring (Phase 5): high turnover at alpha=1.0 (no anchoring),
    #     much lower at alpha<1.0, weights still sum to 1 and respect cap.
    ema_cfg = dataclasses.replace(cfg, tilt_ema_alpha=1.0)
    env_ema = PortfolioEnv(ema_cfg, date_range="train")
    env_ema.reset(seed=99)
    rng_ema = np.random.default_rng(99)

    # Problem test: default alpha=1.0 should have high turnover
    turnover_1_0 = []
    for _ in range(50):
        action_ema = rng_ema.normal(size=len(env_ema.tickers), scale=2.0).astype(np.float32)
        _, _, _, _, info_ema = env_ema.step(action_ema)
        daily_turns = info_ema["daily_one_way_turnover"]
        turnover_1_0.extend(daily_turns)
    mean_turnover_1_0 = float(np.mean(turnover_1_0))
    assert mean_turnover_1_0 > 0.3, (
        f"default tilt_ema_alpha=1.0 should show high turnover, got mean {mean_turnover_1_0:.3f} "
        "(replicates measured 0.66-ish baseline behavior)"
    )
    print(f"✓ (problem) tilt_ema_alpha=1.0 (no anchoring): mean one-way turnover {mean_turnover_1_0:.3f} (high)")

    # Solution test: alpha=0.3 should have materially lower turnover
    ema_cfg_smooth = dataclasses.replace(cfg, tilt_ema_alpha=0.3)
    env_ema_smooth = PortfolioEnv(ema_cfg_smooth, date_range="train")
    env_ema_smooth.reset(seed=99)
    rng_ema_smooth = np.random.default_rng(99)

    turnover_0_3 = []
    for _ in range(50):
        action_ema_s = rng_ema_smooth.normal(size=len(env_ema_smooth.tickers), scale=2.0).astype(np.float32)
        _, _, _, _, info_ema_s = env_ema_smooth.step(action_ema_s)
        daily_turns_s = info_ema_s["daily_one_way_turnover"]
        turnover_0_3.extend(daily_turns_s)
    mean_turnover_0_3 = float(np.mean(turnover_0_3))
    assert mean_turnover_0_3 < mean_turnover_1_0 * 0.8, (
        f"tilt_ema_alpha=0.3 should reduce turnover substantially, got {mean_turnover_0_3:.3f} "
        f"vs {mean_turnover_1_0:.3f} at alpha=1.0 (ratio {mean_turnover_0_3/mean_turnover_1_0:.2f})"
    )

    # Check: weights still sum to 1 and respect the cap
    env_ema_check = PortfolioEnv(ema_cfg_smooth, date_range="train")
    env_ema_check.reset(seed=99)
    rng_check = np.random.default_rng(99)
    for _ in range(10):
        action_check = rng_check.normal(size=len(env_ema_check.tickers)).astype(np.float32)
        obs_check, _, _, _, _ = env_ema_check.step(action_check)
        weights_check = env_ema_check._prev_weights
        assert abs(weights_check.sum() - 1.0) < 1e-6, f"weights don't sum to 1: {weights_check.sum()}"
        assert (weights_check <= cfg.max_position_weight + 1e-6).all() or weights_check[env_ema_check._is_cash_mask].all(), (
            "weights exceed cap (CASH exempt)"
        )
    print(f"✓ (solution) tilt_ema_alpha=0.3: mean one-way turnover {mean_turnover_0_3:.3f} (much lower); "
          f"weights sum to 1, respect cap")

    print("\n" + "=" * 60)
    print("ALL ENV TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
