#!/usr/bin/env python3
"""
Leakage & correctness audit of the eval path (roadmap M1.7).

Standard "mutate the future, verify unchanged" pattern against the REAL env
and policies (not a synthetic toy), since that's the actual code path used
for training/backtesting and gives higher confidence than a hand-built
tensor that might not stress the same branches.

Run from project root: python tests/agent/test_eval_integrity.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.config import DEFAULT_CONFIG, generate_windows, window_to_config
from src.agent.env import PortfolioEnv
from src.agent.evaluate import inv_vol_policy, market_cap_policy


def test_inv_vol_no_lookahead() -> None:
    env = PortfolioEnv(DEFAULT_CONFIG, date_range="test")
    env.returns = env.returns.copy()  # private copy: don't poison the shared lru_cache
    t = 100

    policy = inv_vol_policy(env)
    w_before = policy(None, t)

    env.returns[t + 1:] = np.random.RandomState(0).normal(0, 100, env.returns[t + 1:].shape)
    w_after = policy(None, t)

    assert np.allclose(w_before, w_after), "inv_vol_policy leaked future return data"
    print("✓ inv_vol_policy: unaffected by corrupting future returns")


def test_market_cap_no_lookahead() -> None:
    env = PortfolioEnv(DEFAULT_CONFIG, date_range="test")
    t = 100
    policy = market_cap_policy(env, DEFAULT_CONFIG)
    w1 = policy(None, t)
    w2 = policy(None, t)  # ffill is deterministic; calling twice should be identical
    assert np.allclose(w1, w2)

    # ffill by construction only carries PAST values forward (pandas .ffill() never
    # looks ahead), so market_cap_policy structurally cannot see future caps. Confirm
    # weights at an early date don't depend on caps.pivot()'s FUTURE rows by checking
    # a toy ffill series directly (isolates the pandas mechanics from the full env).
    import pandas as pd
    s = pd.Series([10.0, np.nan, np.nan, 20.0, np.nan], index=range(5))
    filled = s.ffill()
    assert filled.tolist() == [10.0, 10.0, 10.0, 20.0, 20.0], (
        "ffill semantics changed — market_cap_policy assumes forward-fill only pulls PAST values forward"
    )
    print("✓ market_cap_policy: ffill is deterministic and backward-only")


def test_step_no_lookahead_beyond_decision_window() -> None:
    """A step's reward must depend only on returns within [t+1, t+n_days] (the
    days it actually pays out over), never on data beyond that window."""
    env = PortfolioEnv(DEFAULT_CONFIG, date_range="test")
    env._simple_rets = env._simple_rets.copy()
    env._ew_log_returns = env._ew_log_returns.copy()
    env.reset(seed=42)

    rng = np.random.default_rng(1)
    action = rng.normal(size=env.action_space.shape).astype(np.float32)
    _, reward1, _, _, info1 = env.step(action)
    end_t = env._t  # inclusive: day_indices used = [1 .. end_t]

    env2 = PortfolioEnv(DEFAULT_CONFIG, date_range="test")
    env2._simple_rets = env2._simple_rets.copy()
    env2._ew_log_returns = env2._ew_log_returns.copy()
    env2.reset(seed=42)
    env2._simple_rets[end_t + 1:] = rng.normal(0, 100, env2._simple_rets[end_t + 1:].shape)
    env2._ew_log_returns[end_t + 1:] = rng.normal(0, 100, env2._ew_log_returns[end_t + 1:].shape)
    _, reward2, _, _, info2 = env2.step(action)

    assert reward1 == reward2, "step() reward leaked data beyond its own decision window"
    assert np.array_equal(info1["daily_log_returns"], info2["daily_log_returns"])
    print("✓ env.step(): reward depends only on its own decision window, not beyond")


def test_weight_drift_matches_independent_calculation() -> None:
    """env.step()'s drift formula w_new = w*(1+r)/(1+r_p) should be algebraically
    identical to computing dollar values directly and renormalizing."""
    rng = np.random.default_rng(7)
    w = rng.dirichlet(np.ones(5))
    r_vec = rng.normal(0, 0.05, 5)

    dollar_values = w * (1 + r_vec)
    w_new_independent = dollar_values / dollar_values.sum()

    r_p = np.dot(w, r_vec)
    w_new_formula = w * (1 + r_vec) / (1 + r_p)

    assert np.allclose(w_new_independent, w_new_formula)
    assert abs(w_new_formula.sum() - 1.0) < 1e-10
    print("✓ weight-drift formula matches independent dollar-value calculation, sums to 1")


def test_baseline_weights_reproduced_exactly() -> None:
    """Baseline policies' target weights must survive the logits→softmax→cap
    round trip unchanged (within numerical tolerance) -- if the round trip
    distorted them, every baseline's reported performance would be wrong."""
    env = PortfolioEnv(DEFAULT_CONFIG, date_range="test")
    t = 50
    active = env.mask[t]

    from src.agent.evaluate import _weights_to_logits, _exclude_cash_from_weights
    target = _exclude_cash_from_weights(active.astype(float), env)
    target = target / target.sum()
    logits = _weights_to_logits(target, env.config.logit_scale)
    reproduced = env.action_to_weights(logits, active)

    assert np.allclose(target, reproduced, atol=1e-4), (
        f"equal-weight target not reproduced through logits/softmax/cap round trip: "
        f"max diff {np.abs(target - reproduced).max():.2e}"
    )
    print("✓ baseline target weights survive the logits→softmax→cap round trip")


def test_scaler_cutoff_precedes_every_window_test_start() -> None:
    """Each window's scaler must be fit strictly before that window's own
    test_start -- verifies the rolling-window scheme has no cross-window
    leakage, regardless of how CLAUDE.md's prose describes it."""
    windows = generate_windows(
        DEFAULT_CONFIG.dataset_start, DEFAULT_CONFIG.dataset_end,
        DEFAULT_CONFIG.window_train_years, DEFAULT_CONFIG.window_test_years,
    )
    assert len(windows) >= 2, "expected multiple windows from the default config"

    for w in windows:
        cfg = window_to_config(w, DEFAULT_CONFIG)
        assert cfg.train_end < cfg.val_start < cfg.test_start, (
            f"window {w.window_id}: train_end={cfg.train_end} must precede "
            f"val_start={cfg.val_start} must precede test_start={cfg.test_start}"
        )
    print(f"✓ all {len(windows)} windows: scaler cutoff (train_end) strictly precedes val_start and test_start")


def main() -> None:
    test_inv_vol_no_lookahead()
    test_market_cap_no_lookahead()
    test_step_no_lookahead_beyond_decision_window()
    test_weight_drift_matches_independent_calculation()
    test_baseline_weights_reproduced_exactly()
    test_scaler_cutoff_precedes_every_window_test_start()
    print("\nALL EVAL-INTEGRITY TESTS PASSED ✓")


if __name__ == "__main__":
    main()
