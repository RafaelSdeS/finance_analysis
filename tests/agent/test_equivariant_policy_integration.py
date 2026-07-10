"""
SB3-integration tests for EquivariantActorCriticPolicy (needs real data/tensors —
data group). Verifies the full policy constructs correctly against PortfolioEnv's actual
observation/action spaces, deterministic prediction stays equivariant end-to-end (not just
in the standalone extractor), and bc_pretrain.py's existing BC warm-start code works
UNMODIFIED against it (the whole point of matching MlpExtractor's attribute names).
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.config import DEFAULT_CONFIG
from src.agent.env import PortfolioEnv
from src.agent.policy import EquivariantActorCriticPolicy


def main() -> None:
    # Must match whatever agent_tensors.npz was actually built with (checked at PortfolioEnv
    # construction time) -- DEFAULT_CONFIG.universe_size is None, but the on-disk tensors
    # were built with --universe-size 50, so match that rather than requesting a rebuild.
    config = dataclasses.replace(DEFAULT_CONFIG, universe_size=50, total_timesteps=1)
    env = PortfolioEnv(config, date_range="train")
    n_tickers = len(env.tickers)
    n_features = len(config.state_features)

    print(f"Test 1: PPO constructs with EquivariantActorCriticPolicy (n_tickers={n_tickers}, n_features={n_features})")
    model = PPO(
        EquivariantActorCriticPolicy,
        env,
        n_steps=64, batch_size=32, n_epochs=1,
        policy_kwargs=dict(n_tickers=n_tickers, n_features=n_features, encoder_hidden_dim=16, encoder_dim=8),
        seed=0, device="cpu", verbose=0,
    )
    assert model.policy.mlp_extractor.latent_dim_pi == n_tickers
    assert isinstance(model.policy.action_net, torch.nn.Identity), "action_net must be replaced with Identity"
    print(f"  ✓ constructed; action_net is Identity (latent_dim_pi={model.policy.mlp_extractor.latent_dim_pi})")

    print("\nTest 2: model.predict() runs and returns a correctly-shaped action")
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (n_tickers,), f"expected shape ({n_tickers},), got {action.shape}"
    print(f"  ✓ predict() returns shape {action.shape}")

    print("\nTest 3: deterministic prediction is EXACTLY equivariant end-to-end through the full SB3 policy")
    rng = np.random.default_rng(3)
    perm = rng.permutation(n_tickers)
    feat_dim = n_tickers * n_features
    obs2 = obs.copy()
    feat, mask, prev_w = obs2[:feat_dim].reshape(n_tickers, n_features), obs2[feat_dim:feat_dim + n_tickers], obs2[feat_dim + n_tickers:]
    obs_perm = np.concatenate([feat[perm].ravel(), mask[perm], prev_w[perm]]).astype(np.float32)

    action_orig, _ = model.predict(obs, deterministic=True)
    action_perm, _ = model.predict(obs_perm, deterministic=True)
    max_err = np.abs(action_perm - action_orig[perm]).max()
    assert max_err < 1e-4, f"expected end-to-end equivariance through the full SB3 policy, got max error {max_err}"
    print(f"  ✓ predict(permute(obs)) == permute(predict(obs)) end-to-end: max error {max_err:.2e}")

    print("\nTest 4: bc_pretrain.py's existing BC warm-start code runs UNMODIFIED against this policy")
    from src.agent.bc_pretrain import (
        collect_bc_dataset, discounted_returns, pretrain_policy, pretrain_value, teacher_policy, train_teacher,
    )

    teacher_model = train_teacher(env, horizon=5, seed=0)  # small horizon: universe_size=5 has few train dates
    obs_arr, action_arr, mask_arr, reward_arr = collect_bc_dataset(env, teacher_policy(env, teacher_model))
    assert len(obs_arr) > 0, "BC dataset collection produced no rows"

    before = {n: p.clone() for n, p in model.policy.mlp_extractor.policy_net.named_parameters()}
    pretrain_policy(model, obs_arr, action_arr, mask_arr, epochs=2, seed=0)
    after = dict(model.policy.mlp_extractor.policy_net.named_parameters())
    changed = any(not torch.equal(before[n], after[n].detach()) for n in before)
    assert changed, "pretrain_policy ran but no policy_net parameters changed"
    print(f"  ✓ pretrain_policy: {len(obs_arr)} BC rows, policy_net parameters updated")

    returns_arr = discounted_returns(reward_arr, gamma=0.99)
    before_v = {n: p.clone() for n, p in model.policy.mlp_extractor.value_net.named_parameters()}
    pretrain_value(model, obs_arr, returns_arr, epochs=2, seed=0)
    after_v = dict(model.policy.mlp_extractor.value_net.named_parameters())
    changed_v = any(not torch.equal(before_v[n], after_v[n].detach()) for n in before_v)
    assert changed_v, "pretrain_value ran but no value_net parameters changed"
    print("  ✓ pretrain_value: value_net parameters updated")

    print("\n✓ All equivariant_policy_integration tests passed")


if __name__ == "__main__":
    main()
