#!/usr/bin/env python3
"""
bc_pretrain.pretrain_policy: synthetic check that masked-MSE imitation
actually reduces loss (no data files needed — exercises the SB3-internals
call sequence, the highest-risk part of the BC warm-start).

Run from project root: python tests/agent/test_bc_pretrain.py
"""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.bc_pretrain import discounted_returns, pretrain_policy, pretrain_value

N_TICKERS = 6
N_FEATURES = 4
OBS_DIM = N_TICKERS * N_FEATURES + 2 * N_TICKERS


class _DummyPortfolioEnv(gym.Env):
    """Minimal stand-in with PortfolioEnv's obs/action space shapes only."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-10.0, 10.0, shape=(N_TICKERS,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(OBS_DIM, dtype=np.float32), 0.0, True, False, {}


def _masked_mse(model: PPO, obs_t: torch.Tensor, target_t: torch.Tensor, mask_t: torch.Tensor) -> float:
    with torch.no_grad():
        features = model.policy.extract_features(obs_t)
        latent_pi = model.policy.mlp_extractor.forward_actor(features)
        mean_actions = model.policy.action_net(latent_pi)
        loss = ((mean_actions - target_t) ** 2 * mask_t).sum() / mask_t.sum().clamp(min=1)
    return loss.item()


def main() -> None:
    print("=" * 60)
    print("TEST: bc_pretrain.pretrain_policy")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n_rows = 200
    obs_arr = rng.normal(size=(n_rows, OBS_DIM)).astype(np.float32)
    action_arr = rng.uniform(-1.0, 1.0, size=(n_rows, N_TICKERS)).astype(np.float32)
    mask_arr = (rng.uniform(size=(n_rows, N_TICKERS)) > 0.2).astype(np.float32)  # ~80% active

    env = _DummyPortfolioEnv()
    model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", seed=42)

    device = model.policy.device
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
    target_t = torch.as_tensor(action_arr, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask_arr, dtype=torch.float32, device=device)

    loss_before = _masked_mse(model, obs_t, target_t, mask_t)
    pretrain_policy(model, obs_arr, action_arr, mask_arr, epochs=60, lr=3e-3, batch_size=64, seed=42)
    loss_after = _masked_mse(model, obs_t, target_t, mask_t)

    print(f"  masked-MSE before: {loss_before:.5f}")
    print(f"  masked-MSE after:  {loss_after:.5f}")
    assert loss_after < loss_before * 0.5, (
        f"BC pretraining should substantially reduce masked-MSE loss "
        f"(before={loss_before:.5f}, after={loss_after:.5f})"
    )
    print("✓ pretrain_policy reduces masked-MSE loss on synthetic targets")

    # Value head must be untouched by pretrain_policy (actor-only)
    value_params_before = [p.clone() for p in model.policy.value_net.parameters()]
    pretrain_policy(model, obs_arr, action_arr, mask_arr, epochs=1, lr=1e-3, batch_size=64, seed=1)
    for before, after in zip(value_params_before, model.policy.value_net.parameters()):
        assert torch.equal(before, after), "value_net should not be modified by pretrain_policy"
    print("✓ value_net untouched by BC pretraining")

    print("\n✓ Test 2b: log_std recalibrated to match the BC target spread")
    # Regression test for the "PPO applies zero gradient updates after BC-pretrain" bug:
    # leaving log_std at its generic init (unrelated to how differentiated the newly-fit
    # mean-network's outputs are) made the very first post-BC-pretrain minibatch's KL
    # divergence explode (observed empirically: 7.69, ~171x the target_kl=0.03 break
    # threshold), so PPO's trust-region safety valve aborted before ever applying a
    # gradient step -- across 21 consecutive rollouts (69% of a 1M-timestep budget) in a
    # smoke test. Fix: recalibrate log_std to np.log(std of the masked target logits).
    env2 = _DummyPortfolioEnv()
    model2 = PPO("MlpPolicy", env2, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", seed=42)
    log_std_before = float(model2.policy.log_std.data.mean().item())

    # action_arr targets are Uniform(-1, 1) -> std ≈ 1/sqrt(3) ≈ 0.577
    expected_log_std = float(np.log(action_arr[mask_arr.astype(bool)].std()))
    pretrain_policy(model2, obs_arr, action_arr, mask_arr, epochs=1, lr=1e-3, batch_size=64, seed=1)
    log_std_after = float(model2.policy.log_std.data.mean().item())

    print(f"  log_std before: {log_std_before:.4f} (sigma={np.exp(log_std_before):.4f})")
    print(f"  log_std after:  {log_std_after:.4f} (sigma={np.exp(log_std_after):.4f}), "
          f"expected ≈ {expected_log_std:.4f}")
    assert abs(log_std_after - expected_log_std) < 0.05, (
        f"log_std should be recalibrated to ≈log(target_std)={expected_log_std:.4f}, "
        f"got {log_std_after:.4f}"
    )
    assert log_std_after != log_std_before, "log_std should change from its generic init value"
    print("✓ pretrain_policy recalibrates log_std to match the actor's newly-learned output spread")

    print("\n✓ Test 3: discounted_returns computes correct return-to-go")
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    returns = discounted_returns(rewards, gamma=0.5)
    expected = np.array([1.0 + 0.5 * (1.0 + 0.5 * 1.0), 1.0 + 0.5 * 1.0, 1.0], dtype=np.float32)
    assert np.allclose(returns, expected), f"expected {expected}, got {returns}"
    print("✓ discounted_returns matches hand-computed return-to-go")

    print("\n✓ Test 4: pretrain_value reduces MSE loss and leaves the actor untouched")
    returns_arr = rng.normal(size=n_rows).astype(np.float32)

    def _value_mse():
        with torch.no_grad():
            features = model.policy.extract_features(obs_t)
            latent_vf = model.policy.mlp_extractor.forward_critic(features)
            values = model.policy.value_net(latent_vf)
        return torch.nn.functional.mse_loss(values, torch.as_tensor(returns_arr).reshape(-1, 1)).item()

    value_loss_before = _value_mse()
    actor_params_before = [p.clone() for p in model.policy.action_net.parameters()]
    pretrain_value(model, obs_arr, returns_arr, epochs=60, lr=3e-3, batch_size=64, seed=42)
    value_loss_after = _value_mse()
    print(f"  value MSE before: {value_loss_before:.5f}")
    print(f"  value MSE after:  {value_loss_after:.5f}")
    assert value_loss_after < value_loss_before * 0.5, (
        f"BC value pretraining should substantially reduce MSE loss "
        f"(before={value_loss_before:.5f}, after={value_loss_after:.5f})"
    )
    for before, after in zip(actor_params_before, model.policy.action_net.parameters()):
        assert torch.equal(before, after), "action_net should not be modified by pretrain_value"
    print("✓ pretrain_value reduces MSE loss and leaves the actor untouched")

    print("\n✓ All bc_pretrain tests passed")


if __name__ == "__main__":
    main()
