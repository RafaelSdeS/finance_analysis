"""
Behavior-cloning warm start: pretrain the PPO policy to imitate a supervised
ranker before RL fine-tuning, so PPO starts already knowing the cross-sectional
ranking signal instead of discovering it from ~100-150 sparse 21-day rewards.

Teacher: HistGradientBoostingRegressor fit on a window's train span only (same
no-lookahead discipline as everything else in this codebase) using the exact
scaled per-ticker features the policy already sees (env.features), predicting
forward log returns. Its top-quintile equal-weight output is rolled through
the env to produce (obs, target_logits, reward) triples, then the PPO
policy's actor path (mlp_extractor.policy_net + action_net) is fit to the
logits via masked MSE, and the critic path (mlp_extractor.value_net +
value_net) is fit to discounted returns-to-go from the same rollout. Without
the critic warm start, PPO's first rollouts see a confident (BC-trained)
actor paired with a random critic — noisy advantage estimates then drag the
actor into runaway concentration before GAE calibrates (observed: effective_n
collapsing to ~1 name, t-stat -3.06 vs equal-weight).

Plain BC, not DAgger: the dataset reflects states the teacher visits, not
future PPO states. Acceptable since both start near-equal-weight; revisit
with DAgger iterations if BC alone proves insufficient.
"""

import logging
from typing import Callable

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from stable_baselines3 import PPO

from src.agent.env import PortfolioEnv
from src.agent.evaluate import _weights_to_logits

logger = logging.getLogger(__name__)


def train_teacher(env: PortfolioEnv, horizon: int = 21, seed: int = 42) -> HistGradientBoostingRegressor:
    """Fit a ranker on this window's train span: env.features[t,i] -> forward `horizon`-day log return.

    Uses env._simple_rets (zero-if-inactive, same convention the env's own
    equal-weight reward baseline uses) so the target is well-defined even
    across masked/inactive days.
    """
    D, N, F = env.features.shape
    cash_idx = np.where(env._is_cash_mask)[0]
    cash_idx = cash_idx[0] if len(cash_idx) else -1

    X_rows, y_rows = [], []
    for t in range(D - horizon):
        active_t = env.mask[t]
        # Forward cumulative log return per ticker over [t+1, t+horizon]
        fwd_simple = env._simple_rets[t + 1 : t + horizon + 1]  # [horizon, N]
        fwd_log = np.log1p(fwd_simple).sum(axis=0)  # [N]
        for i in range(N):
            if i == cash_idx or not active_t[i]:
                continue
            X_rows.append(env.features[t, i, :])
            y_rows.append(fwd_log[i])

    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    logger.info("BC teacher: training on %d (day,ticker) rows", len(X))

    model = HistGradientBoostingRegressor(max_iter=100, random_state=seed)
    model.fit(X, y)
    return model


def teacher_scores_to_weights(scores: np.ndarray, active: np.ndarray, top_frac: float = 0.2) -> np.ndarray:
    """Top-quintile equal-weight over active names (min 5, else fall back to all-active EW).

    Mirrors ranker_baseline.portfolio_simulator's selection rule.
    """
    n = len(scores)
    weights = np.zeros(n, dtype=np.float64)
    active_idx = np.where(active)[0]
    if len(active_idx) == 0:
        return weights

    active_scores = scores[active_idx]
    k = max(int(len(active_idx) * top_frac), 1)
    top_local = np.argsort(active_scores)[-k:]
    top_idx = active_idx[top_local]

    if len(top_idx) >= 5:
        weights[top_idx] = 1.0 / len(top_idx)
    else:
        weights[active_idx] = 1.0 / len(active_idx)
    return weights


def teacher_policy(env: PortfolioEnv, model: HistGradientBoostingRegressor, top_frac: float = 0.2) -> Callable:
    """act(obs, t) -> logits, matching evaluate.py's policy-function convention."""
    cash_idx = np.where(env._is_cash_mask)[0]
    cash_idx = cash_idx[0] if len(cash_idx) else -1

    def act(obs: np.ndarray, t: int) -> np.ndarray:
        active = env.mask[t].copy()
        if cash_idx >= 0:
            active[cash_idx] = False
        scores = model.predict(env.features[t])
        weights = teacher_scores_to_weights(scores, active, top_frac)
        return _weights_to_logits(weights, env.config.logit_scale)

    return act


def collect_bc_dataset(env: PortfolioEnv, act_fn: Callable) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll the teacher through env, recording (obs, target_logits, active_mask, reward) per decision.

    One row per env.step() call (a 21-day decision), not per day — matches
    what the PPO policy is actually queried on.
    """
    obs, _ = env.reset()
    obs_rows, action_rows, mask_rows, reward_rows = [], [], [], []
    terminated = False
    while not terminated:
        t = env._t
        action = act_fn(obs, t)
        obs_rows.append(obs)
        action_rows.append(action)
        mask_rows.append(env.mask[t].astype(np.float32))
        obs, reward, terminated, _, _ = env.step(action)
        reward_rows.append(reward)

    return (
        np.asarray(obs_rows, dtype=np.float32),
        np.asarray(action_rows, dtype=np.float32),
        np.asarray(mask_rows, dtype=np.float32),
        np.asarray(reward_rows, dtype=np.float32),
    )


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Return-to-go R_t = sum_k gamma^(k-t) * reward_k, for a single episode."""
    returns = np.empty_like(rewards, dtype=np.float32)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def pretrain_policy(
    model: PPO,
    obs_arr: np.ndarray,
    action_arr: np.ndarray,
    mask_arr: np.ndarray,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 42,
) -> None:
    """Fit the policy's actor path (not the value head) to imitate target logits via masked MSE."""
    device = model.policy.device
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
    target_t = torch.as_tensor(action_arr, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask_arr, dtype=torch.float32, device=device)

    actor_params = list(model.policy.mlp_extractor.policy_net.parameters()) + list(
        model.policy.action_net.parameters()
    )
    optimizer = torch.optim.Adam(actor_params, lr=lr)

    n = obs_t.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    final_loss = None
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=generator)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            obs_batch = obs_t[idx]
            target_batch = target_t[idx]
            mask_batch = mask_t[idx]

            features = model.policy.extract_features(obs_batch)
            latent_pi = model.policy.mlp_extractor.forward_actor(features)
            mean_actions = model.policy.action_net(latent_pi)

            loss = ((mean_actions - target_batch) ** 2 * mask_batch).sum() / mask_batch.sum().clamp(min=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / max(1, (n // batch_size))

    logger.info("BC pretrain: %d epochs, final masked-MSE loss=%.5f", epochs, final_loss)


def pretrain_value(
    model: PPO,
    obs_arr: np.ndarray,
    returns_arr: np.ndarray,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 42,
) -> None:
    """Fit the policy's critic path to the teacher rollout's discounted returns-to-go.

    Closes the actor/critic mismatch: without this, PPO's first advantage
    estimates come from a random value_net paired against an already-confident
    (BC-trained) actor, and noisy advantages drag the actor into runaway
    concentration before GAE calibrates.
    """
    device = model.policy.device
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
    target_t = torch.as_tensor(returns_arr, dtype=torch.float32, device=device).reshape(-1, 1)

    critic_params = list(model.policy.mlp_extractor.value_net.parameters()) + list(
        model.policy.value_net.parameters()
    )
    optimizer = torch.optim.Adam(critic_params, lr=lr)

    n = obs_t.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    final_loss = None
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=generator)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            obs_batch = obs_t[idx]
            target_batch = target_t[idx]

            features = model.policy.extract_features(obs_batch)
            latent_vf = model.policy.mlp_extractor.forward_critic(features)
            values = model.policy.value_net(latent_vf)

            loss = torch.nn.functional.mse_loss(values, target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / max(1, (n // batch_size))

    logger.info("BC value pretrain: %d epochs, final MSE loss=%.5f", epochs, final_loss)
