#!/usr/bin/env python
"""Policy geometry + generalization diagnostics for a trained agent.

Consolidates the three ad-hoc checks from the M5 diagnosis (2026-07-09) into
one reusable tool, so re-running them for a new model (e.g. each point in the
logit_scale sweep) is one command instead of three throwaway scripts:

  - concentration: pre-cap vs post-cap top-1 stock weight and softmax entropy.
    Pre-cap near-argmax (found: mean 83% on a single stock, entropy 16% of
    max at logit_scale=10) means the 0.10 cap is doing all the diversification
    work, not the policy.
  - overfit check (M5.2): excess-of-SELIC Sharpe on the model's own TRAIN
    split (memorization trivially available) vs its held-out TEST split.
    In-sample >> out-of-sample is the generalization-gap signature.
  - return attribution (M5.5): splits the test-split return into SELIC carry,
    market exposure, and a stock-selection residual (src/agent/attribution.py)
    -- the residual is the actual "does this agent have stock-picking skill"
    number, not just a single cumulative-return figure.

Usage:
    python tools/policy_diagnostics.py                        # current production model
    python tools/policy_diagnostics.py --model path/to.zip \
        --train-years 24 --test-years 2                       # match a specific window scheme
"""

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root, for `src.*` imports

import numpy as np
from stable_baselines3 import PPO

from src.agent.attribution import decompose_returns
from src.agent.config import DEFAULT_CONFIG, generate_windows, window_to_config
from src.agent.env import PortfolioEnv
from src.agent.evaluate import agent_policy, equal_weight_policy, rollout, selic_policy
from src.agent.metrics import compute_all
from src.agent.model_provenance import SEMANTIC_FIELDS, check_sidecar


def _resolve_config(train_years: int | None, test_years: int | None, model_path: Path):
    """Default config (current production window) unless a window scheme is given,
    in which case reconstruct its first (only, if train+test years span the dataset)
    window -- same pattern used throughout the M5 diagnosis. Then override the
    semantic fields (logit_scale, rebalance_interval_days, max_position_weight,
    transaction_cost_bps) from the MODEL'S OWN provenance sidecar -- without this,
    a swept model (e.g. trained with --logit-scale 2) gets evaluated with whatever
    logit_scale the reconstructed config happens to default to, silently invalidating
    every downstream number. Caught this exact bug in the first sweep run."""
    if train_years is None and test_years is None:
        config = DEFAULT_CONFIG
    else:
        base = dataclasses.replace(
            DEFAULT_CONFIG,
            window_train_years=train_years or DEFAULT_CONFIG.window_train_years,
            window_test_years=test_years or DEFAULT_CONFIG.window_test_years,
        )
        windows = generate_windows(base.dataset_start, base.dataset_end,
                                   base.window_train_years, base.window_test_years)
        config = window_to_config(windows[0], base)

    sidecar = check_sidecar(model_path, config)
    if sidecar is not None:
        semantic_overrides = {f: sidecar[f] for f in SEMANTIC_FIELDS if f in sidecar}
        config = dataclasses.replace(config, **semantic_overrides)
    return config


def concentration_check(model: PPO, config, env: PortfolioEnv) -> dict:
    """Roll the model through env's own split, comparing pre-cap softmax weights
    (what the policy itself expresses) to post-cap weights (what the 0.10 cap's
    redistribution actually produces)."""
    obs, _ = env.reset()
    stock_mask = ~env._is_cash_mask
    pre_top1, post_top1, entropies = [], [], []
    terminated = False
    while not terminated:
        t = env._t
        action, _ = model.predict(obs, deterministic=True)
        active = env.mask[t]
        pre_cap = env._masked_softmax(np.asarray(action, dtype=np.float64) * config.logit_scale, active)
        post_cap = env._cap_weights(pre_cap.astype(np.float32), active)
        stock_active = active & stock_mask
        pre_top1.append(pre_cap[stock_active].max())
        post_top1.append(post_cap[stock_active].max())
        p = pre_cap[active]
        entropies.append(float(-(p * np.log(p + 1e-12)).sum()))
        obs, _, terminated, _, _ = env.step(action)
    mean_n_active = float(env.mask.sum(axis=1).mean())
    return {
        "pre_cap_top1_mean": float(np.mean(pre_top1)),
        "post_cap_top1_mean": float(np.mean(post_top1)),
        "entropy_mean": float(np.mean(entropies)),
        "entropy_max_possible": float(np.log(mean_n_active)),
    }


def excess_sharpe_on_split(model: PPO, config, date_range: str) -> tuple[dict, dict, dict]:
    """Roll the model + SELIC baseline through a given split, return
    (metrics, agent_rollout, selic_rollout) -- callers reuse the rollouts to
    avoid re-simulating for attribution."""
    env = PortfolioEnv(config, date_range=date_range)
    agent_res = rollout(env, agent_policy(model))
    selic_env = PortfolioEnv(config, date_range=date_range)
    selic_res = rollout(selic_env, selic_policy(selic_env, config))
    metrics = compute_all(agent_res["rewards"], agent_res["values"], agent_res["weights"],
                          selic_log_returns=selic_res["rewards"])
    return metrics, agent_res, selic_res


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy geometry + generalization diagnostics")
    parser.add_argument("--model", type=Path, default=DEFAULT_CONFIG.model_dir / "agent_best.zip")
    parser.add_argument("--train-years", type=int, default=None, help="Match a specific window scheme's train-years")
    parser.add_argument("--test-years", type=int, default=None, help="Match a specific window scheme's test-years")
    args = parser.parse_args()

    config = _resolve_config(args.train_years, args.test_years, args.model)
    model = PPO.load(args.model, device=config.device)

    print(f"Model: {args.model}")
    print(f"Window: train {config.train_start}->{config.train_end}, test {config.test_start}->{config.test_end}")
    print(f"logit_scale: {config.logit_scale}")
    print()

    conc = concentration_check(model, config, PortfolioEnv(config, date_range="test"))
    print("Concentration (test split):")
    print(f"  pre-cap top-1 weight:  mean={conc['pre_cap_top1_mean']:.4f}")
    print(f"  post-cap top-1 weight: mean={conc['post_cap_top1_mean']:.4f}")
    print(f"  entropy: mean={conc['entropy_mean']:.4f} "
          f"({conc['entropy_mean'] / conc['entropy_max_possible']:.2%} of max)")
    print()

    test_metrics, test_agent_res, test_selic_res = excess_sharpe_on_split(model, config, "test")
    train_metrics, _, _ = excess_sharpe_on_split(model, config, "train")
    print("Overfit check (excess-of-SELIC Sharpe):")
    print(f"  in-sample (train):    {train_metrics.get('excess_sharpe', float('nan')):+.4f}")
    print(f"  out-of-sample (test): {test_metrics.get('excess_sharpe', float('nan')):+.4f}")
    print()

    ew_env = PortfolioEnv(config, date_range="test")
    ew_res = rollout(ew_env, equal_weight_policy(ew_env))
    cash_idx = int(np.where(ew_env.tickers == "CASH")[0][0])
    attribution = decompose_returns(
        agent_log_returns=test_agent_res["rewards"],
        cash_weights=test_agent_res["weights"][:, cash_idx],
        selic_log_returns=test_selic_res["rewards"],
        ew_stocks_log_returns=ew_res["rewards"],
    )
    print("Return attribution (test split):")
    print(f"  agent cumulative:      {attribution['agent_cumulative']:+.4f}")
    print(f"  -> carry (isolated):   {attribution['carry_cumulative']:+.4f}  "
          f"(mean cash weight={attribution['mean_cash_weight']:.2%})")
    print(f"  -> market exposure:    {attribution['market_exposure_cumulative']:+.4f}")
    print(f"  -> selection residual: {attribution['selection_residual_cumulative']:+.4f}  <-- the skill number")


if __name__ == "__main__":
    main()
