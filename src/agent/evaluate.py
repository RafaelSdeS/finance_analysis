"""
Backtesting: trained agent vs baselines on the TEST split.

All strategies (agent + baselines) are rolled through the same PortfolioEnv,
so portfolio math lives in exactly one place (env.step). Baselines supply
target weights, converted to logits via log(w) so the env's masked softmax
reproduces them exactly.

Baselines:
  equal_weight : 1/n_active, rebalanced daily
  market_cap   : w ∝ market cap (forward-filled, known at time t)
  inv_vol      : w ∝ 1 / trailing 60d return std (trailing window only)

Usage:
    python -m src.agent.evaluate                             # uses agent_best.zip
    python -m src.agent.evaluate --model artifacts/models/agent_final.zip
"""

import argparse
import json
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.agent.config import AgentConfig, DEFAULT_CONFIG, configure_logging
from src.agent.env import PortfolioEnv
from src.agent.metrics import (
    compute_all, sharpe_ratio, max_drawdown, TRADING_DAYS,
    probabilistic_sharpe_ratio, deflated_sharpe_ratio,
)

logger = logging.getLogger(__name__)

VOL_WINDOW = 60  # trailing days for inverse-volatility baseline


# --------------------------------------------------------------- rollouts

def _weights_to_logits(weights: np.ndarray, logit_scale: float) -> np.ndarray:
    """Logits whose temperature-scaled masked softmax reproduces the given weights.

    The env computes softmax(action * logit_scale), so pre-divide by the scale
    to make baseline weights come out exactly as intended.
    """
    return (np.log(np.maximum(weights, 1e-12)) / logit_scale).astype(np.float32)


def _exclude_cash_from_weights(w: np.ndarray, env: PortfolioEnv) -> np.ndarray:
    """
    Zero out CASH weight before normalization (exclude from baselines).
    Baselines stay pure-equity; only the RL agent gets CASH as an option.
    """
    if "CASH" in env.tickers:
        w = w.copy()
        cash_idx = np.where(env.tickers == "CASH")[0][0]
        w[cash_idx] = 0.0
    return w


def rollout(env: PortfolioEnv, act_fn: Callable[[np.ndarray, int], np.ndarray]) -> dict:
    """Roll a policy through an env split. act_fn(obs, env._t) → action logits.

    With N-day steps, one env.step() spans N days. Unpacks daily arrays from info["daily_*"]
    to maintain daily-frequency rewards/dates/weights for backward-compatible backtesting.
    """
    obs, _ = env.reset()
    rewards, values, weights_log, dates, costs = [], [env.config.initial_capital], [], [], []
    terminated = False
    while not terminated:
        t_dec = env._t  # Day index at decision time (passed to act_fn)
        action = act_fn(obs, t_dec)
        obs, reward, terminated, _, info = env.step(action)

        # Unpack daily arrays from this N-day step
        daily_rets = info["daily_log_returns"]  # [n_days]
        daily_dates = info["daily_dates"]        # [n_days]
        daily_weights = info["daily_weights"]    # [n_days, n_tickers]
        cost = info["transaction_cost"]          # Scalar (applied on day 1)

        # Extend daily-frequency lists
        rewards.extend(daily_rets)
        dates.extend(daily_dates)
        weights_log.extend(daily_weights)

        # Costs: cost on first day, 0 on rest
        costs.append(cost)
        for _ in range(len(daily_rets) - 1):
            costs.append(0.0)

        # Reconstruct daily values by compounding daily returns
        for d_ret in daily_rets:
            values.append(values[-1] * np.exp(d_ret))

    return {
        "rewards": np.array(rewards, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),
        "weights": np.array(weights_log, dtype=np.float32),   # [T, n_tickers]
        "dates": pd.DatetimeIndex(dates),
        "costs": np.array(costs, dtype=np.float32),            # [T] — cost on rebalance day, 0 else
    }


def agent_policy(model: PPO) -> Callable:
    def act(obs: np.ndarray, t: int) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action
    return act


def equal_weight_policy(env: PortfolioEnv) -> Callable:
    def act(obs: np.ndarray, t: int) -> np.ndarray:
        active = env.mask[t].astype(float)
        active = _exclude_cash_from_weights(active, env)
        if active.sum() == 0:  # shouldn't happen, but be safe
            active = env.mask[t].astype(float)
        return _weights_to_logits(active / active.sum(), env.config.logit_scale)
    return act


def agent_vs_equal_weight(env: PortfolioEnv, model: PPO) -> dict:
    """Agent rollout vs equal-weight rollout on the same env split; excess = day-by-day diff.

    excess_sharpe is the Sharpe of (agent absolute log return − EW absolute log return) per
    day — NOT a difference of two independently-computed Sharpes.
    """
    agent_res = rollout(env, agent_policy(model))
    ew_res = rollout(env, equal_weight_policy(env))
    excess = agent_res["rewards"] - ew_res["rewards"]
    return {
        "sharpe": sharpe_ratio(agent_res["rewards"]),
        "excess_sharpe": sharpe_ratio(excess),
        "max_drawdown": max_drawdown(agent_res["values"]),
        "final_value": agent_res["values"][-1],
    }


@lru_cache(maxsize=1)
def _market_cap_pivot(dataset_path: Path) -> pd.DataFrame:
    """Full-history ffilled market-cap pivot; read+pivot the 180MB parquet once per process."""
    df = pd.read_parquet(dataset_path, columns=["ticker", "trade_date", "market_cap"])
    return df.pivot(index="trade_date", columns="ticker", values="market_cap").ffill()


def market_cap_policy(env: PortfolioEnv, config: AgentConfig) -> Callable:
    """w ∝ market cap; ffill per ticker so caps are 'last known at t' (no lookahead)."""
    caps = (
        _market_cap_pivot(config.dataset_path)
        .reindex(index=env.dates, columns=env.tickers)
        .to_numpy()
    )

    def act(obs: np.ndarray, t: int) -> np.ndarray:
        w = np.where(env.mask[t], np.nan_to_num(caps[t], nan=0.0), 0.0)
        w = np.maximum(w, 0.0)
        w = _exclude_cash_from_weights(w, env)
        if w.sum() == 0:  # no caps known yet → fall back to equal weight
            w = env.mask[t].astype(float)
            w = _exclude_cash_from_weights(w, env)
        return _weights_to_logits(w / w.sum(), env.config.logit_scale)
    return act


def inv_vol_policy(env: PortfolioEnv) -> Callable:
    """w ∝ 1/std(trailing 60d returns). Trailing only — no lookahead."""
    returns = np.nan_to_num(env.returns, nan=0.0)

    def act(obs: np.ndarray, t: int) -> np.ndarray:
        lo = max(0, t - VOL_WINDOW)
        vol = returns[lo: t + 1].std(axis=0)
        w = np.where(env.mask[t] & (vol > 1e-8), 1.0 / (vol + 1e-8), 0.0)
        w = _exclude_cash_from_weights(w, env)
        if w.sum() == 0:  # first day / degenerate → equal weight
            w = env.mask[t].astype(float)
            w = _exclude_cash_from_weights(w, env)
        return _weights_to_logits(w / w.sum(), env.config.logit_scale)
    return act


def selic_policy(env: PortfolioEnv, config: AgentConfig) -> Callable:
    """100% CASH held at daily SELIC rates (synthetic asset already in dataset)."""
    def act(obs: np.ndarray, t: int) -> np.ndarray:
        # 100% CASH: all weight on CASH ticker
        w = np.zeros(len(env.tickers))
        if "CASH" in env.tickers:
            cash_idx = np.where(env.tickers == "CASH")[0][0]
            w[cash_idx] = 1.0
        return _weights_to_logits(w, env.config.logit_scale)
    return act


def random_policy(env: PortfolioEnv, rng: np.random.Generator) -> Callable:
    """i.i.d. random logits each day — masking/softmax/costs still handled by
    env.step, same as every other policy here. A null baseline: the agent
    should clear this by a wide margin, not just beat the deterministic ones.
    """
    def act(obs: np.ndarray, t: int) -> np.ndarray:
        return rng.normal(size=len(env.tickers)).astype(np.float32)
    return act


def random_baseline_stats(env: PortfolioEnv, n_samples: int = 20, seed: int = 42) -> dict:
    """Roll n_samples random-logit policies through the SAME env to get a null
    distribution of Sharpes, for judging whether the agent's edge is noise."""
    sharpes = [sharpe_ratio(rollout(env, random_policy(env, np.random.default_rng(seed + i)))["rewards"])
               for i in range(n_samples)]
    return {
        "mean": float(np.mean(sharpes)),
        "std": float(np.std(sharpes)),
        "min": float(np.min(sharpes)),
        "max": float(np.max(sharpes)),
        "n_samples": n_samples,
        "sharpes": sharpes,
    }


def bova11_result(dates: pd.DatetimeIndex, config: AgentConfig) -> dict | None:
    """
    Buy-and-hold BOVA11 from raw ETF prices (BOVA11 is not in the stocks-only
    env universe, so this bypasses the env instead of faking it with logits).
    Returns a rollout-shaped dict aligned to the agent's backtest dates.
    """
    bova_path = config.model_dir.parent.parent / "data" / "raw" / "prices" / "BOVA11.parquet"
    if not bova_path.exists():
        logger.warning("BOVA11.parquet not found at %s — skipping bova11 baseline", bova_path)
        return None

    px = (
        pd.read_parquet(bova_path, columns=["trade_date", "adj_close"])
        .set_index("trade_date")["adj_close"]
        .sort_index()
    )
    px = px.reindex(px.index.union(dates)).ffill().reindex(dates)
    rewards = np.log(px).diff().fillna(0.0).to_numpy()
    values = config.initial_capital * np.exp(np.concatenate([[0.0], np.cumsum(rewards)]))
    return {
        "rewards": rewards,
        "values": values,
        "weights": np.ones((len(dates), 1)),  # 100% in the ETF, zero turnover
        "dates": dates,
        "costs": np.zeros_like(rewards),  # Buy-and-hold: no costs
    }



# --------------------------------------------------------------- main

def backtest(model_path: Path, config: AgentConfig = DEFAULT_CONFIG) -> dict:
    env = PortfolioEnv(config, date_range="test")
    model = PPO.load(model_path, device=config.device)
    logger.info("Backtesting %s on test split (%d days)", model_path, len(env.dates))

    policies = {
        "agent": agent_policy(model),
        "equal_weight": equal_weight_policy(env),
        "market_cap": market_cap_policy(env, config),
        "inv_vol": inv_vol_policy(env),
        "selic": selic_policy(env, config),
    }
    results = {name: rollout(env, fn) for name, fn in policies.items()}
    bova = bova11_result(results["agent"]["dates"], config)
    if bova is not None:
        results["bova11"] = bova

    # Metrics table (with cost breakdown for equity baselines)
    metrics = {}
    for name, res in results.items():
        if "costs" in res:
            # Equity strategies: pass costs for gross/net decomposition
            metrics[name] = compute_all(res["rewards"], res["values"], res["weights"],
                                       daily_costs=res["costs"], cost_bps=config.transaction_cost_bps)
        else:
            # bova11: buy-and-hold, no costs
            metrics[name] = compute_all(res["rewards"], res["values"], res["weights"])

    # Luck-vs-skill checks on the agent's Sharpe: PSR always; DSR only if a real
    # multi-window trial record exists (no fabricated trial count); random-policy
    # null distribution through the same env/costs/mask.
    metrics["agent"]["probabilistic_sharpe_ratio"] = probabilistic_sharpe_ratio(results["agent"]["rewards"])

    rolling_results_path = config.model_dir / "rolling_eval_results.json"
    if rolling_results_path.exists():
        with open(rolling_results_path) as f:
            rolling_results = json.load(f)
        trial_sharpes = np.array([w["metrics"]["agent"]["sharpe"] for w in rolling_results["windows"]]) / np.sqrt(TRADING_DAYS)
        if len(trial_sharpes) >= 2:
            metrics["agent"]["deflated_sharpe_ratio"] = deflated_sharpe_ratio(results["agent"]["rewards"], trial_sharpes)
    else:
        logger.info("No rolling_eval_results.json found — skipping deflated Sharpe (needs multi-window trial record)")

    random_stats = random_baseline_stats(env)
    metrics["agent"]["random_baseline_sharpe_mean"] = random_stats["mean"]
    metrics["agent"]["random_baseline_sharpe_std"] = random_stats["std"]
    metrics["agent"]["agent_percentile_vs_random"] = float(
        (np.array(random_stats["sharpes"]) < metrics["agent"]["sharpe"]).mean() * 100
    )

    if config.rebalance_interval_days > 1:
        logger.warning(
            "⚠ Models trained before rebalance_interval_days=%d are STALE "
            "(obs unchanged but step semantics differ). Retrain to use this config.",
            config.rebalance_interval_days
        )

    table = pd.DataFrame(metrics).T
    logger.info("=" * 78)
    logger.info("BACKTEST RESULTS (test set: %s → %s)", config.test_start, config.test_end)
    logger.info("=" * 78)
    logger.info("\n%s", table.round(4).to_string())

    # Persist: metrics.json + results.parquet (viz lives in src/visualizations/*.ipynb)
    config.backtest_dir.mkdir(parents=True, exist_ok=True)
    with open(config.backtest_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    agent_res = results["agent"]
    keep = [i for i in range(len(env.tickers)) if agent_res["weights"][:, i].max() > 0.001]
    out = pd.concat(
        [
            pd.DataFrame({
                "date": agent_res["dates"],
                "log_return": agent_res["rewards"],
                # one value column per strategy so notebooks can compare without re-rolling
                **{f"value_{name}": res["values"][1:] for name, res in results.items()},
            }),
            pd.DataFrame(
                agent_res["weights"][:, keep],
                columns=[f"w_{env.tickers[i]}" for i in keep],
            ),
        ],
        axis=1,
    )
    out.to_parquet(config.backtest_dir / "results.parquet", index=False)
    logger.info("Saved metrics.json + results.parquet → %s", config.backtest_dir)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest agent vs baselines on test set")
    parser.add_argument("--model", type=Path, default=DEFAULT_CONFIG.model_dir / "agent_best.zip")
    parser.add_argument("--online", action="store_true", help="Run online retraining backtest instead of frozen model")
    parser.add_argument("--retrain-every-days", type=int, default=63, help="Retrain every N days (default: 63)")
    parser.add_argument("--retrain-timesteps", type=int, default=20_000, help="Timesteps per retrain (default: 20,000)")
    parser.add_argument("--resume", action="store_true", help="Resume the online backtest from its last checkpointed chunk")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = configure_logging(DEFAULT_CONFIG.log_dir / "agent" / "evaluate", run_id, tag="evaluate")
    logger.info("Session log → %s", log_path)

    if args.online:
        # Deferred import to avoid circular dependency (rolling_eval imports from evaluate)
        from src.agent.rolling_eval import run_online_backtest
        df, metrics = run_online_backtest(
            DEFAULT_CONFIG,
            retrain_every_days=args.retrain_every_days,
            retrain_timesteps=args.retrain_timesteps,
            resume=args.resume,
        )
        # Log summary
        logger.info("=" * 78)
        logger.info("ONLINE RETRAINING BACKTEST RESULTS")
        logger.info("=" * 78)
        table = pd.DataFrame(metrics).T
        logger.info("\n%s", table.round(4).to_string())
    else:
        backtest(args.model)


if __name__ == "__main__":
    main()
