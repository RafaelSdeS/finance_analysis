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
    python -m src.agent.evaluate --model data/models/agent_final.zip
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.agent.config import AgentConfig, DEFAULT_CONFIG
from src.agent.env import PortfolioEnv
from src.agent.metrics import compute_all

logger = logging.getLogger(__name__)

VOL_WINDOW = 60  # trailing days for inverse-volatility baseline


# --------------------------------------------------------------- rollouts

def _weights_to_logits(weights: np.ndarray) -> np.ndarray:
    """Logits whose masked softmax reproduces the given weights."""
    return np.log(np.maximum(weights, 1e-12)).astype(np.float32)


def rollout(env: PortfolioEnv, act_fn: Callable[[np.ndarray, int], np.ndarray]) -> dict:
    """Roll a policy through an env split. act_fn(obs, t) → action logits."""
    obs, _ = env.reset()
    rewards, values, weights_log, dates = [], [env.config.initial_capital], [], []
    t, terminated = 0, False
    while not terminated:
        action = act_fn(obs, t)
        obs, reward, terminated, _, info = env.step(action)
        rewards.append(reward)
        values.append(info["portfolio_value"])
        weights_log.append(info["weights"])
        dates.append(info["date"])
        t += 1
    return {
        "rewards": np.array(rewards),
        "values": np.array(values),
        "weights": np.array(weights_log),   # [T, n_tickers]
        "dates": pd.DatetimeIndex(dates),
    }


def agent_policy(model: PPO) -> Callable:
    def act(obs: np.ndarray, t: int) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action
    return act


def equal_weight_policy(env: PortfolioEnv) -> Callable:
    def act(obs: np.ndarray, t: int) -> np.ndarray:
        active = env.mask[t]
        return _weights_to_logits(active / active.sum())
    return act


def market_cap_policy(env: PortfolioEnv, config: AgentConfig) -> Callable:
    """w ∝ market cap; ffill per ticker so caps are 'last known at t' (no lookahead)."""
    df = pd.read_parquet(config.dataset_path, columns=["ticker", "trade_date", "market_cap"])
    caps = (
        df.pivot(index="trade_date", columns="ticker", values="market_cap")
        .ffill()
        .reindex(index=env.dates, columns=env.tickers)
        .to_numpy()
    )

    def act(obs: np.ndarray, t: int) -> np.ndarray:
        w = np.where(env.mask[t], np.nan_to_num(caps[t], nan=0.0), 0.0)
        w = np.maximum(w, 0.0)
        if w.sum() == 0:  # no caps known yet → fall back to equal weight
            w = env.mask[t].astype(float)
        return _weights_to_logits(w / w.sum())
    return act


def inv_vol_policy(env: PortfolioEnv) -> Callable:
    """w ∝ 1/std(trailing 60d returns). Trailing only — no lookahead."""
    returns = np.nan_to_num(env.returns, nan=0.0)

    def act(obs: np.ndarray, t: int) -> np.ndarray:
        lo = max(0, t - VOL_WINDOW)
        vol = returns[lo: t + 1].std(axis=0)
        w = np.where(env.mask[t] & (vol > 1e-8), 1.0 / (vol + 1e-8), 0.0)
        if w.sum() == 0:  # first day / degenerate → equal weight
            w = env.mask[t].astype(float)
        return _weights_to_logits(w / w.sum())
    return act


# --------------------------------------------------------------- plotting

def make_plots(results: dict[str, dict], env: PortfolioEnv, plots_dir: Path) -> None:
    import plotly.graph_objects as go

    plots_dir.mkdir(parents=True, exist_ok=True)
    agent = results["agent"]

    # 1. Cumulative portfolio value: agent vs baselines
    fig = go.Figure()
    for name, res in results.items():
        fig.add_trace(go.Scatter(x=res["dates"], y=res["values"][1:], name=name, mode="lines"))
    fig.update_layout(title="Portfolio Value (test set)", yaxis_type="log",
                      yaxis_title="Value (R$, log scale)")
    fig.write_html(plots_dir / "cumulative_value.html")

    # 2. Agent drawdown over time
    v = agent["values"]
    dd = (np.maximum.accumulate(v) - v) / np.maximum.accumulate(v)
    fig = go.Figure(go.Scatter(x=agent["dates"], y=-dd[1:] * 100, fill="tozeroy", name="drawdown"))
    fig.update_layout(title="Agent Drawdown (test set)", yaxis_title="Drawdown (%)")
    fig.write_html(plots_dir / "drawdown.html")

    # 3. Daily return distribution
    fig = go.Figure(go.Histogram(x=agent["rewards"] * 100, nbinsx=100))
    fig.update_layout(title="Agent Daily Log Returns (test set)", xaxis_title="Return (%)")
    fig.write_html(plots_dir / "return_distribution.html")

    # 4. Weights timeline: top-10 average holdings, stacked
    mean_w = agent["weights"].mean(axis=0)
    top = np.argsort(mean_w)[::-1][:10]
    fig = go.Figure()
    for i in top:
        fig.add_trace(go.Scatter(x=agent["dates"], y=agent["weights"][:, i] * 100,
                                 name=str(env.tickers[i]), stackgroup="w"))
    fig.update_layout(title="Agent Weights — Top 10 Holdings", yaxis_title="Weight (%)")
    fig.write_html(plots_dir / "weights_timeline.html")

    logger.info("Plots saved → %s", plots_dir)


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
    }
    results = {name: rollout(env, fn) for name, fn in policies.items()}

    # Metrics table
    metrics = {name: compute_all(res["rewards"], res["values"]) for name, res in results.items()}
    table = pd.DataFrame(metrics).T
    print("\n" + "=" * 78)
    print("BACKTEST RESULTS (test set: "
          f"{config.test_start}{config.test_end})")
    print("=" * 78)
    print(table.round(4).to_string())

    # Persist: metrics.json + results.parquet + plots
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

    make_plots(results, env, config.backtest_dir / "plots")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest agent vs baselines on test set")
    parser.add_argument("--model", type=Path, default=Path("data/models/agent_best.zip"))
    args = parser.parse_args()
    backtest(args.model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
