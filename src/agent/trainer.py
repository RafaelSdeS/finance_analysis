"""
PPO trainer for PortfolioEnv.

Trains on the train split, periodically evaluates on the val split
(deterministic rollout → Sharpe), checkpoints, logs JSONL, and stops
early when validation Sharpe degrades for `early_stopping_patience`
consecutive evaluations.

Usage:
    python -m src.agent.trainer                       # full run (config defaults)
    python -m src.agent.trainer --timesteps 20000     # smoke run
    python -m src.agent.trainer --learning-rate 1e-4 --timesteps 500000
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.agent.config import AgentConfig, DEFAULT_CONFIG
from src.agent.env import PortfolioEnv
from src.agent.metrics import max_drawdown, sharpe_ratio

logger = logging.getLogger(__name__)


def evaluate_on_env(model: PPO, env: PortfolioEnv) -> dict:
    """Deterministic rollout over an entire env split; returns metrics."""
    obs, _ = env.reset()
    rewards, values = [], [env.config.initial_capital]
    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, info = env.step(action)
        rewards.append(reward)
        values.append(info["portfolio_value"])
    return {
        "sharpe": sharpe_ratio(np.array(rewards)),
        "max_drawdown": max_drawdown(np.array(values)),
        "final_value": values[-1],
    }


class ValSharpeCallback(BaseCallback):
    """Periodic val evaluation + JSONL logging + checkpoints + early stopping."""

    def __init__(self, config: AgentConfig, val_env: PortfolioEnv, log_path: Path):
        super().__init__()
        self.config = config
        self.val_env = val_env
        self.log_path = log_path
        self.eval_every_steps = config.eval_freq * config.n_steps
        self.best_sharpe = -np.inf
        self.degrade_count = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every_steps != 0:
            return True

        val = evaluate_on_env(self.model, self.val_env)
        record = {
            "timesteps": self.num_timesteps,
            "val_sharpe": val["sharpe"],
            "val_max_drawdown": val["max_drawdown"],
            "val_final_value": val["final_value"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(
            "eval @ %s steps: val_sharpe=%.3f, val_max_dd=%.1f%%, val_value=%s",
            f"{self.num_timesteps:,}", val["sharpe"], val["max_drawdown"] * 100,
            f"{val['final_value']:,.0f}",
        )

        # Checkpoint
        ckpt = self.config.model_dir / f"agent_checkpoint_{self.num_timesteps}.zip"
        self.model.save(ckpt)

        # Early stopping on degrading val Sharpe
        if val["sharpe"] > self.best_sharpe:
            self.best_sharpe = val["sharpe"]
            self.degrade_count = 0
            self.model.save(self.config.model_dir / "agent_best.zip")
        else:
            self.degrade_count += 1
            if self.degrade_count >= self.config.early_stopping_patience:
                logger.warning(
                    "Early stopping: val Sharpe degraded %d consecutive evals "
                    "(best=%.3f)", self.degrade_count, self.best_sharpe,
                )
                return False
        return True


def train(config: AgentConfig) -> Path:
    """Run PPO training; returns path to the final model."""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = config.log_dir / f"{config.log_file_prefix}_{run_id}.jsonl"

    train_env = PortfolioEnv(config, date_range="train")
    val_env = PortfolioEnv(config, date_range="val")

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.entropy_coef,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        seed=config.seed,
        device=config.device,
        verbose=config.verbose,
    )
    logger.info("Training PPO for %s timesteps (device=%s), log → %s",
                f"{config.total_timesteps:,}", model.device, log_path)

    t0 = time.time()
    model.learn(total_timesteps=config.total_timesteps, callback=ValSharpeCallback(config, val_env, log_path))
    logger.info("Training finished in %.1f min", (time.time() - t0) / 60)

    final_path = config.model_dir / "agent_final.zip"
    model.save(final_path)
    logger.info("Saved final model → %s (best-val model → agent_best.zip)", final_path)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO portfolio agent")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    args = parser.parse_args()

    overrides = {}
    if args.timesteps is not None:
        overrides["total_timesteps"] = args.timesteps
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.device is not None:
        overrides["device"] = args.device

    config = AgentConfig(**overrides) if overrides else DEFAULT_CONFIG
    config.log_summary()
    train(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
