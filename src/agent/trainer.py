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
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

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

    def __init__(self, config: AgentConfig, val_env: PortfolioEnv, log_path: Path, total_timesteps: int):
        super().__init__()
        self.config = config
        self.val_env = val_env
        self.log_path = log_path
        self.total_timesteps_target = total_timesteps
        self.eval_every_steps = config.eval_freq * config.n_steps
        self._next_eval = self.eval_every_steps  # threshold, not modulo: robust to vec-env timestep jumps
        self.best_sharpe = -np.inf
        self.degrade_count = 0
        self.pbar = tqdm(total=total_timesteps, unit="step", unit_scale=True, desc="Training")

    def _on_step(self) -> bool:
        self.pbar.update(self.num_timesteps - self.pbar.n)

        if self.num_timesteps < self._next_eval:
            return True
        self._next_eval += self.eval_every_steps

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

        # One-line summary
        self.pbar.set_postfix({
            "sharpe": f"{val['sharpe']:.3f}",
            "dd%": f"{val['max_drawdown']*100:.1f}",
            "value": f"{val['final_value']:,.0f}"
        })

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
                self.pbar.close()
                logger.warning(
                    "Early stopping: val Sharpe degraded %d consecutive evals "
                    "(best=%.3f)", self.degrade_count, self.best_sharpe,
                )
                return False
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


def _find_latest_checkpoint(config: AgentConfig) -> Path | None:
    """Find the latest checkpoint by timestep number."""
    checkpoints = list(config.model_dir.glob("agent_checkpoint_*.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))


def train(config: AgentConfig, resume: bool = False) -> Path:
    """Run PPO training; returns path to the final model."""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = config.log_dir / f"{config.log_file_prefix}_{run_id}.jsonl"

    # Parallel rollout collection: N envs stepped at once → ~N× throughput.
    # SubprocVecEnv (separate processes) sidesteps the GIL; 1 env stays single-process.
    def _make_train_env():
        return PortfolioEnv(config, date_range="train")

    if config.n_envs > 1:
        train_env = SubprocVecEnv([_make_train_env for _ in range(config.n_envs)])
    else:
        train_env = DummyVecEnv([_make_train_env])
    val_env = PortfolioEnv(config, date_range="val")  # single deterministic pass, no need to parallelize

    ckpt_path = _find_latest_checkpoint(config) if resume else None
    if ckpt_path:
        model = PPO.load(ckpt_path, env=train_env)
        resumed_steps = int(ckpt_path.stem.split("_")[-1])
        logger.info("Resumed from checkpoint %s (timestep=%s)", ckpt_path.name, f"{resumed_steps:,}")
    else:
        if resume:
            logger.warning("No checkpoint found; starting fresh")
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
            policy_kwargs=dict(net_arch=[256, 256]),
            device=config.device,
            verbose=config.verbose,
        )
    logger.info("Training PPO (device=%s), log → %s", model.device, log_path)

    t0 = time.time()
    callback = ValSharpeCallback(config, val_env, log_path, config.total_timesteps)
    model.learn(total_timesteps=config.total_timesteps, callback=callback)
    logger.info("Training finished in %.1f min", (time.time() - t0) / 60)

    final_path = config.model_dir / "agent_final.zip"
    model.save(final_path)
    logger.info("Saved final model → %s (best-val model → agent_best.zip)", final_path)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO portfolio agent")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps (default: 1,000,000)")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate (default: 3e-4)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (default: 64)")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Device (default: cuda)")
    parser.add_argument("--n-envs", type=int, default=None, help="Parallel rollout workers (default: 8; use 1 to disable)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    overrides = {}
    if args.timesteps is not None:
        overrides["total_timesteps"] = args.timesteps
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.device is not None:
        overrides["device"] = args.device
    if args.n_envs is not None:
        overrides["n_envs"] = args.n_envs

    config = AgentConfig(**overrides) if overrides else DEFAULT_CONFIG
    config.log_summary()
    train(config, resume=args.resume)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
