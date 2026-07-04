"""
PPO trainer for PortfolioEnv — anchored rolling windows, always.

`python -m src.agent.trainer` trains one PPO model per anchored rolling
window (see `config.generate_windows()`): each window's train span is
tail-carved into train/val for early stopping (this module's job), its
test span is left untouched, and window orchestration/reporting lives in
`rolling_eval.py` (`run_rolling_eval`, `finalize_and_report`).

Per window: trains on the train split, periodically evaluates on the val
split (deterministic rollout → Sharpe), checkpoints, logs JSONL, and stops
early when validation Sharpe degrades for `early_stopping_patience`
consecutive evaluations. The most recent window's model is saved as
`agent_best.zip`/`agent_final.zip` (the production model); earlier windows
are namespaced `window_{id}_best.zip`/`window_{id}_final.zip`.

Usage:
    python -m src.agent.trainer                                 # full run: all windows, 1M timesteps each
    python -m src.agent.trainer --timesteps 20000               # smoke run (per window)
    python -m src.agent.trainer --train-years 2 --test-years 1  # fast smoke: many small windows
    python -m src.agent.trainer --learning-rate 1e-4 --device cpu
"""

import argparse
import dataclasses
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

    def __init__(
        self, config: AgentConfig, val_env: PortfolioEnv, log_path: Path, total_timesteps: int,
        model_tag: str = "agent",
    ):
        super().__init__()
        self.config = config
        self.val_env = val_env
        self.log_path = log_path
        self.total_timesteps_target = total_timesteps
        self.model_tag = model_tag
        self.eval_every_steps = config.eval_freq * config.n_steps
        self._next_eval = self.eval_every_steps  # threshold, not modulo: robust to vec-env timestep jumps
        self.best_sharpe = -np.inf
        self.degrade_count = 0
        self.pbar = tqdm(total=total_timesteps, unit="step", unit_scale=True, desc=f"Training [{model_tag}]")

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
        ckpt = self.config.model_dir / f"{self.model_tag}_checkpoint_{self.num_timesteps}.zip"
        self.model.save(ckpt)

        # Early stopping on degrading val Sharpe
        if val["sharpe"] > self.best_sharpe:
            self.best_sharpe = val["sharpe"]
            self.degrade_count = 0
            self.model.save(self.config.model_dir / f"{self.model_tag}_best.zip")
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


def _find_latest_checkpoint(config: AgentConfig, model_tag: str = "agent") -> Path | None:
    """Find the latest checkpoint by timestep number."""
    checkpoints = list(config.model_dir.glob(f"{model_tag}_checkpoint_*.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))


def train(config: AgentConfig, resume: bool = False, model_tag: str = "agent") -> Path:
    """Run PPO training for one window; returns path to its final model."""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = config.log_dir / f"{config.log_file_prefix}_{model_tag}_{run_id}.jsonl"

    # Parallel rollout collection: N envs stepped at once → ~N× throughput.
    # SubprocVecEnv (separate processes) sidesteps the GIL; 1 env stays single-process.
    def _make_train_env():
        return PortfolioEnv(config, date_range="train")

    if config.n_envs > 1:
        train_env = SubprocVecEnv([_make_train_env for _ in range(config.n_envs)])
    else:
        train_env = DummyVecEnv([_make_train_env])
    val_env = PortfolioEnv(config, date_range="val")  # single deterministic pass, no need to parallelize

    try:
        ckpt_path = _find_latest_checkpoint(config, model_tag) if resume else None
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
        logger.info("Training PPO [%s] (device=%s), log → %s", model_tag, model.device, log_path)

        t0 = time.time()
        callback = ValSharpeCallback(config, val_env, log_path, config.total_timesteps, model_tag=model_tag)
        model.learn(total_timesteps=config.total_timesteps, callback=callback)
        logger.info("Training finished in %.1f min", (time.time() - t0) / 60)

        final_path = config.model_dir / f"{model_tag}_final.zip"
        model.save(final_path)
        logger.info("Saved final model → %s (best-val model → %s_best.zip)", final_path, model_tag)
        return final_path
    finally:
        # Tear down subprocess pools explicitly: main() calls train() once per
        # rolling window in the same process, so leaked SubprocVecEnv workers
        # from an earlier window would otherwise pile up across windows.
        train_env.close()
        val_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO portfolio agent via anchored rolling windows")
    parser.add_argument("--timesteps", type=int, default=None, help="PPO timesteps PER WINDOW (default: 1,000,000)")
    parser.add_argument("--train-years", type=int, default=None, help="Years of history per window's train span (default: 10)")
    parser.add_argument("--test-years", type=int, default=None, help="Years per window's held-out test span (default: 2)")
    parser.add_argument("--val-fraction", type=float, default=None, help="Fraction of each window's train span carved out for early-stopping val (default: 0.15)")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate (default: 3e-4)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (default: 64)")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Device (default: cuda)")
    parser.add_argument("--n-envs", type=int, default=None, help="Parallel rollout workers (default: 8; use 1 to disable)")
    parser.add_argument("--resume", action="store_true", help="Resume the currently in-progress window from its latest checkpoint")
    args = parser.parse_args()

    overrides = {}
    if args.timesteps is not None:
        overrides["total_timesteps"] = args.timesteps
    if args.train_years is not None:
        overrides["window_train_years"] = args.train_years
    if args.test_years is not None:
        overrides["window_test_years"] = args.test_years
    if args.val_fraction is not None:
        overrides["window_val_fraction"] = args.val_fraction
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.device is not None:
        overrides["device"] = args.device
    if args.n_envs is not None:
        overrides["n_envs"] = args.n_envs

    # dataclasses.replace() (not a fresh AgentConfig(**overrides)) so any CLI
    # override still inherits DEFAULT_CONFIG's window-derived split dates
    # rather than silently reverting to the raw template dates.
    base_config = dataclasses.replace(DEFAULT_CONFIG, **overrides) if overrides else DEFAULT_CONFIG
    base_config.log_summary()

    # Deferred import: rolling_eval.train_window() imports `train` from this
    # module at its top level, so this side of the dependency stays lazy to
    # avoid a circular import.
    from src.agent.rolling_eval import run_rolling_eval, finalize_and_report
    results = run_rolling_eval(base_config, resume=args.resume)
    finalize_and_report(results, base_config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
