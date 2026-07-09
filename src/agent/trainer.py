"""
PPO trainer for PortfolioEnv — anchored rolling windows, always.

`python -m src.agent.trainer` trains one PPO model per anchored rolling
window (see `config.generate_windows()`): each window's train span is
tail-carved into train/val for early stopping (this module's job), its
test span is left untouched, and window orchestration/reporting lives in
`rolling_eval.py` (`run_rolling_eval`, `finalize_and_report`).

Per window: trains on the train split, periodically evaluates on the val
split (deterministic rollout → Sharpe of excess return over equal-weight),
checkpoints, logs JSONL, and stops early when that excess-over-equal-weight
Sharpe degrades for `early_stopping_patience` consecutive evaluations. Each
invocation trains all windows under its own `artifacts/models/runs/<session_id>/`
scratch directory (see `rolling_eval.run_rolling_eval()`); the most recent
window's model (`agent_best.zip`/`agent_final.zip`) is then promoted to the
stable top-level `artifacts/models/` (the production model); earlier windows stay
namespaced `window_{id}_best.zip`/`window_{id}_final.zip` inside the run dir.

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

from src.agent.config import AgentConfig, DEFAULT_CONFIG, configure_logging
from src.agent.env import PortfolioEnv
from src.agent.metrics import max_drawdown, sharpe_ratio
from src.agent.evaluate import rollout, agent_policy, equal_weight_policy
from src.agent.model_provenance import write_sidecar

logger = logging.getLogger(__name__)


def _to_native_python(obj):
    """Recursively convert numpy/torch scalars to native Python types for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_to_native_python(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_native_python(v) for k, v in obj.items()}
    return obj


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
        self.best_excess_sharpe = -np.inf   # gates the patience-reset below (needs a MEANINGFUL jump)
        self.best_saved_sharpe = -np.inf    # gates checkpoint saving (ANY improvement saves)
        self.degrade_count = 0
        # Noise floor for resetting the patience counter -- lowered from 0.05 (2026-07-09):
        # window_3 of the 132721 run improved monotonically every single eval (0.283 -> 0.315
        # over 8 evals, ~0.004-0.017/eval) but never once cleared a 0.05 jump in a single step,
        # so degrade_count incremented every eval as if it were regressing, and patience
        # exhausted mid-climb. 0.02 is calibrated to that observed real-improvement magnitude.
        self.improvement_threshold = 0.02
        self.pbar = tqdm(total=total_timesteps, unit="step", unit_scale=True, desc=f"Training [{model_tag}]")
        # Cache the equal-weight rollout's absolute-return series once (env is deterministic, reused ~eval_freq times)
        self.ew_returns = rollout(self.val_env, equal_weight_policy(self.val_env))["rewards"]

    def _on_rollout_start(self) -> None:
        # Defense-in-depth against log_std collapse/explosion: SB3's log_std is an unconstrained
        # nn.Parameter (no built-in bounds), so even with target_kl capping *how far* a rollout's
        # epochs can push it, a single accepted update could still drift outside a numerically safe
        # range. Clamping here (once per rollout, right after train() returns control) keeps
        # exp(log_std) bounded to roughly [0.0067, 7.4] without fighting the conviction objective
        # (log_std_init=-3.0 sits well inside this range).
        if self.model is not None and hasattr(self.model.policy, "log_std"):
            self.model.policy.log_std.data.clamp_(-5.0, 2.0)

    def _on_step(self) -> bool:
        self.pbar.update(self.num_timesteps - self.pbar.n)

        # Scan every 100 steps (each .isnan().any() forces a GPU sync, so scanning every
        # step dominated rollout time). 100-step detection latency is fine — a crash
        # *inside* model.train() (e.g. Normal() rejecting NaN loc) propagates before this
        # callback runs again anyway — see train()'s try/except for that case.
        if self.n_calls % 100 == 0:
            for name, param in self.model.policy.named_parameters():
                if param.data.isnan().any():
                    log_std = self.model.policy.log_std.data
                    logger.error(
                        "NaN detected in policy parameter '%s' at timestep %d. log_std stats: "
                        "mean=%.4f min=%.4f max=%.4f (init=%.2f). learning_rate=%.2e.",
                        name, self.num_timesteps,
                        float(log_std.mean()), float(log_std.min()), float(log_std.max()),
                        self.config.log_std_init, float(self.model.learning_rate),
                    )
                    return False

        if self.num_timesteps < self._next_eval:
            return True
        self._next_eval += self.eval_every_steps

        agent_res = rollout(self.val_env, agent_policy(self.model))
        excess_returns = agent_res["rewards"] - self.ew_returns
        val = {
            "sharpe": sharpe_ratio(agent_res["rewards"]),
            "excess_sharpe": sharpe_ratio(excess_returns),
            "max_drawdown": max_drawdown(agent_res["values"]),
            "final_value": agent_res["values"][-1],
        }
        log_std = self.model.policy.log_std.data
        # SB3's name_to_value is populated by each train() call; values are None before first rollout/train completes
        record = {
            "timesteps": self.num_timesteps,
            "val_sharpe": val["sharpe"],
            "val_sharpe_excess": val["excess_sharpe"],  # Sharpe of the agent's day-by-day excess return over equal-weight
            "val_max_drawdown": val["max_drawdown"],
            "val_final_value": val["final_value"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "train_entropy_loss": self.model.logger.name_to_value.get("train/entropy_loss"),
            "train_approx_kl": self.model.logger.name_to_value.get("train/approx_kl"),
            "train_clip_fraction": self.model.logger.name_to_value.get("train/clip_fraction"),
            "train_value_loss": self.model.logger.name_to_value.get("train/value_loss"),
            "train_policy_gradient_loss": self.model.logger.name_to_value.get("train/policy_gradient_loss"),
            "train_explained_variance": self.model.logger.name_to_value.get("train/explained_variance"),
            "learning_rate": self.model.logger.name_to_value.get("train/learning_rate"),
            "fps": self.model.logger.name_to_value.get("time/fps"),
            "policy_log_std_mean": float(log_std.mean()),
            "policy_log_std_max": float(log_std.max()),
        }
        # Convert numpy/torch scalars to native Python types for JSON serialization
        record = _to_native_python(record)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # One-line summary; include KL for instability detection
        approx_kl = self.model.logger.name_to_value.get("train/approx_kl")
        self.pbar.set_postfix({
            "sharpe": f"{val['sharpe']:.3f}",
            "excess_sharpe": f"{val['excess_sharpe']:.3f}",
            "max_dd%": f"{val['max_drawdown']*100:.1f}",
            "kl": f"{approx_kl:.3f}" if approx_kl is not None else "n/a",
            "value": f"{val['final_value']:,.0f}"
        })

        # Checkpoint: keep only the latest (resume only ever reads the newest)
        prev_ckpts = list(self.config.model_dir.glob(f"{self.model_tag}_checkpoint_*.zip"))
        ckpt = self.config.model_dir / f"{self.model_tag}_checkpoint_{self.num_timesteps}.zip"
        self.model.save(ckpt)
        for old in prev_ckpts:
            if old != ckpt:
                old.unlink(missing_ok=True)

        # Checkpoint on ANY new best (decoupled from the patience-reset threshold below): a real
        # but small improvement that never clears improvement_threshold in a single eval would
        # otherwise never get saved at all, silently discarding it even if training runs longer.
        if val["excess_sharpe"] > self.best_saved_sharpe:
            self.best_saved_sharpe = val["excess_sharpe"]
            best_path = self.config.model_dir / f"{self.model_tag}_best.zip"
            self.model.save(best_path)
            write_sidecar(best_path, self.config, timesteps=self.num_timesteps)

        # Early stopping: degrade counter resets only on meaningful improvement (>= threshold)
        # to avoid noise-driven early stops from small sample validation splits. Keyed on
        # excess-over-equal-weight Sharpe, not absolute Sharpe: keep training as long as the
        # agent is still pulling ahead of the equal-weight baseline.
        if val["excess_sharpe"] > self.best_excess_sharpe + self.improvement_threshold:
            self.best_excess_sharpe = val["excess_sharpe"]
            self.degrade_count = 0
        else:
            self.degrade_count += 1
            if self.degrade_count >= self.config.early_stopping_patience:
                self.pbar.close()
                logger.warning(
                    "Early stopping: val excess Sharpe (vs equal-weight) degraded %d consecutive "
                    "evals (best=%.3f)", self.degrade_count, self.best_excess_sharpe,
                )
                return False
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


class CostAnnealCallback(BaseCallback):
    """Ramp env cost_scale 0→1 over training to unstrangle early exploration."""

    def __init__(self, total_timesteps: int, anneal_frac: float = 0.5):
        """Ramp cost_scale from 0 to 1 over the first anneal_frac of training.

        Args:
            total_timesteps: Total PPO timesteps for the full run
            anneal_frac: Fraction of training over which to ramp (default: 50%)
        """
        super().__init__()
        self.total_timesteps = total_timesteps
        self.anneal_frac = anneal_frac
        self.anneal_until = int(anneal_frac * total_timesteps)

    def _on_rollout_start(self) -> None:
        if self.num_timesteps < self.anneal_until and self.training_env is not None:
            frac = self.num_timesteps / self.anneal_until
            # Set cost_scale uniformly across all parallel envs
            self.training_env.set_attr("cost_scale", min(1.0, frac))

    def _on_step(self) -> bool:
        return True  # Never block


def _find_latest_checkpoint(config: AgentConfig, model_tag: str = "agent") -> Path | None:
    """Find the latest checkpoint by timestep number."""
    checkpoints = list(config.model_dir.glob(f"{model_tag}_checkpoint_*.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))


def _resolve_session_id(config: AgentConfig, resume: bool) -> str:
    """Fresh timestamped session_id per invocation — except --resume, which reuses
    the most-recently-modified existing runs/ dir (a brand-new dir would have
    nothing to resume from)."""
    if resume:
        runs_root = config.model_dir / "runs"
        existing = sorted(runs_root.glob("*/"), key=lambda p: p.stat().st_mtime) if runs_root.exists() else []
        if existing:
            return existing[-1].name
        logger.warning("--resume requested but no existing run found in %s; starting a fresh run", runs_root)
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def train(config: AgentConfig, resume: bool = False, model_tag: str = "agent", bc_pretrain: bool = False, use_subprocess: bool = False) -> Path:
    """Run PPO training for one window; returns path to its final model."""
    # config.log_dir is already run-scoped (see rolling_eval.run_rolling_eval), so the
    # filename doesn't need its own timestamp — the directory disambiguates the run.
    log_path = config.log_dir / f"{model_tag}.jsonl"

    # Parallel rollout collection: N envs batched N-wide through the policy.
    # DummyVecEnv (in-process, default): env.step is cheap numpy, so the N-wide policy
    # forward is the real speedup, and all N envs share the lru_cache'd tensors by
    # reference. GIL-bound after ~16 workers on systems with plenty of cores.
    # SubprocVecEnv (--subprocess): forks N separate Python processes, avoids GIL,
    # but each subprocess copies tensors (~200MB each), risking OOM on small GPUs.
    def _make_train_env():
        return PortfolioEnv(config, date_range="train")

    if use_subprocess:
        logger.info("Using SubprocVecEnv (%d workers) — parallel, but higher memory", config.n_envs)
        train_env = SubprocVecEnv([_make_train_env for _ in range(config.n_envs)])
    else:
        logger.info("Using DummyVecEnv (%d workers) — shared tensors, lower memory", config.n_envs)
        train_env = DummyVecEnv([_make_train_env for _ in range(config.n_envs)])
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
                gamma=config.effective_gamma,
                gae_lambda=config.gae_lambda,
                ent_coef=config.entropy_coef,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                seed=config.seed,
                # target_kl caps how far a single rollout's n_epochs of minibatch updates can push the
                # policy (SB3 aborts remaining epochs once approx_kl > 1.5*target_kl). With ent_coef=0.0
                # (no entropy pressure) and log_std as an unconstrained nn.Parameter, a volatile rollout
                # (e.g. a crash-era window) can otherwise grind through all n_epochs on an already-large
                # update, collapsing/exploding log_std until Normal()'s log-prob produces NaN gradients —
                # this is the actual root cause of the training-divergence crashes, not gradient magnitude
                # (max_grad_norm=0.5 below is already SB3's default; it clips size, not update *count*).
                target_kl=0.03,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[256, 256], log_std_init=config.log_std_init),
                device=config.device,
                verbose=config.verbose,
            )
            if bc_pretrain:
                from src.agent.bc_pretrain import (
                    collect_bc_dataset,
                    discounted_returns,
                    pretrain_policy,
                    pretrain_value,
                    teacher_policy,
                    train_teacher,
                )

                teacher_env = PortfolioEnv(config, date_range="train")  # separate single env; tensors are lru_cached/shared
                logger.info("BC pretrain [%s]: fitting ranker teacher on window train span...", model_tag)
                teacher_model = train_teacher(teacher_env, seed=config.seed)
                obs_arr, action_arr, mask_arr, reward_arr = collect_bc_dataset(
                    teacher_env, teacher_policy(teacher_env, teacher_model)
                )
                logger.info("BC pretrain [%s]: collected %d decisions, training policy...", model_tag, len(obs_arr))
                pretrain_policy(model, obs_arr, action_arr, mask_arr, seed=config.seed)
                returns_arr = discounted_returns(reward_arr, config.effective_gamma)
                pretrain_value(model, obs_arr, returns_arr, seed=config.seed)
                bc_res = rollout(val_env, agent_policy(model))
                logger.info(
                    "BC pretrain [%s]: val Sharpe after imitation = %.3f (pre-PPO)",
                    model_tag, sharpe_ratio(bc_res["rewards"]),
                )
        logger.info("Training PPO [%s] (device=%s), log → %s", model_tag, model.device, log_path)

        t0 = time.time()
        callback = ValSharpeCallback(config, val_env, log_path, config.total_timesteps, model_tag=model_tag)
        cost_anneal_callback = CostAnnealCallback(config.total_timesteps, anneal_frac=0.5)
        try:
            model.learn(total_timesteps=config.total_timesteps, callback=[callback, cost_anneal_callback])
        except ValueError as e:
            if "invalid values" not in str(e):
                raise
            # A NaN/Inf policy output crashes inside SB3's train() (Normal() rejects it) before this
            # callback regains control, so the periodic diagnostics above never get a chance to log the
            # failing point. Surface what we can reconstruct here instead.
            log_std = model.policy.log_std.data
            logger.error(
                "Training diverged at timestep %d (window=%s): policy produced NaN/Inf action "
                "distribution params. log_std at failure: mean=%.4f min=%.4f max=%.4f (init=%.2f, "
                "clamped to [-5, 2] between rollouts). Last logged val_excess_sharpe=%.3f at timesteps=%d. "
                "Likely cause: an unusually large policy update within one train() call pushed log_std "
                "out of a numerically stable range (target_kl=%.3f should bound this — consider "
                "lowering it, or lowering learning_rate=%.2e, if this recurs).",
                model.num_timesteps, model_tag,
                float(log_std.mean()), float(log_std.min()), float(log_std.max()), config.log_std_init,
                callback.best_excess_sharpe, model.num_timesteps, model.target_kl, float(model.learning_rate),
            )
            raise
        logger.info("Training finished in %.1f min", (time.time() - t0) / 60)

        final_path = config.model_dir / f"{model_tag}_final.zip"
        model.save(final_path)
        write_sidecar(final_path, config, timesteps=model.num_timesteps)
        logger.info("Saved final model → %s (best-val model → %s_best.zip)", final_path, model_tag)

        # Checkpoints are resume-only scratch; once the window is done, drop the leftover.
        for ckpt in config.model_dir.glob(f"{model_tag}_checkpoint_*.zip"):
            ckpt.unlink(missing_ok=True)

        return final_path
    finally:
        # main() calls train() once per rolling window in the same process;
        # close envs explicitly so nothing piles up across windows.
        train_env.close()
        val_env.close()


def main(session_id: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train PPO portfolio agent via anchored rolling windows")
    parser.add_argument("--timesteps", type=int, default=None, help="PPO timesteps PER WINDOW (default: 1,000,000)")
    parser.add_argument("--train-years", type=int, default=None, help="Years of history per window's train span (default: 10)")
    parser.add_argument("--test-years", type=int, default=None, help="Years per window's held-out test span (default: 2)")
    parser.add_argument("--val-fraction", type=float, default=None, help="Fraction of each window's train span carved out for early-stopping val (default: 0.15)")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate (default: 3e-4)")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor (default: 0.997)")
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy coefficient (default: 0.001)")
    parser.add_argument("--log-std-init", type=float, default=None, help="Initial policy log_std (default: -3.0)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (default: 64)")
    parser.add_argument("--eval-freq", type=int, default=None, help="Evaluate on val set every N episodes (default: 20)")
    parser.add_argument("--rebalance-days", type=int, default=None, help="Rebalance interval in trading days (default: 21)")
    parser.add_argument("--universe-size", type=int, default=None, help="Filter to top N tickers by market cap (default: None = all)")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Device (default: cuda)")
    parser.add_argument("--n-envs", type=int, default=None, help="In-process envs batched through the policy (default: 8)")
    parser.add_argument("--resume", action="store_true", help="Resume the currently in-progress window from its latest checkpoint")
    parser.add_argument("--bc-pretrain", action="store_true",
                         help="Warm-start the policy via behavior cloning from a supervised ranker before PPO fine-tuning")
    parser.add_argument("--subprocess", action="store_true",
                         help="Use SubprocessVecEnv (parallel worker processes) instead of DummyVecEnv (single-process, GIL-bound). "
                              "Faster for high --n-envs but uses more memory (each subprocess copies tensors)")
    parser.add_argument("--detect-anomaly", action="store_true",
                         help="Enable torch.autograd anomaly detection to pinpoint the exact backward op "
                              "that first produces NaN/Inf (debugging only — significant slowdown)")
    args = parser.parse_args()

    if session_id is None:
        session_id = _resolve_session_id(DEFAULT_CONFIG, resume=args.resume)

    run_log_dir = DEFAULT_CONFIG.log_dir / "agent" / "runs" / session_id
    log_path = configure_logging(run_log_dir, session_id, tag="train")
    logger.info("Session log → %s", log_path)

    if args.detect_anomaly:
        import torch
        torch.autograd.set_detect_anomaly(True)
        logger.warning("torch.autograd anomaly detection enabled — training will be significantly slower")

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
    if args.gamma is not None:
        overrides["gamma"] = args.gamma
    if args.ent_coef is not None:
        overrides["entropy_coef"] = args.ent_coef
    if args.log_std_init is not None:
        overrides["log_std_init"] = args.log_std_init
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.eval_freq is not None:
        overrides["eval_freq"] = args.eval_freq
    if args.rebalance_days is not None:
        overrides["rebalance_interval_days"] = args.rebalance_days
    if args.universe_size is not None:
        overrides["universe_size"] = args.universe_size
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
    results = run_rolling_eval(base_config, resume=args.resume, session_id=session_id, bc_pretrain=args.bc_pretrain, use_subprocess=args.subprocess)
    finalize_and_report(results, base_config)


if __name__ == "__main__":
    main()
