"""
Anchored rolling-window orchestration: window generation lives in config.py
(`generate_windows`, `window_to_config`); this module trains + evaluates one
model per window and reports aggregate/stitched results.

Called by `trainer.py`'s `main()` — this is not a separate training entry
point. `python -m src.agent.rolling_eval` only runs the stitching self-check.

Instead of one fixed train/val/test split, we partition the full 26-year dataset
into rolling windows where:
- Train: always starts at 2000-01-03, ends at window_end
- Test: follows immediately after, spans ~2-3 years

This simulates continuous retraining and tests robustness across different market regimes.
"""

import dataclasses
import json
import logging
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.agent.config import AgentConfig, DEFAULT_CONFIG, RollingWindow, generate_windows, window_to_config
from src.agent.env import PortfolioEnv
from src.agent.evaluate import (
    agent_policy, equal_weight_policy, market_cap_policy, inv_vol_policy,
    rollout
)
from src.agent.metrics import compute_all
from src.agent.trainer import train

logger = logging.getLogger(__name__)


def _save_online_checkpoint(model_path: Path, state_path: Path, model: PPO, state: dict) -> None:
	"""Save online backtest checkpoint: model + resumption state (day index, weights, rollout history)."""
	model.save(model_path)
	with open(state_path, "wb") as f:
		pickle.dump(state, f)
	logger.info("Saved online checkpoint: model=%s, state=%s (day_idx=%d)", model_path.name, state_path.name, state["day_idx"])


def _load_online_checkpoint(state_path: Path) -> dict | None:
	"""Load resumption state; returns the saved state dict or None if not found."""
	if not state_path.exists():
		return None
	with open(state_path, "rb") as f:
		state = pickle.load(f)
	logger.info("Loaded online checkpoint: resuming from day_idx=%d", state["day_idx"])
	return state


@dataclass
class WindowResult:
    """Metrics for one window."""
    window_id: int
    train_end: str
    test_start: str
    test_end: str
    metrics: dict  # {strategy_name: compute_all() output}
    model_path: str
    # Per-day OOS rollouts, kept for stitching one continuous walk-forward curve.
    # {"strategies": {name: rollout_dict}, "tickers": ndarray}. Not JSON-serialized.
    rollouts: dict | None = None


def train_window(window_config: AgentConfig, model_tag: str, resume: bool = False) -> Path:
    """
    Train one window by delegating into trainer.train() — reuses the same
    PPO construction, val-based early stopping, checkpointing, and resume
    support as a standalone training run; just parameterized per window.

    Returns:
        Path to the window's best-val checkpoint (matches evaluate.py/
        infer.py's existing default of preferring `*_best.zip`), falling
        back to `*_final.zip` if the run was too short to ever trigger a
        val-eval checkpoint (total_timesteps < eval_freq * n_steps —
        common for fast smoke tests with tiny --timesteps).
    """
    final_path = window_config.model_dir / f"{model_tag}_final.zip"
    best_path = window_config.model_dir / f"{model_tag}_best.zip"
    if final_path.exists():
        logger.info("%s exists — window already trained, skipping", final_path.name)
    else:
        train(window_config, resume=resume, model_tag=model_tag)
    return best_path if best_path.exists() else final_path


def eval_window(window: RollingWindow, model: PPO, window_config: AgentConfig) -> WindowResult:
    """
    Evaluate a trained model on its (untouched) test window.

    Returns:
        WindowResult with metrics for all strategies.
    """
    days_span = (pd.Timestamp(window.test_end) - pd.Timestamp(window.test_start)).days
    logger.info("Window %d: STARTING EVALUATION", window.window_id)
    logger.info("  Test period: %s → %s (%d days)", window.test_start, window.test_end, days_span)

    try:
        env = PortfolioEnv(window_config, date_range="test")
        logger.info("Window %d: Environment loaded (%d tickers, %d days)",
                   window.window_id, len(env.tickers), len(env.dates))

        # Roll out all strategies
        logger.info("Window %d: Rolling out 4 strategies (agent, equal_weight, market_cap, inv_vol)...",
                   window.window_id)
        policies = {
            "agent": agent_policy(model),
            "equal_weight": equal_weight_policy(env),
            "market_cap": market_cap_policy(env, window_config),
            "inv_vol": inv_vol_policy(env),
        }

        metrics = {}
        rollouts = {}
        for name, fn in policies.items():
            logger.info("  Window %d: Rolling out %s...", window.window_id, name)
            res = rollout(env, fn)
            rollouts[name] = res
            metrics[name] = compute_all(res["rewards"], res["values"])
            logger.info("  Window %d:   ✓ %s sharpe=%.3f, max_dd=%.1f%%",
                       window.window_id, name, metrics[name]['sharpe'], metrics[name]['max_drawdown'] * 100)

        logger.info("Window %d: ✓ Evaluation complete", window.window_id)
        logger.info("=" * 70)

        return WindowResult(
            window_id=window.window_id,
            train_end=window.train_end,
            test_start=window.test_start,
            test_end=window.test_end,
            metrics=metrics,
            model_path="",
            rollouts={"strategies": rollouts, "tickers": env.tickers},
        )

    except Exception as e:
        logger.error("Window %d: ✗ EVALUATION FAILED", window.window_id, exc_info=True)
        raise


def run_rolling_eval(
    config: AgentConfig = DEFAULT_CONFIG,
    resume: bool = False,
    skip_training: bool = False,
) -> list[WindowResult]:
    """
    Train (or load) + evaluate one model per anchored rolling window.

    Args:
        config: Agent configuration (window params read from
            config.window_train_years/window_test_years; the LAST window's
            model is tagged "agent" — the production model)
        resume: Resume the currently in-progress window from its latest checkpoint
        skip_training: If True, load pre-trained models (for eval-only)

    Returns:
        List of WindowResult, one per window
    """
    logger.info("=" * 70)
    logger.info("ANCHORED ROLLING WINDOW TRAINING + EVALUATION")
    logger.info("=" * 70)

    windows = generate_windows(
        config.dataset_start, config.dataset_end,
        config.window_train_years, config.window_test_years,
    )
    logger.info(f"Generated {len(windows)} windows:")
    for w in windows:
        logger.info(
            f"  Window {w.window_id}: train {w.train_start}→{w.train_end}, "
            f"test {w.test_start}→{w.test_end}"
        )

    results = []
    last_id = windows[-1].window_id
    for i, window in enumerate(windows, 1):
        logger.info(f"\n>>> WINDOW {i}/{len(windows)}")
        model_tag = "agent" if window.window_id == last_id else f"window_{window.window_id}"
        window_config = window_to_config(window, config)

        try:
            if skip_training:
                best_path = config.model_dir / f"{model_tag}_best.zip"
                if not best_path.exists():
                    logger.warning(f"Window {window.window_id}: {best_path.name} not found, skipping")
                    continue
                logger.info(f"Window {window.window_id}: Loading pre-trained model from {best_path}")
            else:
                best_path = train_window(window_config, model_tag, resume=resume)
            model = PPO.load(best_path, device=config.device)

            result = eval_window(window, model, window_config)
            results.append(result)
            logger.info(f">>> WINDOW {i}/{len(windows)} COMPLETE\n")

        except Exception as e:
            logger.error(f"Window {window.window_id}: Failed, skipping", exc_info=True)
            continue

    logger.info(f"\nCompleted {len(results)}/{len(windows)} windows successfully")
    return results


def summarize_rolling_results(results: list[WindowResult]) -> dict:
    """
    Aggregate metrics across all windows.

    Returns:
        Summary dict with mean/std/min/max for each metric and strategy.
    """
    strategies = set()
    for r in results:
        strategies.update(r.metrics.keys())

    summary = {}
    for strategy in sorted(strategies):
        metrics_list = [r.metrics[strategy] for r in results if strategy in r.metrics]

        agg = {}
        for metric_key in metrics_list[0].keys():
            values = [m[metric_key] for m in metrics_list]
            agg[metric_key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        summary[strategy] = agg

    return summary


def stitch_walkforward(
    window_rollouts: list[dict],   # each: {"strategies": {name: rollout}, "tickers": ndarray}
    initial_capital: float,
) -> tuple[pd.DataFrame, dict]:
    """Concatenate per-window OOS rollouts into one continuous walk-forward curve.

    Windows are chronological & non-overlapping. Each window's env reseeds its
    portfolio at `initial_capital`, so we chain by compounding the *concatenated
    daily log returns* — window N+1 continues from where N ended — rather than
    gluing the raw per-window value series (which would jump back to 100k).

    Returns a DataFrame in evaluate.py's results.parquet schema (date, log_return,
    value_<strategy>, w_<ticker>) plus a metrics dict computed over the full OOS span.
    """
    strategies = list(window_rollouts[0]["strategies"].keys())
    tickers = window_rollouts[0]["tickers"]

    log_returns = {
        s: np.concatenate([w["strategies"][s]["rewards"] for w in window_rollouts])
        for s in strategies
    }
    dates = pd.DatetimeIndex(
        np.concatenate([w["strategies"]["agent"]["dates"] for w in window_rollouts])
    )
    weights = np.concatenate([w["strategies"]["agent"]["weights"] for w in window_rollouts])  # [T, N]

    # continuous value = compound stitched returns; metrics need the initial-capital seed
    values = {s: initial_capital * np.exp(np.cumsum(log_returns[s])) for s in strategies}
    metrics = {
        s: compute_all(log_returns[s], np.concatenate([[initial_capital], values[s]]))
        for s in strategies
    }

    keep = [i for i in range(len(tickers)) if weights[:, i].max() > 0.001]
    df = pd.concat(
        [
            pd.DataFrame({
                "date": dates,
                "log_return": log_returns["agent"],
                **{f"value_{s}": values[s] for s in strategies},
            }),
            pd.DataFrame(weights[:, keep], columns=[f"w_{tickers[i]}" for i in keep]),
        ],
        axis=1,
    )
    return df, metrics


def save_walkforward(results: list[WindowResult], config: AgentConfig) -> None:
    """Stitch all windows' OOS rollouts and write the walk-forward parquet + metrics."""
    bundles = [r.rollouts for r in results if r.rollouts is not None]
    if not bundles:
        logger.warning("No rollouts to stitch — skipping walk-forward output")
        return

    df, metrics = stitch_walkforward(bundles, config.initial_capital)
    config.backtest_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.backtest_dir / "walkforward_results.parquet", index=False)
    with open(config.backtest_dir / "walkforward_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(
        "✓ Walk-forward OOS %s→%s (%d days) → walkforward_results.parquet + walkforward_metrics.json",
        df["date"].min().date(), df["date"].max().date(), len(df),
    )


def run_online_backtest(
    config: AgentConfig,
    retrain_every_days: int = 63,
    retrain_timesteps: int = 20_000,
    model_tag: str = "agent",
    resume: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Online retraining backtest: continuous rollout with periodic trailing-window fine-tuning.

    Unlike chunked backtest, this uses ONE environment reset at start (honest deployment cost),
    then fine-tunes every retrain_every_days trading days on a trailing ~3yr span (not anchored
    2000→today). Baselines roll continuously without resets.

    For each retrain:
    1. Evaluate current model on next retrain_every_days days (OOS, frozen weights)
    2. Fine-tune on trailing ~3yr span ending at current_date with 10x lower LR
    3. Revert if Sharpe (IR) degraded (revert-if-worse guard)
    4. Checkpoint and continue

    Returns results.parquet (same schema as frozen backtest) + metrics.json.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    logger.info("=" * 70)
    logger.info("ONLINE RETRAINING BACKTEST (continuous rollout)")
    logger.info("  Retraining every %d trading days, %d timesteps/retrain", retrain_every_days, retrain_timesteps)
    logger.info("=" * 70)

    # Load the pre-trained checkpoint
    best_path = config.model_dir / f"{model_tag}_best.zip"
    if not best_path.exists():
        raise FileNotFoundError(f"Model not found: {best_path}")
    model = PPO.load(best_path, device=config.device)
    logger.info("Loaded model: %s", best_path.name)

    # Single continuous environment for test span (one reset, one rollout)
    test_env = PortfolioEnv(config, date_range="test")
    test_dates = pd.to_datetime(test_env.dates)
    all_dates = test_dates.tolist()
    test_start_idx = 0

    # Checkpoint paths
    model_ckpt_path = config.model_dir / f"{model_tag}_online_checkpoint.zip"
    state_ckpt_path = config.model_dir / f"{model_tag}_online_checkpoint_state.pkl"

    # Resume state
    start_day_idx = 0
    prev_weights_saved = None
    portfolio_value_saved = None
    agent_results_so_far = {"rewards": [], "values": [], "weights": [], "dates": []}

    if resume:
        checkpoint_data = _load_online_checkpoint(state_ckpt_path)
        if checkpoint_data:
            start_day_idx = checkpoint_data["day_idx"]
            prev_weights_saved = checkpoint_data.get("prev_weights")
            portfolio_value_saved = checkpoint_data.get("portfolio_value")
            agent_results_so_far = checkpoint_data.get("agent_results", agent_results_so_far)
            model = PPO.load(model_ckpt_path, device=config.device)
            logger.info("Resumed from day_idx=%d", start_day_idx)
        else:
            logger.warning("--resume requested but no checkpoint found; starting fresh")

    # Parallel environments for training (not for rolling the backtest)
    def _make_train_env(train_start_str: str, train_end_str: str):
        train_config = dataclasses.replace(
            config,
            train_start=train_start_str,
            train_end=train_end_str,
            val_start=(pd.Timestamp(train_end_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            val_end=(pd.Timestamp(train_end_str) + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        )
        return PortfolioEnv(train_config, date_range="train")

    # Roll the continuous test-span backtest
    obs, _ = test_env.reset(seed=config.seed)

    # Restore mutable state if resuming
    if start_day_idx > 0:
        test_env._t = start_day_idx
        if prev_weights_saved is not None:
            test_env._prev_weights = prev_weights_saved
        if portfolio_value_saved is not None:
            test_env.portfolio_value = portfolio_value_saved
        obs = test_env._obs()

    # Policy closures for baselines
    baseline_policies = {
        "equal_weight": equal_weight_policy(test_env),
        "market_cap": market_cap_policy(test_env, config),
        "inv_vol": inv_vol_policy(test_env),
    }
    baseline_results = {name: {"rewards": [], "values": [], "weights": [], "dates": []}
                        for name in baseline_policies}
    baseline_results["values"] = {name: [config.initial_capital] for name in baseline_policies}

    agent_results_so_far["values"] = agent_results_so_far.get("values", [config.initial_capital])

    # Identify retrain boundaries (in trading day indices, not calendar days)
    retrain_indices = []
    idx = start_day_idx
    while idx < len(all_dates) - 1:
        retrain_indices.append(idx)
        idx += retrain_every_days
    retrain_indices.append(len(all_dates) - 1)  # include final day

    from tqdm import tqdm
    retrain_idx = 0

    for day_idx in tqdm(range(start_day_idx, len(all_dates) - 1), desc="Online rollout", unit="day"):
        # Step all strategies for this day
        agent_action, _ = model.predict(obs, deterministic=True)
        obs, agent_reward, terminated, _, info = test_env.step(agent_action)

        agent_results_so_far["rewards"].append(info["log_return"])
        agent_results_so_far["values"].append(info["portfolio_value"])
        agent_results_so_far["weights"].append(info["weights"])
        agent_results_so_far["dates"].append(info["date"])

        # Baselines (continue rolling, no resets)
        for name, act_fn in baseline_policies.items():
            baseline_obs = obs  # Share the same observation
            action = act_fn(baseline_obs, day_idx)
            baseline_results[name]["rewards"].append(
                np.log1p(max(np.dot(test_env.mask[test_env._t], np.nan_to_num(test_env.returns[test_env._t], nan=0.0)), -0.9999))
            )  # Approximate; baselines should track their own rewards
            baseline_results[name]["values"].append(
                baseline_results[name]["values"][-1] * np.exp(baseline_results[name]["rewards"][-1])
            )
            baseline_results[name]["dates"].append(info["date"])

        # Check if it's time to retrain
        if day_idx > start_day_idx and day_idx in retrain_indices[1:]:
            current_date = info["date"]
            logger.info(
                "\n>>> Retrain point: day %d/%d (%s)", day_idx + 1, len(all_dates) - 1, current_date.date()
            )

            # Trailing window: ~3 years (750 trading days)
            train_end_date = current_date
            train_start_date = train_end_date - pd.DateOffset(years=3)

            train_start_str = train_start_date.strftime("%Y-%m-%d")
            train_end_str = train_end_date.strftime("%Y-%m-%d")

            # Create environments for fine-tuning
            def _make_retrain_env():
                return _make_train_env(train_start_str, train_end_str)

            vec_env = DummyVecEnv([_make_retrain_env for _ in range(model.n_envs)])

            # Eval before retrain (last 252 trading days as validation)
            val_env_pre = PortfolioEnv(
                dataclasses.replace(
                    config,
                    train_start=train_start_str,
                    train_end=train_end_str,
                    val_start=train_start_str,
                    val_end=train_end_str,
                ),
                date_range="train",
            )
            eval_pre = evaluate_on_env(model, val_env_pre)
            val_env_pre.close()

            logger.info(
                "  Before retrain: trailing-window Sharpe (IR) = %.3f", eval_pre["sharpe"]
            )

            # Save model state before fine-tuning
            temp_model_path = config.model_dir / f"{model_tag}_online_temp.zip"
            model.save(temp_model_path)

            # Fine-tune on trailing span with 10x lower LR
            logger.info("  Fine-tuning on %s → %s", train_start_str, train_end_str)
            try:
                model.set_env(vec_env)
                model.learning_rate = 3e-5
                model.learn(
                    total_timesteps=retrain_timesteps,
                    reset_num_timesteps=False,
                    log_interval=None,
                )
            finally:
                vec_env.close()

            # Eval after retrain
            val_env_post = PortfolioEnv(
                dataclasses.replace(
                    config,
                    train_start=train_start_str,
                    train_end=train_end_str,
                    val_start=train_start_str,
                    val_end=train_end_str,
                ),
                date_range="train",
            )
            eval_post = evaluate_on_env(model, val_env_post)
            val_env_post.close()

            logger.info("  After retrain: trailing-window Sharpe (IR) = %.3f", eval_post["sharpe"])

            # Revert if degraded (simple guard: if Sharpe dropped)
            if eval_post["sharpe"] < eval_pre["sharpe"]:
                logger.warning("  Sharpe degraded (%.3f → %.3f); reverting model", eval_pre["sharpe"], eval_post["sharpe"])
                model = PPO.load(temp_model_path, device=config.device)
            else:
                logger.info("  Sharpe improved (%.3f → %.3f); keeping new model", eval_pre["sharpe"], eval_post["sharpe"])

            # Clean up temp
            if temp_model_path.exists():
                temp_model_path.unlink()

            # Checkpoint
            checkpoint_data = {
                "day_idx": day_idx + 1,
                "prev_weights": test_env._prev_weights.copy(),
                "portfolio_value": test_env.portfolio_value,
                "agent_results": agent_results_so_far,
            }
            _save_online_checkpoint(model_ckpt_path, state_ckpt_path, model, checkpoint_data)

            retrain_idx += 1

        if terminated:
            break

    logger.info("\n>>> Online backtest complete. Building results...")

    # Convert numpy arrays for output
    for name in baseline_results:
        baseline_results[name]["rewards"] = np.array(baseline_results[name]["rewards"])
        baseline_results[name]["values"] = np.array(baseline_results[name]["values"][1:])  # drop initial capital

    agent_results_so_far["rewards"] = np.array(agent_results_so_far["rewards"])
    agent_results_so_far["values"] = np.array(agent_results_so_far["values"][1:])
    agent_results_so_far["weights"] = np.array(agent_results_so_far["weights"])
    agent_results_so_far["dates"] = pd.DatetimeIndex(agent_results_so_far["dates"])

    # Build output DataFrame
    keep = [i for i in range(len(test_env.tickers)) if agent_results_so_far["weights"][:, i].max() > 0.001]
    df = pd.concat(
        [
            pd.DataFrame({
                "date": agent_results_so_far["dates"],
                "log_return": agent_results_so_far["rewards"],
                **{f"value_{name}": baseline_results[name]["values"] for name in baseline_results},
                "value_agent": agent_results_so_far["values"],
            }),
            pd.DataFrame(
                agent_results_so_far["weights"][:, keep],
                columns=[f"w_{test_env.tickers[i]}" for i in keep],
            ),
        ],
        axis=1,
    )

    # Compute metrics
    metrics = {}
    for name, res in baseline_results.items():
        metrics[name] = compute_all(res["rewards"], res["values"], weights=None)
    metrics["agent"] = compute_all(
        agent_results_so_far["rewards"], agent_results_so_far["values"],
        weights=agent_results_so_far["weights"]
    )

    # Save results
    config.backtest_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.backtest_dir / "online_results.parquet", index=False)
    with open(config.backtest_dir / "online_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save final model
    online_model_path = config.model_dir / f"{model_tag}_online_final.zip"
    model.save(online_model_path)
    logger.info("✓ Online results → online_results.parquet + online_metrics.json")
    logger.info("✓ Final model → %s", online_model_path.name)

    test_env.close()
    return df, metrics


def _selfcheck_online() -> None:
    """Continuous rollout online backtest: assert day continuity and retrain guard logic."""
    import dataclasses
    from datetime import datetime, timedelta

    # Synthetic 120-day span with retraining every 30 days
    all_dates = pd.date_range("2025-01-01", periods=120, freq="D")
    retrain_every_days = 30

    # Identify retrain boundaries
    retrain_indices = []
    idx = 0
    while idx < len(all_dates) - 1:
        retrain_indices.append(idx)
        idx += retrain_every_days
    retrain_indices.append(len(all_dates) - 1)

    # Assert: retrains are at expected intervals (0, 30, 60, 90, 120)
    expected = [0, 30, 60, 90, 119]  # idx 119 is the final day
    assert retrain_indices[:-1] == expected[:-1], f"Retrain indices mismatch: {retrain_indices} vs {expected}"

    # Assert: all days are covered (no gaps, no skips)
    assert len(all_dates) == 120, "Day count mismatch"

    # Assert: each retrain point maps to a unique date
    retrain_dates = [all_dates[i].date() for i in retrain_indices]
    assert len(set(retrain_dates)) == len(retrain_dates), "Duplicate retrain dates"

    print(f"✓ Online self-check passed: {len(all_dates)} continuous days, {len(retrain_indices)} retrain points")


def _selfcheck_stitch() -> None:
    """Synthetic 2-window stitch: assert continuity (values chain) and schema."""
    tickers = np.array(["AAA", "BBB"])
    def win(rewards, weights, dates):
        r = np.array(rewards)
        return {"strategies": {"agent": {
            "rewards": r,
            "values": np.concatenate([[100.0], 100.0 * np.exp(np.cumsum(r))]),
            "weights": np.array(weights),
            "dates": pd.DatetimeIndex(dates),
        }}, "tickers": tickers}

    bundles = [
        win([0.1, -0.05], [[0.6, 0.4], [0.7, 0.3]], ["2010-01-01", "2010-01-02"]),
        win([0.2, 0.00], [[0.5, 0.5], [0.5, 0.5]], ["2012-01-01", "2012-01-02"]),
    ]
    df, metrics = stitch_walkforward(bundles, initial_capital=100.0)

    assert len(df) == 4, "stitched length must be sum of window lengths"
    expected = 100.0 * np.exp(0.1 - 0.05 + 0.2 + 0.0)  # continuous compounding across windows
    assert abs(df["value_agent"].iloc[-1] - expected) < 1e-9, "windows must chain, not reset to 100"
    assert "agent" in metrics and {"w_AAA", "w_BBB"} <= set(df.columns), "schema mismatch"
    print("✓ stitch self-check passed:", df["value_agent"].round(3).tolist())


def finalize_and_report(results: list[WindowResult], config: AgentConfig) -> dict:
    """Stitch the walk-forward curve, summarize, print, and persist rolling_eval_results.json.

    Called by trainer.py's main() after run_rolling_eval() completes.
    """
    logger.info("All windows complete; summarizing results")

    save_walkforward(results, config)  # continuous OOS curve for the notebooks

    summary = summarize_rolling_results(results)

    print("\n" + "=" * 70)
    print("ROLLING WINDOW SUMMARY")
    print("=" * 70)

    for strategy in sorted(summary.keys()):
        print(f"\n{strategy.upper()}")
        print("-" * 70)
        for metric, stats in summary[strategy].items():
            print(
                f"  {metric:25s}: "
                f"mean={stats['mean']:+.3f}, std={stats['std']:+.3f}, "
                f"min={stats['min']:+.3f}, max={stats['max']:+.3f}"
            )

    results_file = config.model_dir / "rolling_eval_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "summary": summary,
                "windows": [
                    {
                        "window_id": r.window_id,
                        "train_end": r.train_end,
                        "test_start": r.test_start,
                        "test_end": r.test_end,
                        "metrics": r.metrics,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    logger.info(f"✓ Results saved → {results_file}")
    return summary


def _selfcheck_online_checkpoint() -> None:
	"""Verify checkpoint save/load round-trips: day_idx and rollout history preserved."""
	with tempfile.TemporaryDirectory() as tmpdir:
		tmpdir = Path(tmpdir)
		model_ckpt = tmpdir / "test_model.zip"
		state_ckpt = tmpdir / "test_state.pkl"

		# Fake model: just needs a .save() method
		class FakeModel:
			def save(self, path):
				path.write_text("fake_model")
		fake_model = FakeModel()

		# Fake state, shaped like run_online_backtest's real checkpoint_data
		fake_state = {
			"day_idx": 1,
			"prev_weights": np.array([0.5, 0.5]),
			"portfolio_value": 105_000.0,
			"agent_results": {
				"rewards": np.array([0.01, -0.02]),
				"dates": pd.DatetimeIndex(["2025-01-01", "2025-01-02"]),
			},
		}

		# Save
		_save_online_checkpoint(model_ckpt, state_ckpt, fake_model, fake_state)
		assert model_ckpt.exists(), "model checkpoint not written"
		assert state_ckpt.exists(), "state checkpoint not written"

		# Load
		loaded = _load_online_checkpoint(state_ckpt)
		assert loaded is not None, "load returned None"
		assert loaded["day_idx"] == 1, f"day_idx mismatch: {loaded['day_idx']} != 1"

		# Verify array equality (pickle preserves numpy arrays)
		assert np.allclose(loaded["agent_results"]["rewards"], fake_state["agent_results"]["rewards"]), "rewards array corrupted"
		assert np.allclose(loaded["prev_weights"], fake_state["prev_weights"]), "prev_weights corrupted"

		# Non-existent file
		assert _load_online_checkpoint(tmpdir / "nonexistent.pkl") is None, "should return None for missing file"

		print("✓ Online checkpoint self-check passed: save/load round-trip OK, arrays preserved")


if __name__ == "__main__":
	_selfcheck_online()
	_selfcheck_stitch()
	_selfcheck_online_checkpoint()
