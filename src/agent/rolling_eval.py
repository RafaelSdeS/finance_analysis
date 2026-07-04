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

import json
import logging
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


if __name__ == "__main__":
    _selfcheck_stitch()
