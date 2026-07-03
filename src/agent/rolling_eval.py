"""
Anchored rolling-window evaluation for robustness testing.

Instead of one fixed train/val/test split, we partition the full 26-year dataset
into rolling windows where:
- Train: always starts at 2000-01-03, ends at window_end
- Test: follows immediately after, spans ~2-3 years

This simulates continuous retraining and tests robustness across different market regimes.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.agent.config import AgentConfig, DEFAULT_CONFIG
from src.agent.env import PortfolioEnv
from src.agent.evaluate import (
    agent_policy, equal_weight_policy, market_cap_policy, inv_vol_policy,
    rollout
)
from src.agent.metrics import compute_all

logger = logging.getLogger(__name__)


class WindowConfig:
    """Minimal config shim bypassing AgentConfig's split-order validation."""

    def __init__(self, base_config, *, train_start, train_end,
                 val_start, val_end, test_start, test_end):
        self.train_start = train_start
        self.train_end = train_end
        self.val_start = val_start
        self.val_end = val_end
        self.test_start = test_start
        self.test_end = test_end
        self.model_dir = base_config.model_dir
        self.initial_capital = base_config.initial_capital
        self.device = base_config.device


class RollingWindow(NamedTuple):
    """A single train/test window."""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


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


def generate_windows(
    full_dataset_start: str = "2000-01-03",
    full_dataset_end: str = "2026-06-30",
    train_years: int = 10,
    test_years: int = 2,
) -> list[RollingWindow]:
    """
    Generate anchored rolling windows.

    Train always starts at full_dataset_start, test window slides forward.
    Non-overlapping windows preserve temporal integrity.
    """
    start_date = pd.Timestamp(full_dataset_start)
    end_date = pd.Timestamp(full_dataset_end)

    windows = []
    window_id = 0

    # Slide the test window forward by test_years each iteration
    test_start = start_date + pd.DateOffset(years=train_years)

    while test_start + pd.DateOffset(years=test_years) <= end_date:
        test_end = test_start + pd.DateOffset(years=test_years)
        train_end = test_start - pd.Timedelta(days=1)

        windows.append(RollingWindow(
            window_id=window_id,
            train_start=start_date.strftime("%Y-%m-%d"),
            train_end=train_end.strftime("%Y-%m-%d"),
            test_start=test_start.strftime("%Y-%m-%d"),
            test_end=test_end.strftime("%Y-%m-%d"),
        ))

        window_id += 1
        test_start = test_end + pd.Timedelta(days=1)

    return windows


def train_window(
    window: RollingWindow,
    config: AgentConfig = DEFAULT_CONFIG,
    timesteps: int = 100_000,
) -> tuple[PPO, str]:
    """
    Train a single model for one window's train period.

    Returns:
        (trained_model, checkpoint_path)
    """
    days_span = (pd.Timestamp(window.train_end) - pd.Timestamp(window.train_start)).days
    logger.info("=" * 70)
    logger.info("Window %d: STARTING TRAINING", window.window_id)
    logger.info("  Train period: %s → %s (%d days)", window.train_start, window.train_end, days_span)
    logger.info("  Timesteps: %d (CPU; ~50 min/window at 1M)", timesteps)

    try:
        w_config = WindowConfig(
            config,
            train_start=window.train_start, train_end=window.train_end,
            val_start=window.train_end, val_end=window.train_end,
            test_start=window.train_end, test_end=window.train_end,
        )

        logger.info("Window %d: Loading environment...", window.window_id)
        env = PortfolioEnv(w_config, date_range="train")
        logger.info("Window %d: Environment loaded (%d tickers, %d days)",
                   window.window_id, len(env.tickers), len(env.dates))

        # Train PPO (using CPU for better MlpPolicy performance)
        logger.info("Window %d: Initializing PPO (using CPU)...", window.window_id)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device="cpu",  # MlpPolicy runs better on CPU
            verbose=1,
        )

        logger.info("Window %d: Training for %d timesteps...", window.window_id, timesteps)
        model.learn(total_timesteps=timesteps, progress_bar=True)

        logger.info("Window %d: Training complete! Saving model...", window.window_id)

        # Save checkpoint
        checkpoint_path = config.model_dir / f"window_{window.window_id}_model.zip"
        model.save(checkpoint_path)
        logger.info("Window %d: ✓ Model saved → %s", window.window_id, checkpoint_path)
        logger.info("=" * 70)

        return model, str(checkpoint_path)

    except Exception as e:
        logger.error("Window %d: ✗ TRAINING FAILED", window.window_id, exc_info=True)
        raise


def eval_window(
    window: RollingWindow,
    model: PPO,
    config: AgentConfig = DEFAULT_CONFIG,
) -> WindowResult:
    """
    Evaluate a trained model on its test window.

    Returns:
        WindowResult with metrics for all strategies.
    """
    days_span = (pd.Timestamp(window.test_end) - pd.Timestamp(window.test_start)).days
    logger.info("Window %d: STARTING EVALUATION", window.window_id)
    logger.info("  Test period: %s → %s (%d days)", window.test_start, window.test_end, days_span)

    try:
        w_config = WindowConfig(
            config,
            train_start="2000-01-03", train_end="2000-01-03",  # dummy, not used
            val_start="2000-01-03", val_end="2000-01-03",
            test_start=window.test_start, test_end=window.test_end,
        )

        logger.info("Window %d: Loading test environment...", window.window_id)
        env = PortfolioEnv(w_config, date_range="test")
        logger.info("Window %d: Environment loaded (%d tickers, %d days)",
                   window.window_id, len(env.tickers), len(env.dates))

        # Roll out all strategies
        logger.info("Window %d: Rolling out 4 strategies (agent, equal_weight, market_cap, inv_vol)...",
                   window.window_id)
        policies = {
            "agent": agent_policy(model),
            "equal_weight": equal_weight_policy(env),
            "market_cap": market_cap_policy(env, config),
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
    train_years: int = 10,
    test_years: int = 2,
    timesteps_per_window: int = 100_000,
    skip_training: bool = False,
) -> list[WindowResult]:
    """
    Run full anchored rolling window evaluation.

    Args:
        config: Agent configuration
        train_years: Years of training history per window
        test_years: Years of test period per window
        timesteps_per_window: PPO timesteps to train each window
        skip_training: If True, load pre-trained models (for eval-only)

    Returns:
        List of WindowResult, one per window
    """
    logger.info("=" * 70)
    logger.info("ANCHORED ROLLING WINDOW EVALUATION")
    logger.info("=" * 70)

    windows = generate_windows(train_years=train_years, test_years=test_years)
    logger.info(f"Generated {len(windows)} windows:")
    for w in windows:
        logger.info(
            f"  Window {w.window_id}: train {w.train_start}→{w.train_end}, "
            f"test {w.test_start}→{w.test_end}"
        )

    results = []
    for i, window in enumerate(windows, 1):
        logger.info(f"\n>>> WINDOW {i}/{len(windows)}")

        try:
            if not skip_training:
                model, _ = train_window(window, config, timesteps_per_window)
            else:
                # Load pre-trained model (if exists)
                model_path = config.model_dir / f"window_{window.window_id}_model.zip"
                if not model_path.exists():
                    logger.warning(f"Window {window.window_id}: model not found, skipping")
                    continue
                logger.info(f"Window {window.window_id}: Loading pre-trained model from {model_path}")
                model = PPO.load(model_path, device="cpu")

            result = eval_window(window, model, config)
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


def main():
    parser = argparse.ArgumentParser(description="Anchored walk-forward train + eval")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="PPO timesteps per window (lower = faster; 100000 ≈ 10x quicker)")
    parser.add_argument("--test-years", type=int, default=2, help="Test span per window")
    parser.add_argument("--skip-training", action="store_true",
                        help="Reuse existing window_*_model.zip; only eval + stitch")
    args = parser.parse_args()

    # Setup detailed logging with timestamp
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger.info("ROLLING WINDOW EVALUATION — STARTING")

    config = DEFAULT_CONFIG

    try:
        results = run_rolling_eval(
            config=config,
            train_years=10,
            test_years=args.test_years,
            timesteps_per_window=args.timesteps,
            skip_training=args.skip_training,
        )
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

        # Save results
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
        logger.info("ROLLING WINDOW EVALUATION — COMPLETE")

    except Exception as e:
        logger.error("FATAL ERROR: Rolling window evaluation failed", exc_info=True)
        raise


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck_stitch()
    else:
        main()
