"""
Inference: predict portfolio weights for a given date.

Reuses PortfolioEnv for observation construction and masked softmax so the
state seen at inference is byte-identical to training. Falls back to
equal-weight over active tickers if the model cannot be loaded or predicts
garbage — a daily allocation must never crash.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent.config import AgentConfig, DEFAULT_CONFIG
from src.agent.env import PortfolioEnv
from src.agent.model_provenance import check_sidecar

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = DEFAULT_CONFIG.model_dir / "agent_best.zip"
STALE_DAYS_THRESHOLD = 7  # ~5 trading days; beyond this the tensors don't cover the requested date


def predict_weights(
    date: str | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    config: AgentConfig = DEFAULT_CONFIG,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Predict portfolio weights for `date` (default: latest available date).

    Args:
        strict: if True, raise instead of warning when `date` is beyond the
            tensors' coverage (i.e. the allocation would silently reflect an
            older date than requested).

    Returns:
        DataFrame [ticker, weight] for active tickers with weight > 0,
        sorted by weight descending. Weights sum to 1. `result.attrs` carries
        "date" (the date actually used), "source", and "stale_days" (0 if
        `date` falls within tensor coverage).
    """
    env = PortfolioEnv(config, date_range="test")

    # Resolve date → index in the test calendar
    if date is None:
        t = len(env.dates) - 1
        stale_days = 0
    else:
        ts = pd.Timestamp(date)
        matches = np.where(env.dates <= ts)[0]
        if len(matches) == 0:
            raise ValueError(f"Date {date} predates test range start {env.dates[0].date()}")
        t = int(matches[-1])  # last trading day <= requested date
        stale_days = max(0, (ts - env.dates[-1]).days)
    target_date = env.dates[t]
    active = env.mask[t]

    if stale_days > STALE_DAYS_THRESHOLD:
        msg = (
            f"Requested date {date} is {stale_days} days past the newest data in "
            f"agent_tensors.npz ({env.dates[-1].date()}). This allocation reflects "
            f"{target_date.date()}, NOT {date}. Rebuild the dataset (feature_engineering.py "
            f"+ data_pipeline.py, after refreshing raw data) to get a current allocation."
        )
        if strict:
            raise ValueError(msg)
        logger.warning("⚠ STALE ALLOCATION: %s", msg)

    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device=config.device)
        check_sidecar(model_path, config)
        env._t = t
        action, _ = model.predict(env._obs(), deterministic=True)
        weights = env.action_to_weights(action, active)
        source = f"model ({model_path.name})"
    except Exception:
        logger.exception("Inference failed — falling back to equal weight")
        weights = active.astype(np.float64) / active.sum()
        source = "FALLBACK equal-weight"

    if not np.isfinite(weights).all() or abs(weights.sum() - 1.0) > 1e-6:
        logger.warning("Model produced invalid weights — falling back to equal weight")
        weights = active.astype(np.float64) / active.sum()
        source = "FALLBACK equal-weight"

    result = pd.DataFrame({"ticker": env.tickers, "weight": weights})
    result = result[result["weight"] > 1e-6].sort_values("weight", ascending=False)
    result = result.reset_index(drop=True)

    logger.info(
        "Allocation for %s via %s: %d positions, top: %s (%.1f%%)%s",
        target_date.date(), source, len(result),
        result.iloc[0]["ticker"], result.iloc[0]["weight"] * 100,
        f" [STALE by {stale_days}d]" if stale_days > STALE_DAYS_THRESHOLD else "",
    )
    result.attrs["date"] = str(target_date.date())
    result.attrs["source"] = source
    result.attrs["stale_days"] = stale_days
    return result
