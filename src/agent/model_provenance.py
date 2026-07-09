"""
Model provenance sidecar: records the env/training semantics a saved model
was trained under, next to the .zip, so evaluate.py/infer.py can detect a
stale or mismatched model instead of guessing via an unconditional warning.

Standalone module (not trainer.py or evaluate.py) so both sides — writer at
save time, reader at load time — can import it without a circular dependency
(trainer.py already imports from evaluate.py).
"""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

from src.agent.config import AgentConfig

logger = logging.getLogger(__name__)

# Fields that define "env semantics": if the loaded model's action output would be
# interpreted differently under the current config, it belongs here. Excludes
# universe_size — that's a load-time validation guard only (env.py raises if the
# tensor file's actual ticker count doesn't match it); it never enters the weight
# math, and None just means "no constraint requested", not a behavioral difference.
SEMANTIC_FIELDS = [
    "rebalance_interval_days", "logit_scale", "max_position_weight",
    "transaction_cost_bps",
]


def _git_info() -> dict:
    """Best-effort git SHA + dirty flag; None values if not in a git repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip())
        return {"git_sha": sha, "git_dirty": dirty}
    except Exception:
        return {"git_sha": None, "git_dirty": None}


def sidecar_path(model_path: Path) -> Path:
    return model_path.with_suffix(".json")


def write_sidecar(model_path: Path, config: AgentConfig, timesteps: int | None = None) -> None:
    """Write a provenance sidecar next to a saved model .zip."""
    payload = {
        "model_file": model_path.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "timesteps": timesteps,
        "window_id": config.window_id,
        "train_start": config.train_start,
        "train_end": config.train_end,
        "val_start": config.val_start,
        "val_end": config.val_end,
        "test_start": config.test_start,
        "test_end": config.test_end,
        **{f: getattr(config, f) for f in SEMANTIC_FIELDS},
        "universe_size": config.universe_size,  # informational only, not a semantic field (see SEMANTIC_FIELDS comment)
        "state_features": list(config.state_features),
        **_git_info(),
    }
    with open(sidecar_path(model_path), "w") as f:
        json.dump(payload, f, indent=2)


def check_sidecar(model_path: Path, config: AgentConfig) -> dict | None:
    """Load a model's sidecar and compare its env semantics to `config`.

    Logs a loud warning naming exact mismatched fields, or a single info line
    confirming provenance if everything matches. Returns the sidecar dict, or
    None if no sidecar exists (e.g. a model saved before this was added).
    """
    path = sidecar_path(model_path)
    if not path.exists():
        logger.warning(
            "No provenance sidecar for %s — model's training config is unknown "
            "(saved before provenance tracking, or copied in manually).",
            model_path.name,
        )
        return None

    with open(path) as f:
        sidecar = json.load(f)

    mismatches = []
    for field in SEMANTIC_FIELDS:
        current = getattr(config, field)
        saved = sidecar.get(field)
        if saved != current:
            mismatches.append(f"{field}: model={saved!r} vs current_config={current!r}")

    saved_features = sidecar.get("state_features")
    if saved_features is not None and list(saved_features) != list(config.state_features):
        mismatches.append(
            f"state_features: model has {len(saved_features)} features, "
            f"current config has {len(config.state_features)}"
        )

    if mismatches:
        logger.warning(
            "⚠ Model %s (trained %s, window=%s) MISMATCHES current config — "
            "predictions may be misinterpreted:\n  %s",
            model_path.name, sidecar.get("timestamp"), sidecar.get("window_id"),
            "\n  ".join(mismatches),
        )
    else:
        logger.info(
            "Model provenance OK: %s trained %s (window=%s, git=%s%s)",
            model_path.name, sidecar.get("timestamp"), sidecar.get("window_id"),
            (sidecar.get("git_sha") or "unknown")[:8],
            "-dirty" if sidecar.get("git_dirty") else "",
        )
    return sidecar
