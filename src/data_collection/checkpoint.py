"""
checkpoint.py — resume state, one JSON file per collector per mode.

Enables idempotent, resumable runs: a collector reads its checkpoint to fetch
only new data, and writes it back after a successful save.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from . import config


def _path(name: str, mode: str) -> Path:
    return config.CHECKPOINT_ROOT / mode / f"{name}.json"


def load(name: str, mode: str) -> dict:
    p = _path(name, mode)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save(name: str, mode: str, data: dict) -> None:
    p = _path(name, mode)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {**data, "last_update": datetime.now(timezone.utc).isoformat()}
    p.write_text(json.dumps(data, indent=2, default=str))
