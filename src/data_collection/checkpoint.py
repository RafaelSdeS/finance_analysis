"""
checkpoint.py — resume state, one JSON file per collector per mode.

Enables idempotent, resumable runs: a collector reads its checkpoint to fetch
only new data, and writes it back after a successful save.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from . import config

_lock = threading.Lock()


def _path(name: str, mode: str) -> Path:
    return config.CHECKPOINT_ROOT / mode / f"{name}.json"


def load(name: str, mode: str) -> dict:
    with _lock:
        p = _path(name, mode)
        if p.exists():
            return json.loads(p.read_text())
        return {}


def save(name: str, mode: str, data: dict) -> None:
    with _lock:
        p = _path(name, mode)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {**data, "last_update": datetime.now(timezone.utc).isoformat()}
        p.write_text(json.dumps(data, indent=2, default=str))


def mark_skip(name: str, mode: str, cp: dict, skip: set, ticker: str) -> None:
    """Add `ticker` to the collector's negative cache (`cp["_skip"]`) and persist.

    Shared by collectors that record tickers with no data (persistent 404s/
    500s) so a rerun doesn't burn the retry/backoff budget on them again.
    Mutates `skip` and `cp` in place, matching the closures this replaces.
    """
    skip.add(ticker)
    cp["_skip"] = sorted(skip)
    save(name, mode, cp)
