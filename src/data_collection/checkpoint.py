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


# Re-probe a skip-listed ticker every Nth run instead of excluding it forever.
# Mirrors yf_collectors.EMPTY_RUNS_REPROBE_INTERVAL (same value, same semantics) --
# the yfinance path already had this safety net; BolsAI's `_skip` did not, and
# cleared only by hand-editing the checkpoint JSON. With MAX_RETRIES previously
# at 1 (i.e. no retries at all), a single transient BolsAI 503/timeout was enough
# to blacklist a live ticker permanently and silently.
SKIP_REPROBE_INTERVAL = 10


def load_skip(cp: dict) -> dict:
    """Normalize `cp["_skip"]` to {ticker: consecutive_failure_count}.

    Accepts the legacy plain-list format still on disk (e.g. full_scale/prices.json's
    63 entries): count is unknown for those, so they're seeded due for an immediate
    re-probe rather than trusted as permanently dead.
    """
    raw = cp.get("_skip", [])
    if isinstance(raw, list):
        return {t: SKIP_REPROBE_INTERVAL for t in raw}
    return dict(raw)


def should_skip(skip: dict, ticker: str) -> bool:
    """True if `ticker` is skip-listed AND this isn't its scheduled re-probe run."""
    n = skip.get(ticker, 0)
    return n > 0 and n % SKIP_REPROBE_INTERVAL != 0


def mark_skip(name: str, mode: str, cp: dict, skip: dict, ticker: str) -> None:
    """Record one more unsuccessful outcome for `ticker` and persist.

    Called both when a fetch actually fails AND when a ticker is skipped without
    being attempted -- the counter has to advance on skipped runs too, or a
    ticker sitting on a multiple of SKIP_REPROBE_INTERVAL would re-probe on
    every single run instead of every Nth. Same design as yf_collectors'
    `empty_runs`, which increments on its skip path for exactly this reason.
    Mutates `skip` and `cp` in place.
    """
    skip[ticker] = skip.get(ticker, 0) + 1
    cp["_skip"] = dict(sorted(skip.items()))
    save(name, mode, cp)


def clear_skip(name: str, mode: str, cp: dict, skip: dict, ticker: str) -> None:
    """Drop `ticker` from the negative cache after a successful collection.

    Essential, not tidiness: a re-probe that SUCCEEDS must reset the count, or
    the ticker keeps its stale non-zero count and gets skipped again on the very
    next run despite now having real data on disk.
    """
    if skip.pop(ticker, None) is not None:
        cp["_skip"] = dict(sorted(skip.items()))
        save(name, mode, cp)
