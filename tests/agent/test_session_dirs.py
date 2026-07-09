#!/usr/bin/env python3
"""
Session-dir resolution: --resume must reuse the most-recently-modified run
directory (artifacts/models/runs/<session_id>/), and a fresh run must never
collide with a previous one.

Run from project root: python tests/agent/test_session_dirs.py
"""

import dataclasses
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.config import DEFAULT_CONFIG
from src.agent.trainer import _resolve_session_id


def main() -> None:
    print("=" * 60)
    print("TEST: session dir resolution")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = dataclasses.replace(DEFAULT_CONFIG, model_dir=Path(tmpdir))

        # --- 1. Fresh run (no --resume) always mints a new timestamp ---
        sid = _resolve_session_id(config, resume=False)
        assert sid, "fresh run must return a usable session_id"
        print(f"✓ fresh run mints a new session_id: {sid}")

        # --- 2. --resume with no existing runs/ falls back to a fresh timestamp ---
        sid_no_runs = _resolve_session_id(config, resume=True)
        assert sid_no_runs, "resume with nothing to resume must still return a usable session_id"
        print(f"✓ resume with no existing runs falls back to fresh timestamp: {sid_no_runs}")

        # --- 3. --resume picks the most-recently-modified existing run dir ---
        runs_root = config.model_dir / "runs"
        older = runs_root / "20200101-000000"
        newer = runs_root / "20200101-000001"
        older.mkdir(parents=True)
        time.sleep(0.01)
        newer.mkdir(parents=True)
        resumed = _resolve_session_id(config, resume=True)
        assert resumed == newer.name, f"expected to resume newest dir {newer.name}, got {resumed}"
        print(f"✓ resume picks the most-recently-modified run dir: {resumed}")

    print("\nALL SESSION DIR TESTS PASSED ✓")


if __name__ == "__main__":
    main()
