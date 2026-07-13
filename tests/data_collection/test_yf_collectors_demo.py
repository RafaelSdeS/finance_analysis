#!/usr/bin/env python3
"""
Self-check for yf_collectors.py's pure helper functions -- most importantly
_prices_fetch_start(), the staleness-anchor fix for `--mode update` (CLAUDE.md:
without it, a dividend paid after one quarter's fetch would permanently fail
to propagate back into that quarter's already-stored adj_close -- one silent,
cumulative discontinuity per update, forever).

The checks already exist as yf_collectors._demo() (this repo's convention for
a lightweight self-check on non-trivial logic, run via `python -m
src.data_collection.yf_collectors`) but were never wired into run_all.py/CI,
so a regression here would go unnoticed. yf_collectors.py uses package-relative
imports (`from . import checkpoint, config, validate`), so it can't run as a
bare script outside its package -- this file just imports and runs the
existing _demo() under the standard test-runner convention instead of
duplicating its assertions.

Run from project root: python tests/data_collection/test_yf_collectors_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection.yf_collectors import _demo

if __name__ == "__main__":
    _demo()
