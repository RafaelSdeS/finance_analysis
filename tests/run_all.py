#!/usr/bin/env python3
"""
Single entry point for the test suite: runs each test script and prints
one PASS/FAIL summary line per script instead of 8 separate transcripts.

Run from project root:
    python tests/run_all.py                # fast group (pure code, no data files needed)
    python tests/run_all.py --group data    # needs data/raw + a built ml_dataset.parquet
    python tests/run_all.py --group all
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Pure-code tests: synthetic data only, run anywhere (used by CI).
FAST = [
    "tests/build_dataset/test_build_dataset_features.py",
    "tests/build_dataset/test_split_config.py",
    "tests/build_dataset/test_dataset_versioning.py",
]

# Needs data/raw/* on disk (git-tracked) and/or a built data/processed/ml_dataset.parquet.
DATA = [
    "tests/build_dataset/test_final_dataset.py",
    "tests/data_collection/test_cagr_calculation.py",
    "tests/data_collection/test_blue_chip_tickers.py",
]


def run(script: str) -> bool:
    return subprocess.run([sys.executable, script], cwd=ROOT).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["fast", "data", "all"], default="fast")
    args = parser.parse_args()

    scripts = {"fast": FAST, "data": DATA, "all": FAST + DATA}[args.group]
    results = [(script, run(script)) for script in scripts]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for script, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {script}")

    failed = sum(not ok for _, ok in results)
    print(f"\n{len(results) - failed}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
