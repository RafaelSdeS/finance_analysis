"""
test_reanalyze.py -- checks for reanalyze.dsr_by_era: slicing a saved
run's series to a cutoff date and recomputing deflated Sharpe on each era.

Fast group (synthetic only). Run: python tests/portfolio/test_reanalyze.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.portfolio.reanalyze import dsr_by_era  # noqa: E402
from tests.test_utils import print_check, print_header, print_section_end  # noqa: E402


def main():
    print_header("test_reanalyze")
    passed = failed = 0

    # Two-regime series: a bad first half (negative mean), a clearly good second
    # half (strong positive mean) -- mirrors the pre/post-2011 pattern actually
    # observed (2026-07-26 top_n=50 run: pre=-1.37%, post=+1.51% ann.).
    rng = np.random.default_rng(0)
    dates = pd.date_range("2005-01-01", periods=2000, freq="D")
    bad = rng.normal(-0.001, 0.01, 1000)
    good = rng.normal(0.003, 0.01, 1000)
    series = pd.Series(np.concatenate([bad, good]), index=dates)
    cutoff = dates[1000]

    table = dsr_by_era(series, cutoff, n_trials=16)

    full_n_ok = table["full"]["n"] == 2000
    print_check("full slice keeps every observation", bool(full_n_ok), f"got n={table['full']['n']}")
    passed, failed = passed + full_n_ok, failed + (not full_n_ok)

    since_n_ok = table["since_cutoff"]["n"] == 1000
    print_check("since_cutoff slice keeps only dates >= cutoff", bool(since_n_ok),
                f"got n={table['since_cutoff']['n']}")
    passed, failed = passed + since_n_ok, failed + (not since_n_ok)

    # The whole point: blending in the bad first half must drag the full-sample
    # DSR below the since_cutoff (good-regime-only) DSR.
    dsr_rises_ok = table["since_cutoff"]["dsr_n"] > table["full"]["dsr_n"]
    print_check("restricting to the good-regime era raises DSR vs the blended full sample",
                bool(dsr_rises_ok),
                f"full={table['full']['dsr_n']:.3f}, since_cutoff={table['since_cutoff']['dsr_n']:.3f}")
    passed, failed = passed + dsr_rises_ok, failed + (not dsr_rises_ok)

    # Too few observations in a slice -> None, not a crash or a fabricated stat.
    tiny_cutoff = dates[-1]  # only the very last day qualifies
    tiny_table = dsr_by_era(series, tiny_cutoff, n_trials=16)
    none_ok = tiny_table["since_cutoff"] is None
    print_check("a slice with < 2 observations returns None instead of crashing", bool(none_ok))
    passed, failed = passed + none_ok, failed + (not none_ok)

    print_section_end(passed, failed)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
