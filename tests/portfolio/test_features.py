"""
test_features.py -- checks the §4.4-E keep-list (src/portfolio/features.py)
against the real dataset schema and the live LOOKAHEAD_TAINTED_COLS list,
not just its own internal assert.

Needs data/processed/ml_dataset.parquet -- data group.
Run: python tests/portfolio/test_features.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.build_dataset.manifest import LOOKAHEAD_TAINTED_COLS  # noqa: E402
from src.build_dataset.paths import OUTPUT_PATH  # noqa: E402
from src.portfolio.features import feature_columns  # noqa: E402
from tests.test_utils import print_check, print_header, print_section_end  # noqa: E402


def main():
    print_header("test_features")
    passed = failed = 0

    cols = feature_columns(include_sector=False)
    count_ok = len(cols) == 120 == len(set(cols))
    print_check("120 unique numeric features (proposal §4.4 count)", count_ok, f"got {len(cols)}")
    passed, failed = passed + count_ok, failed + (not count_ok)

    with_sector = feature_columns(include_sector=True)
    sector_ok = len(with_sector) == 121 and "sector" in with_sector
    print_check("121 with include_sector=True", bool(sector_ok), f"got {len(with_sector)}")
    passed, failed = passed + sector_ok, failed + (not sector_ok)

    no_taint_ok = not (set(with_sector) & set(LOOKAHEAD_TAINTED_COLS))
    print_check("no column overlaps the live manifest.LOOKAHEAD_TAINTED_COLS", bool(no_taint_ok))
    passed, failed = passed + no_taint_ok, failed + (not no_taint_ok)

    real_cols = set(pd.read_parquet(OUTPUT_PATH).columns)
    missing = set(with_sector) - real_cols
    exists_ok = not missing
    print_check("every listed column actually exists in ml_dataset.parquet", bool(exists_ok),
                f"missing: {missing}" if missing else "")
    passed, failed = passed + exists_ok, failed + (not exists_ok)

    print_section_end(passed, failed)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
