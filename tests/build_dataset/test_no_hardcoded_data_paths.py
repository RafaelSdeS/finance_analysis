"""
test_no_hardcoded_data_paths.py
================================
Guards the path-consolidation invariant (docs/DATA_COLLECTION_REORGANIZATION_PLAN.md
S15): raw/processed data locations must be imported from config.py/paths.py, never
retyped as a literal -- that drift is what made the 2026-08 data/raw/br/ move touch
10 files instead of 2.

Usage:
    python tests/build_dataset/test_no_hardcoded_data_paths.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
ALLOWED = {
    ROOT / "src/data_collection/config.py",
    ROOT / "src/build_dataset/paths.py",
    ROOT / "src/data_collection/cvm/crosswalk.py",
    ROOT / "src/data_collection/cvm/filing_dates.py",
    # Deliberately standalone, zero-project-import CLI utility (usable on its
    # own with no `src.*` dependency) -- its argparse default has to be a
    # literal, not drift.
    ROOT / "src/build_dataset/cagr_handler.py",
}
PATTERN = re.compile(r'''["']\.{0,2}/?data/(raw|processed)/''')


def find_violations() -> list[str]:
    violations = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path in ALLOWED:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if PATTERN.search(line):
                violations.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")
    return violations


def test_no_hardcoded_data_paths():
    violations = find_violations()
    assert not violations, (
        "hardcoded data/raw or data/processed path(s) found outside "
        "config.py/paths.py -- import the constant instead:\n" + "\n".join(violations)
    )
    print("OK: no hardcoded data/raw or data/processed paths outside config.py/paths.py")


if __name__ == "__main__":
    test_no_hardcoded_data_paths()
