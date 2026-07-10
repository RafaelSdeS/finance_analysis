"""
Data-group tests for data_pipeline.py's universe selection + cross-sectional features
(needs the real ml_dataset_training.parquet on disk — run via
`python tests/run_all.py --group data`).

Wraps the module's own self-check functions (previously dead code: not called from
__main__ or any test runner) so they run automatically instead of silently rotting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.data_pipeline import (
    _selfcheck_cross_sectional_features,
    _selfcheck_pit_universe,
    _selfcheck_universe_filter,
)


def main():
    _selfcheck_universe_filter()
    _selfcheck_pit_universe()
    _selfcheck_cross_sectional_features()
    print("\n✓ All data_pipeline universe tests passed")


if __name__ == "__main__":
    main()
