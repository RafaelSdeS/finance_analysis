"""
paths.py -- shared filesystem paths for the Conviction Model track
(docs/conviction_model/CONVICTION_MODEL_PLAN.md). Mirrors src/h_series/paths.py's
and src/rl_agent/paths.py's convention: each package holds its own copy of the
paths it needs rather than importing a shared paths module cross-package.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs/conviction_model"

DATASET_PATH = ROOT / "data/processed/ml_dataset.parquet"
MEMBERSHIP_PATH = ROOT / "data/processed/top50_universe_membership.parquet"
TOP150_MEMBERSHIP_PATH = ROOT / "data/processed/top150_universe_membership.parquet"
CDI_PATH = ROOT / "data/raw/macro/cdi.parquet"
