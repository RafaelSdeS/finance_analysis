"""
paths.py — shared filesystem paths for the EIIE agent (docs/eiie_agent/EIIE_AGENT_PLAN.md).

Mirrors src/build_dataset/paths.py's convention: a single source of truth so
data.py, environment.py, experiment.py etc. each import only the paths they
need without importing each other.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT / "data/processed/ml_dataset.parquet"
MEMBERSHIP_PATH = ROOT / "data/processed/top50_universe_membership.parquet"
CDI_PATH = ROOT / "data/raw/macro/cdi.parquet"
BOVA11_PATH = ROOT / "data/raw/prices/BOVA11.parquet"
EXPERIMENTS_DIR = ROOT / "experiments"
