"""
paths.py — shared filesystem paths for the H-series research program
(MEDIUM_HORIZON_RESEARCH_PLAN.md). Mirrors src/rl_agent/paths.py's convention.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs/h_series"

DATASET_PATH = ROOT / "data/processed/ml_dataset.parquet"
MEMBERSHIP_PATH = ROOT / "data/processed/top50_universe_membership.parquet"
BOVA11_PATH = ROOT / "data/raw/prices/BOVA11.parquet"

H0_FINDINGS_MD = DOCS_DIR / "H0_FINDINGS.md"
H0_FINDINGS_JSON = DOCS_DIR / "H0_FINDINGS.json"
H1_FINDINGS_MD = DOCS_DIR / "H1_FINDINGS.md"
H1_FINDINGS_JSON = DOCS_DIR / "H1_FINDINGS.json"
H2_FINDINGS_MD = DOCS_DIR / "H2_FINDINGS.md"
H2_FINDINGS_JSON = DOCS_DIR / "H2_FINDINGS.json"
