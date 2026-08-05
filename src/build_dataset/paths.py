"""
paths.py — shared filesystem paths for the dataset build.

RAW-side paths are owned by data_collection/config.py (Stage 1 decides where
collected data lives) and re-exported here so Stage-2 submodules (loaders,
repair, merge, quality_filters, manifest, ...) keep a single import to reach
for. This module defines only its own PROCESSED-side outputs -- the one thing
genuinely specific to the dataset build.
"""

from pathlib import Path

from src.data_collection.config import (
    PRICES_DIR,
    FUND_DIR as FUNDAMENTALS_DIR,
    COMPANY_INFO_PATH,
    MACRO_DIR,
    DIVIDENDS_DIR,
    CORP_EVENTS_PATH as CORPORATE_EVENTS_PATH,
    BR_RAW_DIR,
    US_PRICES_DIR, US_FUNDAMENTALS_DIR, US_DIVIDENDS_DIR, US_MACRO_DIR,
    US_COMPANY_INFO_PATH,
)
from src.data_collection.cvm.crosswalk import CROSSWALK_PATH as CVM_CROSSWALK_PATH
from src.data_collection.cvm.filing_dates import OUTPUT_PATH as FILING_DATES_PATH

# Re-exported names below are imported, not used in this file -- this module's
# entire job is being the one place Stage-2 submodules import paths from.
__all__ = [
    "PRICES_DIR", "FUNDAMENTALS_DIR", "COMPANY_INFO_PATH", "MACRO_DIR",
    "DIVIDENDS_DIR", "CORPORATE_EVENTS_PATH", "CVM_CROSSWALK_PATH",
    "FILING_DATES_PATH", "CONTINUITY_PATH",
    "US_PRICES_DIR", "US_FUNDAMENTALS_DIR", "US_DIVIDENDS_DIR", "US_MACRO_DIR",
    "US_COMPANY_INFO_PATH",
    "OUTPUT_PATH", "SPLIT_CONFIG_PATH", "SCALER_DIR",
    "TOP50_UNIVERSE_PATH", "TOP50_MEMBERSHIP_PATH",
    "US_OUTPUT_PATH", "US_SPLIT_CONFIG_PATH", "US_SCALER_DIR",
]

ROOT = Path(__file__).resolve().parents[2]

# Hand-maintained (ticker renames/mergers); no producer in data_collection to
# import from, so this one stays a locally-built path, just off the shared root.
CONTINUITY_PATH = BR_RAW_DIR / "reference/ticker_continuity.json"

# --- This module's own concern: build output ---
OUTPUT_PATH = ROOT / "data/processed/ml_dataset.parquet"
SPLIT_CONFIG_PATH = ROOT / "data/processed/split_config.json"
SCALER_DIR = ROOT / "data/processed/scalers"
TOP50_UNIVERSE_PATH = ROOT / "data/processed/ml_dataset_top50_universe.parquet"
TOP50_MEMBERSHIP_PATH = ROOT / "data/processed/top50_universe_membership.parquet"

# US equities (docs/US_DATASET_BUILD_PLAN.md) -- separate raw tree, separate output.
US_OUTPUT_PATH = ROOT / "data/processed/us_ml_dataset.parquet"
US_SPLIT_CONFIG_PATH = ROOT / "data/processed/us_split_config.json"
US_SCALER_DIR = ROOT / "data/processed/us_scalers"
