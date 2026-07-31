"""
paths.py — shared filesystem paths for the dataset build.

Single source of truth so build_ml_dataset.py's submodules (loaders,
repair, merge, quality_filters, manifest, ...) can each import only the
paths they need without importing each other or the orchestrator.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PRICES_DIR = ROOT / "data/raw/prices"
FUNDAMENTALS_DIR = ROOT / "data/raw/fundamentals"
COMPANY_INFO_PATH = ROOT / "data/raw/company_info/company_info.parquet"
CVM_CROSSWALK_PATH = ROOT / "data/raw/cvm/fca_crosswalk.parquet"
MACRO_DIR = ROOT / "data/raw/macro"
DIVIDENDS_DIR = ROOT / "data/raw/dividends"
CORPORATE_EVENTS_PATH = ROOT / "data/raw/corporate_events/corporate_events.parquet"
FILING_DATES_PATH = ROOT / "data/raw/filing_dates/filing_dates.parquet"
CONTINUITY_PATH = ROOT / "data/raw/reference/ticker_continuity.json"
OUTPUT_PATH = ROOT / "data/processed/ml_dataset.parquet"
SPLIT_CONFIG_PATH = ROOT / "data/processed/split_config.json"
SCALER_DIR = ROOT / "data/processed/scalers"
TOP50_UNIVERSE_PATH = ROOT / "data/processed/ml_dataset_top50_universe.parquet"
TOP50_MEMBERSHIP_PATH = ROOT / "data/processed/top50_universe_membership.parquet"

# US equities (docs/US_DATASET_BUILD_PLAN.md) -- separate raw tree, separate output.
US_PRICES_DIR = ROOT / "data/raw/us/prices"
US_FUNDAMENTALS_DIR = ROOT / "data/raw/us/fundamentals"
US_DIVIDENDS_DIR = ROOT / "data/raw/us/dividends"
US_MACRO_DIR = ROOT / "data/raw/us/macro"
US_COMPANY_INFO_PATH = ROOT / "data/raw/us/sec/company_info.parquet"
US_OUTPUT_PATH = ROOT / "data/processed/us_ml_dataset.parquet"
US_SPLIT_CONFIG_PATH = ROOT / "data/processed/us_split_config.json"
US_SCALER_DIR = ROOT / "data/processed/us_scalers"
