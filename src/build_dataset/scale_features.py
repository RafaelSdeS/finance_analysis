"""
scale_features.py
==================

Fits a leak-safe feature scaler on the train split only (per split_config.json,
built by build_ml_dataset.py) and persists it for reuse at training/inference
time.

Ratio-style fundamentals (P/E, P/B, margins, leverage, growth rates - unitless,
fat-tailed) get RobustScaler (median/IQR - robust to the extreme outliers this
dataset already has in these columns; RobustScaler ignores NaN when fitting
and leaves NaN as NaN on transform, matching the project's no-imputation rule).
Everything else (already-bounded percentiles/z-scores/binary flags, returns,
prices, identifiers) passes through unscaled via ColumnTransformer's
remainder="passthrough" - scaling an already-[0,1] or already-z-scored column
doesn't help and just makes the audit harder to read.

Output:
    data/processed/scalers/feature_scaler.joblib
    data/processed/scalers/scaler_metadata.json

Usage:
    python -m src.build_dataset.scale_features
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from .paths import OUTPUT_PATH, SPLIT_CONFIG_PATH

ROOT = Path(__file__).resolve().parents[2]
SCALER_DIR = ROOT / "data/processed/scalers"
SCALER_PATH = SCALER_DIR / "feature_scaler.joblib"
METADATA_PATH = SCALER_DIR / "scaler_metadata.json"

# Unitless ratios/multiples/growth rates with fat tails -> RobustScaler.
# Everything else (percentiles, z-scores, binary flags, returns, prices,
# volumes, identifiers, macro rates) is left to remainder="passthrough".
RATIO_COLUMNS = [
    "pl", "pvp", "ev_ebitda", "ev_ebit", "p_ebitda", "p_ebit", "p_sr", "p_assets",
    "lpa", "vpa",
    "gross_margin", "net_margin", "ebitda_margin", "ebit_margin",
    "roe", "roa", "roic", "ebit_over_assets", "asset_turnover",
    "current_ratio", "cash_ratio", "net_debt_to_assets", "working_capital_ratio",
    "debt_equity", "net_debt_equity", "net_debt_ebitda", "net_debt_ebit",
    "cagr_revenue_5y", "cagr_earnings_5y", "cagr_revenue_5y_final", "cagr_earnings_5y_final",
    "book_to_market", "earnings_yield",
    "revenue_growth_yoy", "earnings_growth_yoy", "ebitda_growth_yoy",
    "total_assets_growth_yoy", "total_debt_growth_yoy",
    "gross_margin_qoq", "net_margin_qoq", "roe_qoq", "debt_equity_qoq", "current_ratio_qoq",
    "peg_ratio", "pvp_to_roe_ratio", "earnings_yield_vs_selic",
    "payout_ratio", "dividend_coverage_ratio", "revenue_per_earning",
    "revenue_vs_earnings_growth_delta",
    "roe_trend_4q", "margin_trend_4q", "debt_trend_4q", "roa_trend_4q",
]


def build_scaler(ratio_columns=RATIO_COLUMNS) -> ColumnTransformer:
    ct = ColumnTransformer(
        transformers=[("robust", RobustScaler(), ratio_columns)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return ct.set_output(transform="pandas")


def transform_features(ct: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    """Apply a fitted scaler and restore the original column order.

    ColumnTransformer's pandas output groups columns by transformer (all
    "robust" columns first, then all passthrough columns) rather than
    preserving input order - reindex back so downstream code can't silently
    read the wrong column by position.
    """
    scaled = ct.transform(df)
    return scaled[df.columns]


def fit_scaler_on_train_split(dataset: pd.DataFrame) -> ColumnTransformer:
    """Fit RATIO_COLUMNS' scaler on rows at/before split_config.json's train_end."""
    split_config = json.loads(SPLIT_CONFIG_PATH.read_text())
    train_end = pd.Timestamp(split_config["train_end"])
    train = dataset[dataset["trade_date"] <= train_end]

    ct = build_scaler()
    ct.fit(train)
    return ct


def write_scaler_metadata(ct: ColumnTransformer) -> None:
    """Audit file, derived from the fitted transformer itself (not hand-copied)
    so it can't drift from what was actually fit.
    """
    robust = ct.named_transformers_["robust"]
    metadata = {
        "method": "RobustScaler",
        "scaled_columns": list(RATIO_COLUMNS),
        "center": dict(zip(RATIO_COLUMNS, robust.center_.tolist())),
        "scale": dict(zip(RATIO_COLUMNS, robust.scale_.tolist())),
        "passthrough_columns": [c for c in ct.feature_names_in_ if c not in RATIO_COLUMNS],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=1))
    print(f"Scaler metadata saved to: {METADATA_PATH}")


def main():
    SCALER_DIR.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(OUTPUT_PATH)

    ct = fit_scaler_on_train_split(dataset)
    joblib.dump(ct, SCALER_PATH)
    print(f"Scaler saved to: {SCALER_PATH}")

    write_scaler_metadata(ct)


if __name__ == "__main__":
    main()
