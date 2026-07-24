"""
scale_features.py
==================

Fits a leak-safe feature scaler and persists it for reuse at training/inference
time. The fit boundary is never a hardcoded date: it's injected as a FitWindow
(manifest.py, resolved from split_config.json via iter_fit_windows) so this
module works unchanged whether the active evaluation methodology is a single
fixed split, rolling windows, or multiple folds (docs/PER_TICKER_SCALING_PLAN.md
§3.3/§3.5) -- a config that resolves to more than one window fits one artifact
per window instead of one global fit.

Ratio-style fundamentals (P/E, P/B, margins, leverage, growth rates - unitless,
fat-tailed) get RobustScaler (median/IQR - robust to the extreme outliers this
dataset already has in these columns; RobustScaler ignores NaN when fitting
and leaves NaN as NaN on transform, matching the project's no-imputation rule).
Everything else (already-bounded percentiles/z-scores/binary flags, returns,
prices, identifiers) passes through unscaled via ColumnTransformer's
remainder="passthrough" - scaling an already-[0,1] or already-z-scored column
doesn't help and just makes the audit harder to read.

Output (single-window config -- today's default):
    data/processed/scalers/feature_scaler.joblib
    data/processed/scalers/scaler_metadata.json
Output (multi-window config): one subdirectory per fold_id under scalers/,
    each holding its own feature_scaler.joblib + scaler_metadata.json.

Usage:
    python -m src.build_dataset.scale_features
"""

import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from .manifest import LOOKAHEAD_TAINTED_COLS, FitWindow, iter_fit_windows
from .paths import OUTPUT_PATH, SCALER_DIR, SPLIT_CONFIG_PATH

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
    "volatility_ratio_20_60", "volume_ratio_20d", "amihud_illiquidity", "turnover_ratio",
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


def fit_scaler(dataset: pd.DataFrame, window: FitWindow, ratio_columns=RATIO_COLUMNS) -> ColumnTransformer:
    """Fit RATIO_COLUMNS' scaler on rows inside `window` (fit_start < trade_date
    <= fit_end; fit_start=None means "from the beginning of history").

    Pure: no file I/O, no read of split_config.json -- the boundary is always
    injected via `window` (see iter_fit_windows in manifest.py), so this works
    unchanged under any evaluation methodology (single split, rolling,
    expanding, multi-fold): re-cutting the split never requires touching this
    function, only how the caller resolves `window`.
    """
    in_window = dataset["trade_date"] <= window.fit_end
    if window.fit_start is not None:
        in_window &= dataset["trade_date"] > window.fit_start
    train = dataset[in_window]

    ct = build_scaler(ratio_columns)
    ct.fit(train)
    return ct


def fit_scaler_on_train_split(dataset: pd.DataFrame) -> ColumnTransformer:
    """Back-compat convenience: resolves the single window from
    split_config.json (today's fixed-split config) and fits on it. Code that
    needs to handle a multi-window split config should call
    iter_fit_windows() + fit_scaler() directly instead.
    """
    split_config = json.loads(SPLIT_CONFIG_PATH.read_text())
    window = iter_fit_windows(split_config)[0]
    return fit_scaler(dataset, window)


def _window_to_json(window: FitWindow) -> dict:
    return {
        "fold_id": window.fold_id,
        "fit_start": str(window.fit_start.date()) if window.fit_start is not None else None,
        "fit_end": str(window.fit_end.date()),
    }


def write_scaler_metadata(ct: ColumnTransformer, window: FitWindow, metadata_path=METADATA_PATH) -> None:
    """Audit file, derived from the fitted transformer itself (not hand-copied)
    so it can't drift from what was actually fit. Records the FitWindow that
    produced it so a params/split mismatch is detectable at load time instead
    of silently transforming with stale boundaries.
    """
    robust = ct.named_transformers_["robust"]
    metadata = {
        "method": "RobustScaler",
        "fit_window": _window_to_json(window),
        "scaled_columns": list(RATIO_COLUMNS),
        "center": dict(zip(RATIO_COLUMNS, robust.center_.tolist())),
        "scale": dict(zip(RATIO_COLUMNS, robust.scale_.tolist())),
        "passthrough_columns": [c for c in ct.feature_names_in_ if c not in RATIO_COLUMNS],
        # Passed through by the scaler (not scaled, not dropped) but must never
        # be fed to a model as a feature -- see manifest.LOOKAHEAD_TAINTED_COLS.
        "lookahead_tainted_columns": [c for c in LOOKAHEAD_TAINTED_COLS if c in ct.feature_names_in_],
    }
    metadata_path.write_text(json.dumps(metadata, indent=1))
    print(f"Scaler metadata saved to: {metadata_path}")


def main():
    dataset = pd.read_parquet(OUTPUT_PATH)
    split_config = json.loads(SPLIT_CONFIG_PATH.read_text())
    windows = iter_fit_windows(split_config)

    # Single window (today's default): flat scalers/ layout, unchanged from
    # before. Multiple windows: one subdirectory per fold_id (§3.5 layout) --
    # no implicit "latest", callers must name the window they want.
    for window in windows:
        scaler_dir = SCALER_DIR if len(windows) == 1 else SCALER_DIR / window.fold_id
        scaler_dir.mkdir(parents=True, exist_ok=True)

        ct = fit_scaler(dataset, window)
        scaler_path = scaler_dir / "feature_scaler.joblib"
        joblib.dump(ct, scaler_path)
        print(f"Scaler saved to: {scaler_path}")

        write_scaler_metadata(ct, window, scaler_dir / "scaler_metadata.json")


if __name__ == "__main__":
    main()
