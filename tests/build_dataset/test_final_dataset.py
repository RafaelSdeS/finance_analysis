# inspect_final_dataset.py
#
# Visualiza e inspeciona o dataset final consolidado.
#
# Uso:
#   python test_final_dataset.py
#
# Ou:
#   python test_final_dataset.py --file data/processed/ml_dataset.parquet

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_FILE = Path(__file__).resolve().parents[2] / "data/processed/ml_dataset.parquet"


def print_separator():
    print("-" * 80)


def check_stale_prices(df, price_col="close", run_len=5):
    """Flag runs of >= run_len identical closes while volume > 0."""
    findings = []
    for t, g in df.sort_values("date").groupby("ticker"):
        same = (g[price_col].diff() == 0) & (g["volume"] > 0)
        run = same.groupby((~same).cumsum()).cumsum()
        hits = g[run >= run_len]
        for _, row in hits.iterrows():
            findings.append({"ticker": t, "date": row["date"], "value": row[price_col]})
    return pd.DataFrame(findings)


def check_outliers_zscore(df, feature_cols, threshold=8.0):
    """Robust (median/MAD) z-score outlier flagging, computed within each
    ticker's own history — a global z-score would flag every large-cap as an
    outlier on absolute-scale columns (market_cap, volume, ...) just for
    being large, since company size varies by orders of magnitude."""
    findings = []
    by_ticker = df.groupby("ticker")
    for col in feature_cols:
        s = df[col]
        med = by_ticker[col].transform("median")
        mad = (s - med).abs().groupby(df["ticker"]).transform("median")
        z = 0.6745 * (s - med) / mad.replace(0, float("nan"))
        mask = z.abs() > threshold
        if mask.any():
            sub = df.loc[mask, ["ticker", "date"]].copy()
            sub["column"] = col
            sub["value"] = s[mask].values
            findings.append(sub)
    if not findings:
        return pd.DataFrame(columns=["ticker", "date", "column", "value"])
    return pd.concat(findings, ignore_index=True)


def validate(df):
    """Golden gate: collect failures, exit(1) if any. Inspector runs first."""

    print("\n")
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)

    checks = []

    # No lookahead: every merged fundamental is dated on/before its trade_date
    has_dates = df["reference_date"].notna()
    no_lookahead = (df.loc[has_dates, "reference_date"] <= df.loc[has_dates, "trade_date"]).all()
    checks.append(("no lookahead (reference_date <= trade_date)", bool(no_lookahead)))

    # No duplicate (ticker, trade_date) rows
    dupes = df.duplicated(subset=["ticker", "trade_date"]).sum()
    checks.append((f"no duplicate (ticker, trade_date) [{dupes} found]", dupes == 0))

    # CAGR final columns present (proves fill_cagr_columns ran)
    cagr_ok = {"cagr_earnings_5y_final", "cagr_revenue_5y_final"}.issubset(df.columns)
    checks.append(("CAGR _final columns present", cagr_ok))

    # Critical columns have no NaN
    for col in ("close", "volume"):
        checks.append((f"no NaN in {col}", col in df.columns and df[col].notna().all()))

    # No inf/-inf leaking through (division-by-zero in growth rates/ratios
    # must be cleaned to NaN by clean_dataset(), never a literal inf)
    numeric_cols = df.select_dtypes(include="number").columns
    n_inf = np.isinf(df[numeric_cols]).sum().sum()
    checks.append((f"no inf values in numeric columns [{n_inf} found]", n_inf == 0))

    # Macro merged and not entirely null
    for col in ("selic", "cdi", "ipca"):
        present = col in df.columns and df[col].notna().any()
        checks.append((f"macro {col} merged", present))

    # Stage 2 keeps sparse tickers (MIN_PRICE_ROWS=10 in build_ml_dataset.py);
    # the 252-row (1 trading year) full-history bar is a Stage 3 concern,
    # enforced by MIN_ROWS_PER_TICKER in src/agent/data_pipeline.py.
    row_counts = df.groupby("ticker").size()
    min_rows = row_counts.min()
    n_short = (row_counts < 252).sum()
    checks.append((f"all tickers >= 10 rows [min {min_rows}] "
                    f"({n_short} tickers < 252 rows, dropped later by data_pipeline.py)",
                   min_rows >= 10))

    # Valuation staleness regression guard: P/L must be re-anchored to the
    # daily close (recompute_valuation_daily), not frozen at the filing price
    if {"pl", "reference_date"}.issubset(df.columns):
        grp = df[df["pl"].notna()].groupby(["ticker", "reference_date"])["pl"]
        sizes, nun = grp.size(), grp.nunique()
        eligible = (sizes >= 5).sum()
        frozen = ((sizes >= 5) & (nun == 1)).sum()
        checks.append((f"P/L varies daily within quarter [{frozen}/{eligible} frozen]",
                       eligible > 0 and frozen / eligible < 0.01))
    checks.append(("stale close_price column dropped", "close_price" not in df.columns))
    checks.append(("has_fundamentals flag present", "has_fundamentals" in df.columns))

    failed = 0
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failed += not ok

    print()
    if failed:
        print(f"VALIDATION FAILED: {failed} check(s)")
        sys.exit(1)
    print("VALIDATION PASSED")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        type=str,
        default=str(DEFAULT_FILE),
        help="Path do parquet final"
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Número de linhas de exemplo"
    )

    args = parser.parse_args()

    file_path = Path(args.file)

    print("=" * 80)
    print("FINAL DATASET INSPECTION")
    print("=" * 80)

    if not file_path.exists():
        print(f"\nERROR: arquivo não encontrado:\n{file_path}")
        return

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    df = pd.read_parquet(file_path)

    print(f"File     : {file_path}")
    print(f"Rows     : {len(df):,}")
    print(f"Columns  : {len(df.columns)}")
    print(
        f"Memory   : "
        f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )

    # -------------------------------------------------------------------------
    # COLUMN LIST
    # -------------------------------------------------------------------------

    print("\n")
    print_separator()
    print("COLUMNS")
    print_separator()

    for col in df.columns:

        dtype = str(df[col].dtype)

        null_pct = (df[col].isna().mean()) * 100

        unique = df[col].nunique(dropna=True)

        print(
            f"{col:<35} "
            f"{dtype:<15} "
            f"nulls={null_pct:6.2f}% "
            f"unique={unique}"
        )

    # -------------------------------------------------------------------------
    # SAMPLE ROWS
    # -------------------------------------------------------------------------

    print("\n")
    print_separator()
    print("SAMPLE ROWS")
    print_separator()

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.max_colwidth",
        40
    ):
        print(df.head(args.rows))

    # -------------------------------------------------------------------------
    # BASIC STATISTICS
    # -------------------------------------------------------------------------

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if numeric_cols:

        print("\n")
        print_separator()
        print("NUMERIC STATISTICS")
        print_separator()

        stats = df[numeric_cols].describe().T

        with pd.option_context(
            "display.max_rows",
            200,
            "display.max_columns",
            None,
            "display.width",
            200
        ):
            print(stats)

    # -------------------------------------------------------------------------
    # NULL ANALYSIS
    # -------------------------------------------------------------------------

    print("\n")
    print_separator()
    print("NULL ANALYSIS")
    print_separator()

    nulls = df.isnull().sum()

    nulls = nulls[nulls > 0].sort_values(ascending=False)

    if len(nulls) == 0:
        print("No null values.")
    else:

        for col, count in nulls.items():

            pct = 100 * count / len(df)

            print(f"{col:<35} {count:>10} ({pct:6.2f}%)")

    # -------------------------------------------------------------------------
    # DUPLICATE CHECK
    # -------------------------------------------------------------------------

    print("\n")
    print_separator()
    print("DUPLICATE CHECK")
    print_separator()

    total_duplicates = df.duplicated().sum()

    print(f"Duplicate rows: {total_duplicates}")

    if "ticker" in df.columns:
        print(f"Unique tickers: {df['ticker'].nunique()}")

    # -------------------------------------------------------------------------
    # ANOMALY REPORT (informational, does not affect exit code)
    # -------------------------------------------------------------------------

    print("\n")
    print_separator()
    print("ANOMALY REPORT (informational, does not affect exit code)")
    print_separator()

    df_dated = df.rename(columns={"trade_date": "date"})

    stale = check_stale_prices(df_dated)
    print(f"Stale price runs (>=5 identical closes, volume>0): {len(stale)}")
    if len(stale):
        print(stale.head(20).to_string(index=False))

    # Exclude raw macro passthrough columns: identical across all tickers on a
    # given date by construction, so they flood this report with repeats of
    # the same macro regime shift rather than per-observation data errors.
    macro_cols = {"selic", "cdi", "ipca", "selic_trend_20d"}
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in macro_cols]
    outliers = check_outliers_zscore(df_dated, numeric_cols)
    print(f"\nOutliers (robust z-score > 8): {len(outliers)}")
    if len(outliers):
        print(outliers["column"].value_counts().head(10).to_string())
        print(outliers.head(20).to_string(index=False))

    # -------------------------------------------------------------------------
    # DATE RANGE
    # -------------------------------------------------------------------------

    possible_date_cols = [
        "trade_date",
        "date",
        "reference_date",
        "trading_date",
        "datetime"
    ]

    for col in possible_date_cols:

        if col in df.columns:

            print("\n")
            print_separator()
            print(f"DATE RANGE ({col})")
            print_separator()

            print(f"Min date: {df[col].min()}")
            print(f"Max date: {df[col].max()}")

            break

    # -------------------------------------------------------------------------
    # TICKERS
    # -------------------------------------------------------------------------

    if "ticker" in df.columns:

        print("\n")
        print_separator()
        print("TICKERS")
        print_separator()

        print(df["ticker"].value_counts())

    # -------------------------------------------------------------------------
    # SECTORS
    # -------------------------------------------------------------------------

    if "sector" in df.columns:

        print("\n")
        print_separator()
        print("SECTORS")
        print_separator()

        print(df["sector"].value_counts())

    # -------------------------------------------------------------------------
    # DATAFRAME INFO
    # -------------------------------------------------------------------------

    print("\n")
    print_separator()
    print("DATAFRAME INFO")
    print_separator()

    print(df.info())

    print("\n")
    print("=" * 80)
    print("INSPECTION FINISHED")
    print("=" * 80)

    validate(df)


if __name__ == "__main__":
    main()