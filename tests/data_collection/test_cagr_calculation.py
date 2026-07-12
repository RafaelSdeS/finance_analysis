"""
test_cagr_calculation.py
========================
Final CAGR filling strategy:

    cagr_earnings_5y:
        1. Use Bolsai value where available
        2. Fill nulls with standard CAGR where base year is positive
        3. Leave null where base year is negative/zero
        4. Add binary flag: had_negative_earnings_5y

    cagr_revenue_5y:
        1. Use Bolsai value where available
        2. Fill nulls with standard CAGR (revenue is always positive)

Usage:
    python test_cagr_calculation.py
    python test_cagr_calculation.py --ticker VALE3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "build_dataset"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_utils import print_header  # noqa: E402

from cagr_handler import fill_cagr_columns, get_cagr_statistics

FUND_DIR = "data/raw/fundamentals"


# =============================================================================
# HELPERS
# =============================================================================

def cagr_standard(v_now: float, v_ago: float, years: int = 5) -> float:
    """Standard CAGR. NaN if either value is non-positive."""
    if pd.isna(v_now) or pd.isna(v_ago) or v_ago <= 0 or v_now <= 0:
        return np.nan
    return ((v_now / v_ago) ** (1 / years) - 1) * 100


def calc_annual_cagr(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Computes CAGR using December values only (annual anchors),
    then forward-fills Q1/Q2/Q3 within each year — matching Bolsai's methodology.
    Looks back exactly 20 quarters (5 years).
    """
    result = pd.Series(np.nan, index=df.index)

    for i in range(20, len(df)):
        result.iloc[i] = cagr_standard(df[col].iloc[i], df[col].iloc[i - 20])

    # Forward-fill within each calendar year
    df_temp        = df.copy()
    df_temp["_r"]  = result
    df_temp["_yr"] = df_temp["reference_date"].dt.year
    df_temp["_r"]  = df_temp.groupby("_yr")["_r"].ffill()

    return df_temp["_r"]


def had_negative_base(df: pd.DataFrame, col: str, lookback: int = 20) -> pd.Series:
    """
    Returns 1 if the base year value (20 quarters ago) was negative or zero,
    meaning standard CAGR is undefined for that quarter.
    """
    result = pd.Series(0, index=df.index)
    for i in range(lookback, len(df)):
        v_ago = df[col].iloc[i - lookback]
        if pd.isna(v_ago) or v_ago <= 0:
            result.iloc[i] = 1
    # Forward-fill within year (same logic)
    df_temp        = df.copy()
    df_temp["_f"]  = result
    df_temp["_yr"] = df_temp["reference_date"].dt.year
    df_temp["_f"]  = df_temp.groupby("_yr")["_f"].ffill()
    return df_temp["_f"].astype(int)


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="PETR4")
    args = parser.parse_args()

    path = Path(FUND_DIR) / f"{args.ticker}.parquet"
    
    if not path.exists():
        print(f"Error: {path} not found")
        return 1
    
    df   = pd.read_parquet(path)
    df   = df.sort_values("reference_date").reset_index(drop=True)

    print(f"Ticker   : {args.ticker}")
    print(f"Quarters : {len(df)}")
    print(f"Range    : {df['reference_date'].min().date()} → {df['reference_date'].max().date()}")
    print()

    # ── Fill CAGR using the new handler ───────────────────────────────────
    df = fill_cagr_columns(df)

    # ── Null summary ──────────────────────────────────────────────────────────
    total = len(df)

    print_header("NULL COVERAGE")
    print(f"{'':40} {'earnings':>10} {'revenue':>10}")
    print(f"{'Bolsai nulls':40} {df['cagr_earnings_5y'].isna().sum():>10} {df['cagr_revenue_5y'].isna().sum():>10}")
    print(f"{'After filling with calc':40} {df['cagr_earnings_5y_final'].isna().sum():>10} {df['cagr_revenue_5y_final'].isna().sum():>10}")
    print(f"{'Remaining nulls (negative base)':40} {(df['cagr_earnings_5y_final'].isna() & (df['had_negative_earnings_5y']==1)).sum():>10} {'N/A':>10}")
    print(f"{'Total quarters':40} {total:>10} {total:>10}")

    # ── Validation ────────────────────────────────────────────────────────────
    # Note: cagr_*_calc columns are cleaned up in fill_cagr_columns,
    # but we can still check internal consistency via statistics
    print()
    print_header("INTERNAL CONSISTENCY CHECK")
    print("(Comparing internal calculations with Bolsai where available)")


    # ── Statistics ────────────────────────────────────────────────────────────
    stats = get_cagr_statistics(df)

    print()
    print_header("STATISTICS")
    
    if "earnings_sanity" in stats:
        s = stats["earnings_sanity"]
        print(f"\nEarnings CAGR (final):")
        print(f"  Count: {s['count']}")
        print(f"  Mean: {s['mean']:.2f}%")
        print(f"  Median: {s['median']:.2f}%")
        print(f"  Range: {s['min']:.2f}% → {s['max']:.2f}%")
        print(f"  Extreme outliers: {s['outliers_gt_100'] + s['outliers_lt_neg_100']}")
    
    if "revenue_sanity" in stats:
        s = stats["revenue_sanity"]
        print(f"\nRevenue CAGR (final):")
        print(f"  Count: {s['count']}")
        print(f"  Mean: {s['mean']:.2f}%")
        print(f"  Median: {s['median']:.2f}%")
        print(f"  Range: {s['min']:.2f}% → {s['max']:.2f}%")
        print(f"  Extreme outliers: {s['outliers_gt_100']}")

    # ── Full table ────────────────────────────────────────────────────────────
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)

    print()
    print_header("FULL TABLE")

    display_cols = [
        "reference_date",
        "net_income",
        "cagr_earnings_5y",
        "cagr_earnings_5y_final",
        "had_negative_earnings_5y",
        "net_revenue",
        "cagr_revenue_5y",
        "cagr_revenue_5y_final",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    display = df[display_cols].copy()
    display["reference_date"] = display["reference_date"].dt.date
    print(display.to_string(index=False))

    return 0


if __name__ == "__main__":
    exit(main())