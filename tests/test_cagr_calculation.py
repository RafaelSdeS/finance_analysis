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
import numpy as np
import pandas as pd

FUND_DIR = "../data/raw/fundamentals"


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

    path = f"{FUND_DIR}/{args.ticker}.parquet"
    df   = pd.read_parquet(path)
    df   = df.sort_values("reference_date").reset_index(drop=True)

    print(f"Ticker   : {args.ticker}")
    print(f"Quarters : {len(df)}")
    print(f"Range    : {df['reference_date'].min().date()} → {df['reference_date'].max().date()}")
    print()

    # ── Calculate from fundamentals ───────────────────────────────────────────
    df["cagr_earn_calc"] = calc_annual_cagr(df, "net_income")
    df["cagr_rev_calc"]  = calc_annual_cagr(df, "net_revenue")

    # ── Negative base flag ────────────────────────────────────────────────────
    df["had_negative_earnings_5y"] = had_negative_base(df, "net_income")

    # ── Final columns: Bolsai first, fill with calc where null ───────────────
    df["cagr_earnings_final"] = df["cagr_earnings_5y"].combine_first(df["cagr_earn_calc"])
    df["cagr_revenue_final"]  = df["cagr_revenue_5y"].combine_first(df["cagr_rev_calc"])

    # ── Null summary ──────────────────────────────────────────────────────────
    total = len(df)

    print("=" * 70)
    print("NULL COVERAGE")
    print("=" * 70)
    print(f"{'':40} {'earnings':>10} {'revenue':>10}")
    print(f"{'Bolsai nulls':40} {df['cagr_earnings_5y'].isna().sum():>10} {df['cagr_revenue_5y'].isna().sum():>10}")
    print(f"{'After filling with calc':40} {df['cagr_earnings_final'].isna().sum():>10} {df['cagr_revenue_final'].isna().sum():>10}")
    print(f"{'Remaining nulls (negative base)':40} {(df['cagr_earnings_final'].isna() & (df['had_negative_earnings_5y']==1)).sum():>10} {'N/A':>10}")
    print(f"{'Total quarters':40} {total:>10} {total:>10}")

    # ── Validation ────────────────────────────────────────────────────────────
    both = df[df["cagr_earnings_5y"].notna() & df["cagr_earn_calc"].notna()].copy()
    if not both.empty:
        both["diff_earn"] = (both["cagr_earnings_5y"] - both["cagr_earn_calc"]).abs()
        both["diff_rev"]  = (both["cagr_revenue_5y"]  - both["cagr_rev_calc"]).abs()
        print()
        print("=" * 70)
        print("VALIDATION (where both Bolsai and calc exist)")
        print("=" * 70)
        print(f"Mean absolute diff (earnings): {both['diff_earn'].mean():.4f}%")
        print(f"Mean absolute diff (revenue) : {both['diff_rev'].mean():.4f}%")

    # ── Full table ────────────────────────────────────────────────────────────
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)

    print()
    print("=" * 140)
    print("FULL TABLE")
    print("=" * 140)

    display = df[[
        "reference_date",
        "net_income",
        "cagr_earnings_5y",
        "cagr_earn_calc",
        "cagr_earnings_final",
        "had_negative_earnings_5y",
        "net_revenue",
        "cagr_revenue_5y",
        "cagr_rev_calc",
        "cagr_revenue_final",
    ]].copy()
    display["reference_date"] = display["reference_date"].dt.date
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()