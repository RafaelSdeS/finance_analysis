"""
cagr_handler.py
===============

Handles CAGR (Compound Annual Growth Rate) calculation and filling strategy:

1. Where Bolsai CAGR values exist → use them directly
2. Where Bolsai values are null → calculate from fundamentals (if valid)
3. Where base year is negative/zero → leave null and flag

This ensures maximum data coverage while maintaining data integrity.

Usage (as a module):
    from cagr_handler import fill_cagr_columns
    
    fundamentals = pd.read_parquet("fundamentals/PETR4.parquet")
    fundamentals = fill_cagr_columns(fundamentals)

Usage (as a script):
    python cagr_handler.py --ticker PETR4
    python cagr_handler.py --ticker PETR4 --output report.txt
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================================
# CAGR CALCULATION FUNCTIONS
# =============================================================================

def cagr_standard(v_now: float, v_ago: float, years: int = 5) -> float:
    """
    Standard CAGR formula: (V_now / V_ago)^(1/years) - 1
    
    Returns NaN if:
        - Either value is missing
        - Base year (V_ago) is non-positive (undefined mathematically)
        - Current value (V_now) is non-positive (negative growth rate)
    """
    if pd.isna(v_now) or pd.isna(v_ago) or v_ago <= 0 or v_now <= 0:
        return np.nan
    return ((v_now / v_ago) ** (1 / years) - 1) * 100


def calc_annual_cagr(df: pd.DataFrame, col: str, lookback: int = 20) -> pd.Series:
    """
    Calculate CAGR using December values only (annual anchors),
    then forward-fill Q1/Q2/Q3 within each year.

    This matches Bolsai's reported methodology where CAGR is anchored
    to fiscal year ends.

    Parameters:
        df: DataFrame with 'reference_date' and a data column
        col: Column name containing values to calculate CAGR from
        lookback: Number of quarters to look back (default 20 = 5 years)

    Returns:
        Series with CAGR values and annual forward-fill applied
    """
    values = df[col].to_numpy()
    result = np.full(len(values), np.nan, dtype=np.float64)

    # ponytail: vectorized CAGR calculation using numpy shift
    # Avoid loop over .iloc[], use numpy slicing instead
    v_now = values[lookback:]
    v_ago = values[:-lookback]

    # Apply CAGR formula element-wise: only where both values are positive.
    # lookback is in quarters; CAGR exponent is per-year → years = lookback / 4
    years = lookback / 4
    valid = (v_now > 0) & (v_ago > 0) & (~np.isnan(v_now)) & (~np.isnan(v_ago))
    result[lookback:][valid] = ((v_now[valid] / v_ago[valid]) ** (1 / years) - 1) * 100

    # Forward-fill within each calendar year
    df_temp = df.copy()
    df_temp["_cagr"] = result
    df_temp["_year"] = df_temp["reference_date"].dt.year
    df_temp["_cagr"] = df_temp.groupby("_year")["_cagr"].ffill()

    return df_temp["_cagr"]


def had_negative_base(df: pd.DataFrame, col: str, lookback: int = 20) -> pd.Series:
    """
    Binary flag indicating whether base year (lookback quarters ago)
    had negative or zero value, making standard CAGR undefined.

    This is important for earnings: negative earnings 5 years ago
    means we can't compute a meaningful CAGR.

    Returns:
        Series with 1 where base was negative/zero, 0 otherwise
    """
    values = df[col].to_numpy()
    result = np.zeros(len(values), dtype=np.int32)

    # ponytail: vectorized flag check using numpy slicing
    v_ago = values[:-lookback]
    result[lookback:] = (v_ago <= 0) | np.isnan(v_ago)

    # Forward-fill within year (same logic as CAGR)
    df_temp = df.copy()
    df_temp["_flag"] = result
    df_temp["_year"] = df_temp["reference_date"].dt.year
    df_temp["_flag"] = df_temp.groupby("_year")["_flag"].ffill()

    return df_temp["_flag"].astype(int)


def fill_cagr_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main CAGR filling function.
    
    Fills missing CAGR values in df with calculated values where valid,
    and adds a flag for cases where calculation is impossible.
    
    Input columns required:
        - cagr_earnings_5y (from Bolsai, may be null)
        - cagr_revenue_5y (from Bolsai, may be null)
        - net_income (from Bolsai fundamentals)
        - net_revenue (from Bolsai fundamentals)
        - reference_date (datetime)
    
    Output columns added:
        - cagr_earnings_5y_final: Bolsai value or calculated, or NaN
        - cagr_revenue_5y_final: Bolsai value or calculated
        - had_negative_earnings_5y: Binary flag (1 if negative base, 0 otherwise)
    
    Returns:
        DataFrame with filled CAGR columns added
    """
    df = df.copy()
    df = df.sort_values("reference_date").reset_index(drop=True)
    
    # Calculate CAGR from fundamentals
    if "net_income" in df.columns:
        df["cagr_earnings_calc"] = calc_annual_cagr(df, "net_income")
        df["had_negative_earnings"] = had_negative_base(df, "net_income")
    else:
        df["cagr_earnings_calc"] = np.nan
        df["had_negative_earnings"] = 0
    
    if "net_revenue" in df.columns:
        df["cagr_revenue_calc"] = calc_annual_cagr(df, "net_revenue")
    else:
        df["cagr_revenue_calc"] = np.nan
    
    # Use combine_first: Bolsai first, then fill with calculated
    if "cagr_earnings_5y" in df.columns:
        df["cagr_earnings_5y_final"] = df["cagr_earnings_5y"].combine_first(
            df["cagr_earnings_calc"]
        )
    else:
        df["cagr_earnings_5y_final"] = df["cagr_earnings_calc"]
    
    if "cagr_revenue_5y" in df.columns:
        df["cagr_revenue_5y_final"] = df["cagr_revenue_5y"].combine_first(
            df["cagr_revenue_calc"]
        )
    else:
        df["cagr_revenue_5y_final"] = df["cagr_revenue_calc"]
    
    # Rename the flag for clarity
    df = df.rename(columns={"had_negative_earnings": "had_negative_earnings_5y"})
    
    # Clean up temporary columns
    temp_cols = [c for c in df.columns if c.endswith("_calc")]
    df = df.drop(columns=temp_cols)
    
    return df


# =============================================================================
# STATISTICS & REPORTING
# =============================================================================

def get_cagr_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate statistics on CAGR filling results.
    
    Returns dict with:
        - null_coverage: Coverage improvement from filling
        - comparison_stats: Where both Bolsai and calc exist
        - sanity_checks: Data quality indicators
    """
    stats = {}
    
    # Coverage improvement
    if "cagr_earnings_5y" in df.columns and "cagr_earnings_5y_final" in df.columns:
        before = df["cagr_earnings_5y"].notna().sum()
        after = df["cagr_earnings_5y_final"].notna().sum()
        stats["earnings_null_reduction"] = {
            "before": df["cagr_earnings_5y"].isna().sum(),
            "after": df["cagr_earnings_5y_final"].isna().sum(),
            "filled": after - before,
        }
    
    if "cagr_revenue_5y" in df.columns and "cagr_revenue_5y_final" in df.columns:
        before = df["cagr_revenue_5y"].notna().sum()
        after = df["cagr_revenue_5y_final"].notna().sum()
        stats["revenue_null_reduction"] = {
            "before": df["cagr_revenue_5y"].isna().sum(),
            "after": df["cagr_revenue_5y_final"].isna().sum(),
            "filled": after - before,
        }
    
    # Comparison where both exist
    if "cagr_earnings_5y" in df.columns and "cagr_earnings_calc" in df.columns:
        both = df[
            df["cagr_earnings_5y"].notna() & df["cagr_earnings_calc"].notna()
        ].copy()
        if not both.empty:
            both["diff"] = (both["cagr_earnings_5y"] - both["cagr_earnings_calc"]).abs()
            stats["earnings_comparison"] = {
                "count": len(both),
                "mean_diff": both["diff"].mean(),
                "max_diff": both["diff"].max(),
                "std_diff": both["diff"].std(),
            }
    
    # Sanity checks
    if "cagr_earnings_5y_final" in df.columns:
        final = df["cagr_earnings_5y_final"].dropna()
        if len(final) > 0:
            stats["earnings_sanity"] = {
                "count": len(final),
                "mean": final.mean(),
                "median": final.median(),
                "min": final.min(),
                "max": final.max(),
                "outliers_gt_100": (final > 100).sum(),
                "outliers_lt_neg_100": (final < -100).sum(),
            }
    
    if "cagr_revenue_5y_final" in df.columns:
        final = df["cagr_revenue_5y_final"].dropna()
        if len(final) > 0:
            stats["revenue_sanity"] = {
                "count": len(final),
                "mean": final.mean(),
                "median": final.median(),
                "min": final.min(),
                "max": final.max(),
                "outliers_gt_100": (final > 100).sum(),
            }
    
    return stats


# =============================================================================
# CLI SCRIPT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calculate and validate CAGR values from fundamentals."
    )
    parser.add_argument("--ticker", default="PETR4", help="Ticker symbol")
    parser.add_argument("--fund-dir", default="../data/raw/fundamentals",
                        help="Fundamentals data directory")
    parser.add_argument("--output", help="Output report file (optional)")
    args = parser.parse_args()
    
    fund_path = Path(args.fund_dir) / f"{args.ticker}.parquet"
    
    if not fund_path.exists():
        print(f"Error: {fund_path} not found")
        return 1
    
    # Load and process
    df = pd.read_parquet(fund_path)
    df = df.sort_values("reference_date").reset_index(drop=True)
    
    print(f"\nTicker: {args.ticker}")
    print(f"Quarters: {len(df)}")
    print(f"Range: {df['reference_date'].min().date()} → {df['reference_date'].max().date()}")
    print()
    
    # Fill CAGR
    df = fill_cagr_columns(df)
    
    # Get statistics
    stats = get_cagr_statistics(df)
    
    # Print coverage
    print("=" * 70)
    print("NULL COVERAGE")
    print("=" * 70)
    
    if "earnings_null_reduction" in stats:
        s = stats["earnings_null_reduction"]
        print(f"cagr_earnings_5y:")
        print(f"  Before filling: {s['before']} nulls")
        print(f"  After filling:  {s['after']} nulls")
        print(f"  Filled: {s['filled']}")
    
    if "revenue_null_reduction" in stats:
        s = stats["revenue_null_reduction"]
        print(f"\ncagr_revenue_5y:")
        print(f"  Before filling: {s['before']} nulls")
        print(f"  After filling:  {s['after']} nulls")
        print(f"  Filled: {s['filled']}")
    
    # Print comparison
    if "earnings_comparison" in stats:
        print()
        print("=" * 70)
        print("VALIDATION (Bolsai vs Calculated where both exist)")
        print("=" * 70)
        s = stats["earnings_comparison"]
        print(f"Overlapping rows: {s['count']}")
        print(f"Mean absolute difference: {s['mean_diff']:.4f}%")
        print(f"Max difference: {s['max_diff']:.4f}%")
        print(f"Std deviation: {s['std_diff']:.4f}%")
    
    # Print sanity checks
    if "earnings_sanity" in stats:
        print()
        print("=" * 70)
        print("EARNINGS CAGR SANITY CHECK")
        print("=" * 70)
        s = stats["earnings_sanity"]
        print(f"Non-null values: {s['count']}")
        print(f"Mean: {s['mean']:.2f}%")
        print(f"Median: {s['median']:.2f}%")
        print(f"Range: {s['min']:.2f}% → {s['max']:.2f}%")
        print(f"Extreme high (>100%): {s['outliers_gt_100']}")
        print(f"Extreme low (<-100%): {s['outliers_lt_neg_100']}")
    
    if "revenue_sanity" in stats:
        print()
        print("=" * 70)
        print("REVENUE CAGR SANITY CHECK")
        print("=" * 70)
        s = stats["revenue_sanity"]
        print(f"Non-null values: {s['count']}")
        print(f"Mean: {s['mean']:.2f}%")
        print(f"Median: {s['median']:.2f}%")
        print(f"Range: {s['min']:.2f}% → {s['max']:.2f}%")
        print(f"Extreme high (>100%): {s['outliers_gt_100']}")
    
    # Print full table
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 150)
    
    print()
    print("=" * 160)
    print("FULL TABLE")
    print("=" * 160)
    
    display_cols = [
        "reference_date",
        "net_income",
        "net_revenue",
    ]
    
    if "cagr_earnings_5y" in df.columns:
        display_cols.extend([
            "cagr_earnings_5y",
            "cagr_earnings_5y_final",
            "had_negative_earnings_5y",
        ])
    
    if "cagr_revenue_5y" in df.columns:
        display_cols.extend([
            "cagr_revenue_5y",
            "cagr_revenue_5y_final",
        ])
    
    display_cols = [c for c in display_cols if c in df.columns]
    display = df[display_cols].copy()
    display["reference_date"] = display["reference_date"].dt.date
    print(display.to_string(index=False))
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(f"CAGR Report for {args.ticker}\n")
            f.write(f"Generated from {len(df)} quarters\n")
            f.write(f"Range: {df['reference_date'].min().date()} → {df['reference_date'].max().date()}\n\n")
            f.write(display.to_string(index=False))
        print(f"\nReport saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
