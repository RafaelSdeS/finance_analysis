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

def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    """Approximate equality allowing for floating-point rounding; NaN == NaN."""
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) < tol


def cagr_standard(v_now: float, v_ago: float, years: int = 5) -> float:
    """Standard CAGR. NaN if either value is non-positive."""
    if pd.isna(v_now) or pd.isna(v_ago) or v_ago <= 0 or v_now <= 0:
        return np.nan
    return ((v_now / v_ago) ** (1 / years) - 1) * 100


def calc_annual_cagr(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Computes CAGR at December rows only (annual anchors), then forward-fills
    Q1/Q2/Q3 from the most recent anchor — matching Bolsai's methodology.
    Looks back exactly 20 quarters (5 years).

    A non-December row must never produce its own value here: only December
    is the annual anchor. And when a December's own base year is invalid,
    the anchor must become NaN from that December onward, never silently
    keep an older anchor's value (see had_negative_base) — hence an explicit
    "current anchor" tracked row by row, not a plain .ffill(), which would
    skip the invalid December and reach back to a stale prior value.
    """
    result = pd.Series(np.nan, index=df.index)
    is_december = df["reference_date"].dt.month == 12
    anchor = np.nan
    for i in range(len(df)):
        if is_december.iloc[i] and i >= 20:
            anchor = cagr_standard(df[col].iloc[i], df[col].iloc[i - 20])
        result.iloc[i] = anchor
    return result


def had_negative_base(df: pd.DataFrame, col: str, lookback: int = 20) -> pd.Series:
    """
    Returns 1 if the currently-active annual anchor's base year (December,
    lookback quarters back) was negative or zero, meaning standard CAGR is
    undefined for that quarter. Anchored to December the same way as
    calc_annual_cagr so the two never disagree about which quarter's base
    year they're describing.
    """
    result = pd.Series(0, index=df.index)
    is_december = df["reference_date"].dt.month == 12
    anchor = 0
    for i in range(len(df)):
        if is_december.iloc[i] and i >= lookback:
            v_ago = df[col].iloc[i - lookback]
            anchor = 1 if (pd.isna(v_ago) or v_ago <= 0) else 0
        result.iloc[i] = anchor
    return result.astype(int)


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

    # ── Independent reference calculation (this file's own loop-based
    # formulas above, not cagr_handler's vectorized ones) ─────────────────
    ref_earnings_calc = calc_annual_cagr(df, "net_income")
    ref_had_neg = had_negative_base(df, "net_income")
    ref_revenue_calc = calc_annual_cagr(df, "net_revenue")

    # ── Null summary ──────────────────────────────────────────────────────────
    total = len(df)

    print_header("NULL COVERAGE")
    print(f"{'':40} {'earnings':>10} {'revenue':>10}")
    print(f"{'Bolsai nulls':40} {df['cagr_earnings_5y'].isna().sum():>10} {df['cagr_revenue_5y'].isna().sum():>10}")
    print(f"{'After filling with calc':40} {df['cagr_earnings_5y_final'].isna().sum():>10} {df['cagr_revenue_5y_final'].isna().sum():>10}")
    print(f"{'Remaining nulls (negative base)':40} {(df['cagr_earnings_5y_final'].isna() & (df['had_negative_earnings_5y']==1)).sum():>10} {'N/A':>10}")
    print(f"{'Total quarters':40} {total:>10} {total:>10}")

    # ── Validation ────────────────────────────────────────────────────────────
    # Note: cagr_*_calc columns are cleaned up in fill_cagr_columns, so the
    # reference values computed above (independent loop-based formulas, not
    # cagr_handler's vectorized ones) are the only way to check fill_cagr_columns'
    # actual output against a known-correct value, not just "did it run."
    print()
    print_header("INTERNAL CONSISTENCY CHECK")
    print("(fill_cagr_columns output vs. this file's independent reference calc)")

    bolsai_null_earnings = df["cagr_earnings_5y"].isna()
    earnings_mismatches = [
        i for i in df.index[bolsai_null_earnings]
        if not approx(df.loc[i, "cagr_earnings_5y_final"], ref_earnings_calc.loc[i])
    ]
    assert not earnings_mismatches, (
        f"cagr_earnings_5y_final diverges from reference calc on "
        f"{len(earnings_mismatches)} row(s) where Bolsai was null: {earnings_mismatches}"
    )
    print(f"  OK: cagr_earnings_5y_final matches reference calc "
          f"({bolsai_null_earnings.sum()} Bolsai-null rows checked)")

    bolsai_null_revenue = df["cagr_revenue_5y"].isna()
    revenue_mismatches = [
        i for i in df.index[bolsai_null_revenue]
        if not approx(df.loc[i, "cagr_revenue_5y_final"], ref_revenue_calc.loc[i])
    ]
    assert not revenue_mismatches, (
        f"cagr_revenue_5y_final diverges from reference calc on "
        f"{len(revenue_mismatches)} row(s) where Bolsai was null: {revenue_mismatches}"
    )
    print(f"  OK: cagr_revenue_5y_final matches reference calc "
          f"({bolsai_null_revenue.sum()} Bolsai-null rows checked)")

    flag_mismatches = df.index[df["had_negative_earnings_5y"] != ref_had_neg].tolist()
    assert not flag_mismatches, (
        f"had_negative_earnings_5y diverges from reference flag on "
        f"{len(flag_mismatches)} row(s): {flag_mismatches}"
    )
    print(f"  OK: had_negative_earnings_5y matches reference flag ({len(df)} rows checked)")

    # Definedness contract: when the base year was negative/zero AND Bolsai
    # supplied no override, fill_cagr_columns must leave the quarter null,
    # never fabricate a value from an undefined calculation.
    calc_only = bolsai_null_earnings & (ref_had_neg == 1)
    leaked = df.loc[calc_only, "cagr_earnings_5y_final"].notna().sum()
    assert leaked == 0, (
        f"{leaked} row(s) have a cagr_earnings_5y_final value despite a negative "
        f"base year and no Bolsai override — CAGR is mathematically undefined there"
    )
    print(f"  OK: no fabricated CAGR on negative-base, Bolsai-null quarters "
          f"({calc_only.sum()} such quarters found)")


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