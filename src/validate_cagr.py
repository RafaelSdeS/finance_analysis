"""
validate_cagr.py
================

Validate CAGR values against multiple sources:

1. **Internal Consistency**: Compare Bolsai CAGR vs calculated CAGR
2. **Sanity Checks**: Check if values are in realistic ranges
3. **Outlier Detection**: Identify suspicious values for manual review
4. **Cross-ticker Comparison**: Compare ranges across companies (sector-based)

Realistic CAGR Ranges (based on financial literature):
    - Earnings CAGR:
        * Mature companies: -50% to +30%
        * Growth companies: -30% to +100%
        * Outliers (>100%): Very rare, usually small cap/speculative
    - Revenue CAGR:
        * Mature companies: -20% to +20%
        * Growth companies: -10% to +50%
        * Outliers (>50%): Possible but worth investigating

Usage (as a module):
    from validate_cagr import validate_fundamentals
    
    df = pd.read_parquet("fundamentals/PETR4.parquet")
    report = validate_fundamentals(df, "PETR4")
    print(report)

Usage (as a script):
    python validate_cagr.py --ticker PETR4
    python validate_cagr.py --fund-dir ../data/raw/fundamentals --output validation_report.txt
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CAGRValidationResult:
    """Result of CAGR validation for a single ticker."""
    ticker: str
    total_quarters: int
    date_range: str
    
    # Internal consistency
    bolsai_calc_correlation: float
    mean_abs_diff: float
    max_abs_diff: float
    std_abs_diff: float
    
    # Sanity checks
    earnings_reasonable: bool
    revenue_reasonable: bool
    earnings_outliers: Dict
    revenue_outliers: Dict
    
    # Coverage
    earnings_coverage_improvement: Dict
    revenue_coverage_improvement: Dict
    
    # Warnings
    warnings: List[str]
    overall_quality: str  # 'pass', 'warning', 'fail'


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

EARNINGS_RANGES = {
    "mature": {"min": -50, "max": 30},
    "growth": {"min": -30, "max": 100},
    "extreme": {"min": -100, "max": 500},  # Possible but flagged
}

REVENUE_RANGES = {
    "mature": {"min": -20, "max": 20},
    "growth": {"min": -10, "max": 50},
    "extreme": {"min": -50, "max": 200},
}


def classify_outlier_severity(value: float, metric: str = "earnings") -> str:
    """Classify how extreme a CAGR value is."""
    ranges = EARNINGS_RANGES if metric == "earnings" else REVENUE_RANGES
    
    if np.isnan(value):
        return "missing"
    
    if ranges["mature"]["min"] <= value <= ranges["mature"]["max"]:
        return "normal"
    elif ranges["growth"]["min"] <= value <= ranges["growth"]["max"]:
        return "growth_company"
    elif ranges["extreme"]["min"] <= value <= ranges["extreme"]["max"]:
        return "extreme"
    else:
        return "impossible"


def check_internal_consistency(df: pd.DataFrame) -> Dict:
    """
    Compare Bolsai CAGR vs calculated CAGR where both exist.
    
    Returns stats on agreement/disagreement.
    """
    result = {
        "has_comparison": False,
        "count": 0,
        "correlation": np.nan,
        "mean_diff": np.nan,
        "mean_abs_diff": np.nan,
        "max_abs_diff": np.nan,
        "std_abs_diff": np.nan,
        "high_diff_rows": [],  # Rows where difference > 10%
    }
    
    # Check for comparison data
    has_bolsai = "cagr_earnings_5y" in df.columns
    has_calc = "cagr_earnings_calc" in df.columns
    
    if not (has_bolsai and has_calc):
        return result
    
    # Filter to rows where both exist
    compare_df = df[
        df["cagr_earnings_5y"].notna() & 
        df["cagr_earnings_calc"].notna()
    ].copy()
    
    if len(compare_df) == 0:
        return result
    
    result["has_comparison"] = True
    result["count"] = len(compare_df)
    
    # Calculate differences
    compare_df["diff"] = compare_df["cagr_earnings_5y"] - compare_df["cagr_earnings_calc"]
    compare_df["abs_diff"] = compare_df["diff"].abs()
    
    result["mean_diff"] = compare_df["diff"].mean()
    result["mean_abs_diff"] = compare_df["abs_diff"].mean()
    result["max_abs_diff"] = compare_df["abs_diff"].max()
    result["std_abs_diff"] = compare_df["abs_diff"].std()
    
    # Correlation
    if len(compare_df) > 1:
        result["correlation"] = compare_df["cagr_earnings_5y"].corr(
            compare_df["cagr_earnings_calc"]
        )
    
    # Flag large discrepancies (>10% difference)
    high_diff = compare_df[compare_df["abs_diff"] > 10].copy()
    if len(high_diff) > 0:
        high_diff["date"] = high_diff["reference_date"].dt.date
        result["high_diff_rows"] = high_diff[[
            "date",
            "cagr_earnings_5y",
            "cagr_earnings_calc",
            "abs_diff"
        ]].to_dict("records")
    
    return result


def check_sanity(df: pd.DataFrame) -> Dict:
    """
    Check if CAGR values are in realistic ranges.
    
    Returns classification of each metric.
    """
    result = {
        "earnings": {
            "normal": 0,
            "growth_company": 0,
            "extreme": 0,
            "impossible": 0,
            "missing": 0,
        },
        "revenue": {
            "normal": 0,
            "growth_company": 0,
            "extreme": 0,
            "impossible": 0,
            "missing": 0,
        },
        "earnings_extremes": [],  # Details of extreme values
        "revenue_extremes": [],
    }
    
    # Check earnings
    if "cagr_earnings_5y_final" in df.columns:
        for val in df["cagr_earnings_5y_final"]:
            severity = classify_outlier_severity(val, "earnings")
            result["earnings"][severity] += 1
            
            if severity in ["extreme", "impossible"]:
                result["earnings_extremes"].append({
                    "value": float(val) if not np.isnan(val) else None,
                    "severity": severity,
                })
    
    # Check revenue
    if "cagr_revenue_5y_final" in df.columns:
        for val in df["cagr_revenue_5y_final"]:
            severity = classify_outlier_severity(val, "revenue")
            result["revenue"][severity] += 1
            
            if severity in ["extreme", "impossible"]:
                result["revenue_extremes"].append({
                    "value": float(val) if not np.isnan(val) else None,
                    "severity": severity,
                })
    
    return result


def check_coverage(df: pd.DataFrame) -> Dict:
    """
    Check null coverage before/after filling.
    """
    total = len(df)
    result = {
        "total_rows": total,
        "earnings": {
            "bolsai_nulls": 0,
            "final_nulls": 0,
            "coverage_improvement": 0.0,
        },
        "revenue": {
            "bolsai_nulls": 0,
            "final_nulls": 0,
            "coverage_improvement": 0.0,
        },
    }
    
    if "cagr_earnings_5y" in df.columns and "cagr_earnings_5y_final" in df.columns:
        bolsai_nulls = df["cagr_earnings_5y"].isna().sum()
        final_nulls = df["cagr_earnings_5y_final"].isna().sum()
        result["earnings"]["bolsai_nulls"] = bolsai_nulls
        result["earnings"]["final_nulls"] = final_nulls
        if bolsai_nulls > 0:
            result["earnings"]["coverage_improvement"] = (
                (bolsai_nulls - final_nulls) / bolsai_nulls * 100
            )
    
    if "cagr_revenue_5y" in df.columns and "cagr_revenue_5y_final" in df.columns:
        bolsai_nulls = df["cagr_revenue_5y"].isna().sum()
        final_nulls = df["cagr_revenue_5y_final"].isna().sum()
        result["revenue"]["bolsai_nulls"] = bolsai_nulls
        result["revenue"]["final_nulls"] = final_nulls
        if bolsai_nulls > 0:
            result["revenue"]["coverage_improvement"] = (
                (bolsai_nulls - final_nulls) / bolsai_nulls * 100
            )
    
    return result


def generate_warnings(df: pd.DataFrame, sanity: Dict, consistency: Dict) -> List[str]:
    """Generate warnings based on validation checks."""
    warnings = []
    
    # Internal consistency issues
    if consistency["has_comparison"]:
        if consistency["correlation"] < 0.8 and consistency["count"] > 10:
            warnings.append(
                f"Low correlation between Bolsai and calculated CAGR "
                f"({consistency['correlation']:.2f}). Possible calculation issue."
            )
        if consistency["mean_abs_diff"] > 5:
            warnings.append(
                f"Mean absolute difference between Bolsai and calculated CAGR is "
                f"{consistency['mean_abs_diff']:.2f}%. High divergence detected."
            )
    
    # Sanity check issues
    if sanity["earnings"]["impossible"] > 0:
        warnings.append(
            f"Found {sanity['earnings']['impossible']} impossible earnings CAGR values "
            f"(outside -100% to +500%). Data quality issue?"
        )
    
    if sanity["revenue"]["impossible"] > 0:
        warnings.append(
            f"Found {sanity['revenue']['impossible']} impossible revenue CAGR values. "
            f"Data quality issue?"
        )
    
    extreme_earnings = sanity["earnings"]["extreme"]
    if extreme_earnings > len(df) * 0.1:  # >10% extreme
        warnings.append(
            f"Many extreme earnings CAGR values ({extreme_earnings}). "
            f"May indicate high-volatility company."
        )
    
    # Coverage warnings
    if "cagr_earnings_5y_final" in df.columns:
        coverage = 100 * (1 - df["cagr_earnings_5y_final"].isna().sum() / len(df))
        if coverage < 70:
            warnings.append(
                f"Low earnings CAGR coverage ({coverage:.0f}%). "
                f"Many quarters with missing data."
            )
    
    return warnings


def validate_fundamentals(df: pd.DataFrame, ticker: str) -> CAGRValidationResult:
    """
    Main validation function. Run all checks and return comprehensive report.
    """
    df = df.sort_values("reference_date").reset_index(drop=True)
    
    # Fill CAGR first if not already filled
    if "cagr_earnings_5y_final" not in df.columns:
        from cagr_handler import fill_cagr_columns
        df = fill_cagr_columns(df)
    
    date_range = f"{df['reference_date'].min().date()} → {df['reference_date'].max().date()}"
    
    # Run all checks
    consistency = check_internal_consistency(df)
    sanity = check_sanity(df)
    coverage = check_coverage(df)
    warnings = generate_warnings(df, sanity, consistency)
    
    # Determine overall quality
    if len(warnings) > 2:
        overall_quality = "fail"
    elif len(warnings) > 0:
        overall_quality = "warning"
    else:
        overall_quality = "pass"
    
    return CAGRValidationResult(
        ticker=ticker,
        total_quarters=len(df),
        date_range=date_range,
        bolsai_calc_correlation=consistency.get("correlation", np.nan),
        mean_abs_diff=consistency.get("mean_abs_diff", np.nan),
        max_abs_diff=consistency.get("max_abs_diff", np.nan),
        std_abs_diff=consistency.get("std_abs_diff", np.nan),
        earnings_reasonable=sanity["earnings"]["impossible"] == 0,
        revenue_reasonable=sanity["revenue"]["impossible"] == 0,
        earnings_outliers=sanity["earnings"],
        revenue_outliers=sanity["revenue"],
        earnings_coverage_improvement=coverage["earnings"],
        revenue_coverage_improvement=coverage["revenue"],
        warnings=warnings,
        overall_quality=overall_quality,
    )


# =============================================================================
# REPORTING
# =============================================================================

def print_validation_report(result: CAGRValidationResult):
    """Pretty-print validation results."""
    print()
    print("=" * 80)
    print(f"CAGR VALIDATION REPORT: {result.ticker}")
    print("=" * 80)
    
    print(f"\nTicket: {result.ticker}")
    print(f"Quarters: {result.total_quarters}")
    print(f"Date Range: {result.date_range}")
    print(f"Overall Quality: {result.overall_quality.upper()}")
    
    # Internal consistency
    print()
    print("-" * 80)
    print("INTERNAL CONSISTENCY (Bolsai vs Calculated)")
    print("-" * 80)
    print(f"Correlation: {result.bolsai_calc_correlation:.3f}")
    print(f"Mean absolute difference: {result.mean_abs_diff:.2f}%")
    print(f"Max absolute difference: {result.max_abs_diff:.2f}%")
    print(f"Std deviation: {result.std_abs_diff:.2f}%")
    
    # Sanity
    print()
    print("-" * 80)
    print("EARNINGS CAGR CLASSIFICATION")
    print("-" * 80)
    print(f"Normal: {result.earnings_outliers['normal']}")
    print(f"Growth company: {result.earnings_outliers['growth_company']}")
    print(f"Extreme: {result.earnings_outliers['extreme']}")
    print(f"Impossible: {result.earnings_outliers['impossible']}")
    print(f"Missing: {result.earnings_outliers['missing']}")
    print(f"Reasonable: {'✓' if result.earnings_reasonable else '✗'}")
    
    print()
    print("-" * 80)
    print("REVENUE CAGR CLASSIFICATION")
    print("-" * 80)
    print(f"Normal: {result.revenue_outliers['normal']}")
    print(f"Growth company: {result.revenue_outliers['growth_company']}")
    print(f"Extreme: {result.revenue_outliers['extreme']}")
    print(f"Impossible: {result.revenue_outliers['impossible']}")
    print(f"Missing: {result.revenue_outliers['missing']}")
    print(f"Reasonable: {'✓' if result.revenue_reasonable else '✗'}")
    
    # Coverage
    print()
    print("-" * 80)
    print("COVERAGE IMPROVEMENT")
    print("-" * 80)
    earnings = result.earnings_coverage_improvement
    print(f"Earnings:")
    print(f"  Bolsai nulls: {earnings['bolsai_nulls']}")
    print(f"  Final nulls: {earnings['final_nulls']}")
    print(f"  Improvement: {earnings['coverage_improvement']:.1f}%")
    
    revenue = result.revenue_coverage_improvement
    print(f"Revenue:")
    print(f"  Bolsai nulls: {revenue['bolsai_nulls']}")
    print(f"  Final nulls: {revenue['final_nulls']}")
    print(f"  Improvement: {revenue['coverage_improvement']:.1f}%")
    
    # Warnings
    if result.warnings:
        print()
        print("-" * 80)
        print("⚠ WARNINGS")
        print("-" * 80)
        for i, warning in enumerate(result.warnings, 1):
            print(f"{i}. {warning}")
    else:
        print()
        print("✓ No warnings detected")


# =============================================================================
# CLI SCRIPT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate CAGR values for consistency and realism."
    )
    parser.add_argument("--ticker", help="Single ticker to validate")
    parser.add_argument("--fund-dir", default="../data/raw/fundamentals",
                        help="Fundamentals data directory")
    parser.add_argument("--output", help="Output JSON report file")
    args = parser.parse_args()
    
    fund_dir = Path(args.fund_dir)
    
    results = []
    
    if args.ticker:
        # Validate single ticker
        ticker_path = fund_dir / f"{args.ticker}.parquet"
        if not ticker_path.exists():
            print(f"Error: {ticker_path} not found")
            return 1
        
        df = pd.read_parquet(ticker_path)
        result = validate_fundamentals(df, args.ticker)
        results.append(result)
        print_validation_report(result)
    
    else:
        # Validate all tickers in directory
        parquet_files = sorted(fund_dir.glob("*.parquet"))
        
        if not parquet_files:
            print(f"Error: No parquet files found in {fund_dir}")
            return 1
        
        print(f"Validating {len(parquet_files)} tickers...\n")
        
        for file in parquet_files:
            ticker = file.stem
            df = pd.read_parquet(file)
            result = validate_fundamentals(df, ticker)
            results.append(result)
            print_validation_report(result)
    
    # Save JSON report if requested
    if args.output and results:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "total_tickers": len(results),
            "results": [asdict(r) for r in results],
        }
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\n✓ Report saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
