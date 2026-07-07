#!/usr/bin/env python3
"""
Dataset Verification for ML Agent Training

Checks dataset quality, coverage, and feature completeness before training.
Determines actual train/val/test split dates based on data availability.

Run: python tests/agent/verify_dataset_for_training.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path


def verify_dataset_for_training(dataset_path: str = "data/processed/ml_dataset.parquet") -> dict:
    """Comprehensive dataset verification before training."""

    print("=" * 70)
    print("DATASET VERIFICATION FOR ML AGENT TRAINING")
    print("=" * 70)

    # Load dataset
    print(f"\nLoading {dataset_path}...")
    if not Path(dataset_path).exists():
        print(f"✗ FAIL: File not found: {dataset_path}")
        return {"pass": False, "error": "File not found"}

    df = pd.read_parquet(dataset_path)
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")

    results = {}

    # ===== V1: Date Coverage & Continuity =====
    print("\n" + "=" * 70)
    print("V1: DATE COVERAGE & CONTINUITY")
    print("=" * 70)

    # Detect date column
    date_candidates = ['trade_date', 'date', 'datetime', 'timestamp']
    date_col = None
    for col in date_candidates:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        print(f"✗ FAIL: No date column found. Available: {df.columns.tolist()[:10]}")
        return {"pass": False, "error": "No date column found"}

    if date_col != 'date':
        print(f"ℹ Using column '{date_col}' as date")

    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    results['date_min'] = df[date_col].min()
    results['date_max'] = df[date_col].max()
    results['span_days'] = (results['date_max'] - results['date_min']).days
    results['span_years'] = results['span_days'] / 365.25

    print(f"Date range: {results['date_min'].date()} → {results['date_max'].date()}")
    print(f"Span: {results['span_years']:.2f} years ({results['span_days']} days)")

    results['pass_v1'] = results['span_years'] >= 2.0
    if results['pass_v1']:
        print(f"✓ PASS: ≥2 years of data")
    else:
        print(f"✗ FAIL: <2 years (need minimum 2 years for train/val/test split)")

    # ===== V2: Ticker Coverage & Completeness =====
    print("\n" + "=" * 70)
    print("V2: TICKER COVERAGE & COMPLETENESS")
    print("=" * 70)

    results['n_tickers'] = df['ticker'].nunique()
    print(f"Unique tickers: {results['n_tickers']}")

    ticker_counts = df.groupby('ticker').size()
    ticker_counts_sorted = ticker_counts.sort_values(ascending=False)

    results['tickers_full_history'] = (ticker_counts >= 252).sum()
    results['tickers_partial_history'] = ((ticker_counts >= 100) & (ticker_counts < 252)).sum()
    results['tickers_insufficient'] = (ticker_counts < 100).sum()

    print(f"  • ≥1 year (252+ rows): {results['tickers_full_history']} tickers")
    print(f"  • 4-12 months (100-251 rows): {results['tickers_partial_history']} tickers")
    print(f"  • <4 months (<100 rows): {results['tickers_insufficient']} tickers")
    print(f"\nTop 10 tickers by row count:")
    for ticker, count in ticker_counts_sorted.head(10).items():
        years = count / 252
        print(f"  {ticker:8} {count:5} rows (~{years:.1f} years)")

    results['pass_v2'] = results['tickers_full_history'] >= 20
    if results['pass_v2']:
        print(f"\n✓ PASS: ≥20 tickers with full history")
    else:
        print(f"\n✗ FAIL: <20 tickers with full history (need ≥20 for portfolio diversity)")

    # ===== V3: Feature Completeness =====
    print("\n" + "=" * 70)
    print("V3: FEATURE COMPLETENESS")
    print("=" * 70)

    required_features = {
        # 'returns' is a Stage-3 feature_engineering.py column (from adj_close);
        # this script verifies the Stage-2 dataset, which has 'log_return' instead
        'price': ['close', 'volume', 'log_return', 'open', 'high', 'low'],
        'technical': ['volatility_20d', 'volatility_60d', 'rsi_14'],
        'fundamental': ['pl', 'pvp', 'roe', 'debt_equity'],  # Brazilian names: pl=P/E, pvp=P/B
        'macro': ['selic', 'ipca', 'cdi'],
        'meta': ['ticker', date_col, 'sector'],
    }

    print("Expected features by category:")
    missing_cols = []
    for category, features in required_features.items():
        present = [f for f in features if f in df.columns]
        absent = [f for f in features if f not in df.columns]
        missing_cols.extend(absent)

        status = "✓" if len(absent) == 0 else "✗"
        print(f"  {status} {category:12} {len(present)}/{len(features)} present")
        if absent:
            print(f"      Missing: {', '.join(absent)}")

    results['missing_cols'] = missing_cols
    results['pass_v3'] = len(missing_cols) == 0

    if results['pass_v3']:
        print(f"\n✓ PASS: All required features present")
    else:
        print(f"\n✗ FAIL: Missing features: {', '.join(missing_cols)}")

    # ===== V4: Missing Data (NaN Rates) =====
    print("\n" + "=" * 70)
    print("V4: MISSING DATA (NaN RATES)")
    print("=" * 70)

    # Overall
    total_cells = len(df) * len(df.columns)
    null_cells = df.isnull().sum().sum()
    results['nan_rate_overall'] = null_cells / total_cells if total_cells > 0 else 0

    print(f"Overall NaN rate: {results['nan_rate_overall']:.2%} ({null_cells:,} / {total_cells:,} cells)")

    # Per-column (critical columns)
    critical_cols = ['ticker', date_col, 'close', 'volume', 'sector']
    critical_ok = True
    print(f"\nCritical columns (should be 0% NaN):")
    for col in critical_cols:
        if col in df.columns:
            nan_rate = df[col].isnull().sum() / len(df)
            critical_ok = critical_ok and nan_rate == 0
            status = "✓" if nan_rate == 0 else "✗"
            print(f"  {status} {col:15} {nan_rate:6.2%}")

    # Per-column (all columns >20% NaN)
    nan_per_col = df.isnull().sum() / len(df)
    high_nan_cols = nan_per_col[nan_per_col > 0.2].sort_values(ascending=False)
    if len(high_nan_cols) > 0:
        print(f"\nColumns with >20% NaN:")
        for col, rate in high_nan_cols.items():
            print(f"  {col:30} {rate:6.2%}")

    # Per-ticker NaN rate
    ticker_nan_rates = df.groupby('ticker').apply(lambda g: g.isnull().sum().sum() / (len(g) * len(g.columns)))
    high_nan_tickers = ticker_nan_rates[ticker_nan_rates > 0.2].sort_values(ascending=False)
    if len(high_nan_tickers) > 0:
        print(f"\nTickers with >20% NaN (may want to exclude):")
        for ticker, rate in high_nan_tickers.head(5).items():
            print(f"  {ticker:8} {rate:6.2%}")

    # Structural missingness: fundamentals only exist from a ticker's first
    # filing (CVM digital filings start ~2011); before that NaN is legitimate.
    # Gate on NaN where data could exist, not on the overall rate.
    fund_cols = [c for c in ['pl', 'pvp', 'roe', 'debt_equity', 'roic', 'roa',
                             'net_margin', 'gross_margin', 'ebitda_margin',
                             'current_ratio', 'cash_ratio', 'earnings_growth_yoy',
                             'revenue_growth_yoy', 'ebitda_growth_yoy']
                 if c in df.columns]
    if 'reference_date' in df.columns and fund_cols:
        first_filing = df.loc[df['reference_date'].notna()].groupby('ticker')[date_col].min()
        covered = df[date_col] >= df['ticker'].map(first_filing)
        results['nan_rate_fund_covered'] = float(df.loc[covered, fund_cols].isnull().to_numpy().mean())
        print(f"\nRows before ticker's first filing (structural NaN, expected): {(~covered).mean():.2%}")
        print(f"Fundamental-feature NaN rate after first filing: {results['nan_rate_fund_covered']:.2%}")
        results['pass_v4'] = critical_ok and results['nan_rate_fund_covered'] < 0.10
        if results['pass_v4']:
            print(f"\n✓ PASS: critical columns clean, post-filing fundamental NaN <10%")
        else:
            print(f"\n✗ FAIL: critical-column NaN, or post-filing fundamental NaN ≥10%")
    else:
        results['pass_v4'] = critical_ok and results['nan_rate_overall'] < 0.05
        if results['pass_v4']:
            print(f"\n✓ PASS: Overall NaN rate <5%")
        else:
            print(f"\n✗ FAIL: Overall NaN rate ≥5% (imputation needed)")

    # ===== V5: Feature Distributions & Outliers =====
    print("\n" + "=" * 70)
    print("V5: FEATURE DISTRIBUTIONS & OUTLIERS")
    print("=" * 70)

    # Returns
    if 'returns' in df.columns:
        ret_mean = df['returns'].mean()
        ret_std = df['returns'].std()
        ret_min = df['returns'].min()
        ret_max = df['returns'].max()

        print(f"Returns distribution:")
        print(f"  Mean: {ret_mean:8.4f} (expect ~0)")
        print(f"  Std:  {ret_std:8.4f} (expect 0.01-0.10)")
        print(f"  Min:  {ret_min:8.4f} (max loss per day)")
        print(f"  Max:  {ret_max:8.4f} (max gain per day)")

        results['return_mean'] = ret_mean
        results['return_std'] = ret_std

    # Duplicates
    results['n_duplicates'] = len(df) - len(df.drop_duplicates(['ticker', date_col]))
    print(f"\nDuplicates (ticker, {date_col}): {results['n_duplicates']}")

    # Extreme outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_count = 0
    for col in numeric_cols:
        if col not in ['returns']:  # Skip returns, it's allowed to have extremes
            q75 = df[col].quantile(0.75)
            q25 = df[col].quantile(0.25)
            iqr = q75 - q25
            upper_bound = q75 + 3 * iqr
            lower_bound = q25 - 3 * iqr
            extreme = ((df[col] > upper_bound) | (df[col] < lower_bound)).sum()
            if extreme > 0:
                outlier_count += extreme

    # Valuation freshness: daily-recomputed P/L must vary within a quarter.
    # Frozen ticker-quarters mean the stale filing-date API ratio regressed in.
    results['pl_frozen_share'] = None
    if {'pl', 'reference_date'}.issubset(df.columns):
        grp = df[df['pl'].notna()].groupby(['ticker', 'reference_date'])['pl']
        sizes, nun = grp.size(), grp.nunique()
        eligible = sizes >= 5  # a quarter spanning <5 trading days may legitimately be constant
        n_eligible = int(eligible.sum())
        frozen_share = float(((nun == 1) & eligible).sum() / max(n_eligible, 1))
        results['pl_frozen_share'] = frozen_share
        print(f"\nP/L frozen within quarter (stale valuation, expect ~0%): {frozen_share:.1%} of {n_eligible} ticker-quarters")

    results['pass_v5'] = (results['n_duplicates'] == 0
                          and abs(results.get('return_mean', 0)) < 0.1
                          and (results['pl_frozen_share'] is None or results['pl_frozen_share'] < 0.01))
    if results['pass_v5']:
        print(f"\n✓ PASS: No duplicates, reasonable distributions, valuation moves daily")
    else:
        print(f"\n✗ WARNING: Check distributions (duplicates={results['n_duplicates']}, return_mean={results.get('return_mean', 'N/A')}, pl_frozen={results['pl_frozen_share']})")

    # ===== V6: Temporal Alignment (No Lookahead Bias) =====
    print("\n" + "=" * 70)
    print("V6: TEMPORAL ALIGNMENT (NO LOOKAHEAD BIAS)")
    print("=" * 70)

    fund_date_col = 'reference_date' if 'reference_date' in df.columns else 'fundamental_date'
    if fund_date_col in df.columns:
        lookahead_violations = (df[fund_date_col] > df[date_col]).sum()
        results['lookahead_bias'] = lookahead_violations

        print(f"Rows where {fund_date_col} > {date_col}: {lookahead_violations}")

        if lookahead_violations > 0:
            print(f"✗ FAIL: Lookahead bias detected! {lookahead_violations} violations")
            results['pass_v6'] = False
        else:
            print(f"✓ PASS: No lookahead bias")
            results['pass_v6'] = True
    else:
        print(f"⚠ no fundamental date column found, skipping check")
        results['pass_v6'] = True

    # ===== V7: Sector Coverage =====
    print("\n" + "=" * 70)
    print("V7: SECTOR COVERAGE")
    print("=" * 70)

    if 'sector' in df.columns:
        sector_dist = df['sector'].value_counts()
        results['n_sectors'] = len(sector_dist)

        print(f"Unique sectors: {results['n_sectors']}")
        print(f"Distribution:")
        for sector, count in sector_dist.items():
            pct = count / len(df) * 100
            print(f"  {sector:20} {count:6} rows ({pct:5.1f}%)")

        max_sector_pct = sector_dist.iloc[0] / len(df) * 100
        results['pass_v7'] = results['n_sectors'] >= 3 and max_sector_pct < 40

        if results['pass_v7']:
            print(f"\n✓ PASS: ≥3 sectors, max sector <40%")
        else:
            print(f"\n✗ FAIL: Insufficient sector diversity")
    else:
        print(f"⚠ 'sector' column not found, skipping check")
        results['pass_v7'] = True

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    checks = {
        'V1 Date Coverage': results['pass_v1'],
        'V2 Ticker Coverage': results['pass_v2'],
        'V3 Feature Completeness': results['pass_v3'],
        'V4 NaN Rates': results['pass_v4'],
        'V5 Distributions': results['pass_v5'],
        'V6 No Lookahead Bias': results['pass_v6'],
        'V7 Sector Coverage': results['pass_v7'],
    }

    for check, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}  {check}")

    all_pass = all(checks.values())
    results['pass'] = all_pass

    # Train/val/test splits are derived from anchored rolling windows, not a
    # fixed recommendation — see `python src/agent/config.py` (DEFAULT_CONFIG
    # .log_summary()) for the current split, or `generate_windows()` in
    # config.py for the full window schedule.

    # ===== FINAL VERDICT =====
    print("\n" + "=" * 70)
    if all_pass:
        print("✓ READY FOR TRAINING")
        print("Dataset passes all verification checks. Proceed with Phase 3a implementation.")
    else:
        print("✗ ISSUES DETECTED")
        print("Fix failing checks before proceeding with Phase 3a implementation.")
    print("=" * 70 + "\n")

    return results


if __name__ == '__main__':
    results = verify_dataset_for_training()
    sys.exit(0 if results.get('pass') else 1)
