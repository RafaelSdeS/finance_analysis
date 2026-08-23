"""
Test data quality on the 50 most-traded tickers (union of whole-period top 50
+ per-year top 50).

The global ml_dataset validation (test_final_dataset.py) averages over 528
tickers; this narrows the lens to the most-liquid names where data bugs matter
most for the agent. Checks OHLC consistency, trading-calendar gaps, and
fundamental coverage — not duplicated by the global test.

Run from project root:
    python tests/build_dataset/test_top_traded_quality.py
    python tests/build_dataset/test_top_traded_quality.py --file data/processed/ml_dataset.parquet --strict
    python tests/build_dataset/test_top_traded_quality.py --top-n 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from tests.build_dataset.test_final_dataset import (  # noqa: E402
    check_stale_prices,
    check_outliers_zscore,
)
from test_utils import print_header, print_check, print_separator, numeric_columns  # noqa: E402

DEFAULT_FILE = ROOT / "data/processed/ml_dataset.parquet"
# With no --file, sweep both markets rather than BR alone. The US top-100
# subset is the same shape and carries every column these checks read
# (macro_cols below is an EXCLUSION set, so the US panel lacking `cdi` is
# harmless). Gitignored and rebuildable, so a missing US file is a SKIP, not
# a failure -- an explicit --file that is missing is still an error.
DEFAULT_FILES = [DEFAULT_FILE, ROOT / "data/processed/us_ml_dataset_top100.parquet"]


def build_universe(df, top_n=50):
    """
    Union of whole-period top N + per-year top N by traded_amount.
    Returns (set of tickers, subset dataframe).
    """
    whole_period = df.groupby("ticker")["traded_amount"].sum().nlargest(top_n).index

    per_year = (df
        .assign(year=df["trade_date"].dt.year)
        .groupby(["year", "ticker"])["traded_amount"].sum()
        .groupby("year", group_keys=False)
        .apply(lambda x: x.nlargest(top_n), include_groups=False)
    )
    per_year_tickers = set(per_year.index.get_level_values("ticker"))

    universe = set(whole_period) | per_year_tickers
    return universe, df[df["ticker"].isin(universe)]


def check_ohlc_consistency(df):
    """
    Verify OHLC internal consistency: low <= open/close <= high.
    Float tolerance ~1e-9 for rounding errors.
    Returns list of violations [ticker, date, column, reason].
    """
    findings = []
    tol = 1e-9

    for t, g in df.sort_values("trade_date").groupby("ticker"):
        for col in ["open", "close"]:
            # low <= col <= high
            below = g[g[col] < g["low"] - tol]
            above = g[g[col] > g["high"] + tol]
            for _, row in below.iterrows():
                findings.append({
                    "ticker": t,
                    "date": row["trade_date"],
                    "column": col,
                    "reason": f"{col}={row[col]:.2f} < low={row['low']:.2f}",
                })
            for _, row in above.iterrows():
                findings.append({
                    "ticker": t,
                    "date": row["trade_date"],
                    "column": col,
                    "reason": f"{col}={row[col]:.2f} > high={row['high']:.2f}",
                })

        # high >= low
        bad = g[g["high"] < g["low"] - tol]
        for _, row in bad.iterrows():
            findings.append({
                "ticker": t,
                "date": row["trade_date"],
                "column": "high/low",
                "reason": f"high={row['high']:.2f} < low={row['low']:.2f}",
            })

    return pd.DataFrame(findings)


def check_trading_calendar_gaps(df, max_gap_days=90):
    """
    Per ticker, find the max gap (in calendar days) between consecutive
    trade_date rows. Fail if any gap exceeds max_gap_days for the most-
    traded names; this is likely a collection bug, not normal illiquidity.

    ponytail: fixed 90-day threshold (≈18 trading weeks), not calendar-aware;
    revisit if false-positives on known events (circuit breakers, extended
    market closures, delistings). Lower bound (20-30 days) catches bugs; upper
    bound (90+) catches actual delistings/long suspensions.
    """
    findings = []

    for t, g in df.sort_values("trade_date").groupby("ticker"):
        dates = g["trade_date"].sort_values().values
        if len(dates) < 2:
            continue
        gaps = pd.to_datetime(dates[1:]) - pd.to_datetime(dates[:-1])
        gap_days = gaps.days.max()
        if gap_days > max_gap_days:
            # Find the actual max-gap pair for reporting
            max_idx = gaps.days.argmax()
            d1, d2 = pd.to_datetime(dates[max_idx]), pd.to_datetime(dates[max_idx + 1])
            findings.append({
                "ticker": t,
                "date1": d1,
                "date2": d2,
                "gap_days": gap_days,
            })

    return pd.DataFrame(findings)


def check_fundamentals_coverage(df):
    """
    For rows with has_fundamentals=1, report % NaN in key ratio columns.
    Printed informational; fundamentals sparsity is expected (~60-67% per
    CLAUDE.md), so this doesn't fail — just surfaces the profile.
    """
    have_fund = df[df.get("has_fundamentals", 0) == 1]
    if len(have_fund) == 0:
        return "No rows with has_fundamentals=1; skipping coverage check."

    lines = []
    for col in ["pl", "pvp", "roe", "net_income"]:
        if col not in df.columns:
            continue
        pct_nan = 100 * have_fund[col].isna().mean()
        lines.append(f"  {col:<15} {pct_nan:6.2f}% NaN (n={len(have_fund)})")

    return "\n".join(lines) if lines else "No fundamental columns found."


def validate(df, universe_size):
    """
    Hard-fail checks, gated on exit code. Reports PASS/FAIL.
    """
    print()
    print_header("VALIDATION")

    checks = []

    # No NaN in critical columns.
    #
    # adj_close is deliberately NOT in this list (2026-08-23). It used to be,
    # and that made this test contradict the repo's own decided policy: a NaN
    # or non-positive adj_close is an ACCEPTED, unrecoverable BolsAI 2-decimal
    # precision underflow (CLAUDE.md; DATA_LAYER_CORRECTNESS_PLAN.md §2a),
    # already masked before log() -- provided it carries
    # adj_close_precision_degraded == 1. test_final_dataset.py asserts exactly
    # that rule; this file asserted zero-tolerance instead and hard-failed the
    # whole top-50 gate on BPAC11/EPAR3, data nobody had decided was a defect.
    # Measured 2026-08-23 on the full BR panel: 543 such rows, 0 unflagged.
    for col in ("open", "high", "low", "close", "volume", "traded_amount"):
        if col in df.columns:
            nan_count = df[col].isna().sum()
            checks.append((f"no NaN in {col} [{nan_count} found]", nan_count == 0))

    if {"adj_close", "adj_close_precision_degraded"}.issubset(df.columns):
        bad_adj = df["adj_close"].isna() | (df["adj_close"] <= 0)
        unflagged = int((bad_adj & (df["adj_close_precision_degraded"] != 1)).sum())
        checks.append((f"NaN/non-positive adj_close always flagged degraded=1 "
                       f"[{unflagged} unflagged of {int(bad_adj.sum())} such rows]",
                       unflagged == 0))

    # No non-positive prices (exception: old delisted tickers may have adj_close=0 from collection)
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            bad = (df[col] <= 0).sum()
            checks.append((f"all {col} > 0 [{bad} found]", bad == 0))

    # OHLC consistency
    ohlc_violations = check_ohlc_consistency(df)
    checks.append((f"OHLC consistency [{len(ohlc_violations)} violations]", len(ohlc_violations) == 0))

    # No duplicate (ticker, trade_date)
    dupes = df.duplicated(subset=["ticker", "trade_date"]).sum()
    checks.append((f"no duplicate (ticker, trade_date) [{dupes} found]", dupes == 0))

    # No inf in numeric columns
    numeric_cols = numeric_columns(df)
    n_inf = sum(int(np.isinf(df[col]).sum()) for col in numeric_cols)
    checks.append((f"no inf values [{n_inf} found]", n_inf == 0))

    # No weekend trade_date
    weekend_rows = int((df["trade_date"].dt.dayofweek >= 5).sum())
    checks.append((f"no weekend trade_date [{weekend_rows} found]", weekend_rows == 0))

    # No NaN/negative volume
    vol_nan = int(df["volume"].isna().sum()) if "volume" in df.columns else 0
    vol_neg = int((df["volume"] < 0).sum()) if "volume" in df.columns else 0
    checks.append((f"volume >= 0, no NaN [{vol_nan} NaN, {vol_neg} negative]", vol_nan == 0 and vol_neg == 0))


    failed = 0
    passed = 0
    for label, ok in checks:
        print_check(label, ok)
        if ok:
            passed += 1
        else:
            failed += 1

    print()
    if failed:
        print(f"VALIDATION FAILED: {failed} check(s)")
        if len(ohlc_violations) > 0:
            print("\nOHLC violations (first 10):")
            print(ohlc_violations.head(10).to_string(index=False))
        return False

    print("VALIDATION PASSED")
    return True


def run_one(file_path, args) -> bool:
    """One panel end-to-end. True if its hard validation passed."""
    print_separator()
    print(f"TOP-TRADED TICKERS DATA QUALITY TEST -- {file_path.name}")
    print_separator()
    # -------------------------------------------------------------------------
    # LOAD AND BUILD UNIVERSE
    # -------------------------------------------------------------------------

    df = pd.read_parquet(file_path)
    universe, sub = build_universe(df, top_n=args.top_n)

    print(f"\nFile      : {file_path}")
    print(f"Total rows: {len(df):,}")
    print(f"Total tickers: {df['ticker'].nunique()}")
    print(f"Universe size: {len(universe)} tickers")
    print(f"Universe rows: {len(sub):,}")
    sample_tickers = sorted(universe)[:5]
    print(f"Sample tickers: {', '.join(sample_tickers)}")

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------

    ok = validate(sub, len(universe))

    # -------------------------------------------------------------------------
    # ANOMALY REPORT (informational)
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("ANOMALY REPORT (informational)")
    print_separator()

    stale = check_stale_prices(sub, date_col="trade_date")
    print(f"\nStale price runs (>=5 identical closes, volume>0): {len(stale)}")
    if len(stale):
        print(stale.head(10).to_string(index=False))

    macro_cols = {"selic", "cdi", "ipca", "ipca_daily_equiv", "selic_trend_20d"}
    numeric_cols = [c for c in numeric_columns(sub) if c not in macro_cols]
    outliers = check_outliers_zscore(sub, numeric_cols, date_col="trade_date")
    print(f"\nOutliers (robust z-score > 8): {len(outliers)}")
    if len(outliers):
        print("Top outlier columns:")
        print(outliers["column"].value_counts().head(10).to_string())
        print("\nFirst 10 outlier cells:")
        print(outliers.head(10).to_string(index=False))

    print("\nFundamentals coverage (rows with has_fundamentals=1):")
    print(check_fundamentals_coverage(sub))

    print("\nAdjusted close price quality (informational; old delisted tickers may have adj_close=0):")
    bad_adj_close = sub[sub["adj_close"] <= 0][["ticker", "trade_date", "close", "adj_close"]].drop_duplicates("ticker")
    if len(bad_adj_close) > 0:
        print(f"  {len(bad_adj_close)} tickers with adj_close <= 0:")
        print(bad_adj_close.to_string(index=False))
    else:
        print("  None found.")

    print("\nTrading-calendar gaps (informational; long gaps are delistings/suspensions):")
    gap_violations = check_trading_calendar_gaps(sub, max_gap_days=90)
    if len(gap_violations) > 0:
        print(f"  {len(gap_violations)} tickers with gaps > 90 days:")
        print(gap_violations.head(15).to_string(index=False))
    else:
        print("  None found.")

    # -------------------------------------------------------------------------
    # EXIT
    # -------------------------------------------------------------------------

    if not ok:
        print("\n" + "=" * 80)
        print("HARD VALIDATION FAILED")
        print_separator()
        return False

    # --strict is not passed by run_all.py or ci.yml (see test_final_dataset.py's
    # matching note) -- stays informational until someone deliberately wires it in.
    if args.strict and (len(stale) or len(outliers)):
        print("\n" + "=" * 80)
        print(f"STRICT MODE FAILED: {len(stale)} stale-price rows, "
              f"{len(outliers)} outlier cells")
        print_separator()
        return False

    print("\n" + "=" * 80)
    print("TEST PASSED")
    print_separator()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test data quality on top-traded tickers"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a panel parquet (default: BR + US top-100)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Top N tickers per period (whole + per-year)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on anomaly report findings (stale prices, outliers)",
    )

    args = parser.parse_args()

    if args.file:
        files = [Path(args.file)]
        if not files[0].exists():
            print(f"\nERROR: file not found:\n{files[0]}")
            sys.exit(1)
    else:
        files = [f for f in DEFAULT_FILES if f.exists()]
        for f in DEFAULT_FILES:
            if not f.exists():
                print(f"SKIP  {f.name}: not built yet")

    if not all([run_one(f, args) for f in files]):
        sys.exit(1)


if __name__ == "__main__":
    main()
