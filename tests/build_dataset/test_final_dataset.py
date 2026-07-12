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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "tests"))

from src.build_dataset.build_ml_dataset import (  # noqa: E402
    FILING_LAG_DAYS_QUARTERLY,
    MIN_DETECTABLE_JUMP,
    JUMP_MATCH_TOL,
    EVENT_WINDOW_DAYS,
)
from test_utils import print_header, print_check, print_section_start, print_section_end, print_separator  # noqa: E402

DEFAULT_FILE = ROOT / "data/processed/ml_dataset.parquet"
CORPORATE_EVENTS_FILE = ROOT / "data/raw/corporate_events/corporate_events.parquet"
FUNDAMENTALS_DIR = ROOT / "data/raw/fundamentals"


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

    print()
    print_header("VALIDATION")

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

    # Publication lag (T31): a fundamental only becomes visible once actually
    # filed (CVM DT_RECEB, statutory deadline as fallback) — never before
    filed = df[df["has_fundamentals"] == 1]
    if "fundamentals_available_date" in df.columns:
        early = int((filed["trade_date"] < filed["fundamentals_available_date"]).sum())
        pre_quarter = int((filed["fundamentals_available_date"] < filed["reference_date"]).sum())
        gap = (filed["trade_date"] - filed["reference_date"]).dt.days
        checks.append((f"fundamentals respect filing date [{early} rows early, "
                       f"{pre_quarter} filed pre-quarter-end, min ref-lag {gap.min()}d]",
                       early == 0 and pre_quarter == 0))
    else:  # dataset built before filing-date wiring: statutory floor applies
        gap = (filed["trade_date"] - filed["reference_date"]).dt.days
        checks.append((f"fundamentals respect filing lag [min gap {gap.min()}d]",
                       gap.min() >= FILING_LAG_DAYS_QUARTERLY))

    # No fabricated trading days (T4)
    weekend_rows = int((df["trade_date"].dt.dayofweek >= 5).sum())
    checks.append((f"no weekend trade_date rows [{weekend_rows} found]", weekend_rows == 0))

    # Rows without a filing carry no fundamental values (T9): NaN, never stale/zero-filled
    nf = df[df["has_fundamentals"] == 0]
    leaked = {c: int(nf[c].notna().sum())
              for c in ("pl", "pvp", "roe", "net_income", "market_cap") if c in df.columns}
    checks.append((f"has_fundamentals=0 rows have NaN fundamentals {leaked}",
                   sum(leaked.values()) == 0))

    # asof merge correctness (T5): the merged quarter is the MOST RECENT one
    # available by each trade_date — the per-quarter availability calendar is
    # reconstructed from the dataset's own (reference_date, available_date)
    # pairs, so this catches stale-pick merge bugs under either date source
    mismatches = 0
    avail_col = ("fundamentals_available_date"
                 if "fundamentals_available_date" in df.columns else None)
    if avail_col:
        for t in ("PETR4", "VALE3", "ABEV3"):
            g = df[df["ticker"] == t]
            if g.empty:
                continue
            quarters = (g[["reference_date", avail_col]]
                        .dropna().drop_duplicates().sort_values(avail_col))
            sample = g.sample(min(100, len(g)), random_state=0)
            for _, row in sample.iterrows():
                visible = quarters.loc[
                    quarters[avail_col] <= row["trade_date"], "reference_date"]
                expected = visible.max() if len(visible) else pd.NaT
                actual = row["reference_date"]
                if (pd.isna(expected) != pd.isna(actual)) or \
                        (pd.notna(expected) and expected != actual):
                    mismatches += 1
    checks.append((f"asof merge picks most recent filed quarter (sampled) [{mismatches} mismatches]",
                   mismatches == 0))

    # Corporate events (T8): no split's raw jump ln(1/factor) may survive in
    # log_return — that's a fake ±90-99.99% move from an unadjusted adj_close
    leaks = 0
    if CORPORATE_EVENTS_FILE.exists() and "log_return" in df.columns:
        ev = pd.read_parquet(CORPORATE_EVENTS_FILE)
        ev = ev[ev["factor"] > 0].copy()
        ev["date"] = pd.to_datetime(ev["date"])
        ev = ev[np.abs(np.log(1.0 / ev["factor"])) >= MIN_DETECTABLE_JUMP]
        by_ticker = {t: g for t, g in df.groupby("ticker")}
        for _, e in ev.iterrows():
            g = by_ticker.get(e["ticker"])
            if g is None:
                continue
            # both directions: the audit log's factor convention is inconsistent
            expected = np.log(1.0 / e["factor"])
            lo = e["date"] + pd.Timedelta(days=EVENT_WINDOW_DAYS[0])
            hi = e["date"] + pd.Timedelta(days=EVENT_WINDOW_DAYS[1])
            w = g[g["trade_date"].between(lo, hi)]
            leaks += int(((w["log_return"] - expected).abs() < JUMP_MATCH_TOL).any()
                         or ((w["log_return"] + expected).abs() < JUMP_MATCH_TOL).any())
    checks.append((f"no unadjusted split jumps in log_return [{leaks} events leaking]",
                   leaks == 0))

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
        print_section_start("VALIDATION FAILED")
        print(f"  {failed} check(s) failed")
        print("└─")
        sys.exit(1)
    print_section_start("VALIDATION PASSED")
    print_section_end(passed, failed)


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

    parser.add_argument(
        "--strict",
        action="store_true",
        help="anomaly-report findings (stale prices, outliers) also fail the run"
    )

    args = parser.parse_args()

    file_path = Path(args.file)

    print_header("FINAL DATASET INSPECTION")

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

    # ponytail: strict mode gates on the anomaly report too; default keeps it
    # informational because legitimate extremes (circuit breakers, penny-stock
    # moves) land there alongside real data errors — triage before enabling.
    if args.strict and (len(stale) or len(outliers)):
        print(f"\nSTRICT MODE FAILED: {len(stale)} stale-price rows, "
              f"{len(outliers)} outlier cells")
        sys.exit(1)


if __name__ == "__main__":
    main()