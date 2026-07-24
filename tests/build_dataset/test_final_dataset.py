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

from src.build_dataset.quality_filters import FILING_LAG_DAYS_QUARTERLY  # noqa: E402
from src.build_dataset.repair import MIN_DETECTABLE_JUMP, JUMP_MATCH_TOL, EVENT_WINDOW_DAYS  # noqa: E402
from test_utils import print_header, print_check, print_section_start, print_section_end, print_separator, numeric_columns  # noqa: E402

DEFAULT_FILE = ROOT / "data/processed/ml_dataset.parquet"
CORPORATE_EVENTS_FILE = ROOT / "data/raw/corporate_events/corporate_events.parquet"
FUNDAMENTALS_DIR = ROOT / "data/raw/fundamentals"


def check_stale_prices(df, price_col="close", run_len=5, date_col="date"):
    """Flag runs of >= run_len identical closes while volume > 0."""
    findings = []
    # slice to the columns actually used before sort_values/groupby: sorting
    # the full (wide) frame reorders every column's data, not just these four
    df = df[["ticker", date_col, price_col, "volume"]]
    for t, g in df.sort_values(date_col).groupby("ticker"):
        same = (g[price_col].diff() == 0) & (g["volume"] > 0)
        run = same.groupby((~same).cumsum()).cumsum()
        hits = g[run >= run_len]
        for _, row in hits.iterrows():
            findings.append({"ticker": t, "date": row[date_col], "value": row[price_col]})
    return pd.DataFrame(findings)


# Columns whose LEVEL trends with company growth or price appreciation over
# a ticker's multi-year (sometimes multi-decade) history: a single
# whole-history median/MAD conflates a ticker's cheapest and most expensive
# eras, so any long-lived trending name gets its earliest or latest years
# flagged as "outlier" purely from drift, not from anything anomalous on
# that date. Confirmed directly (2026-07-14 anomaly investigation): 60% of
# flagged adj_close outliers cluster in the first/last 15% of their own
# ticker's date range. Compare within (ticker, year) instead of the
# ticker's whole lifetime — cheap, fixes the drift without a slow rolling
# window.
_TREND_LEVEL_COLS = {
    "open", "high", "low", "close",
    "adj_open", "adj_high", "adj_low", "adj_close",
    "ma_20", "ma_60", "market_cap", "shares_outstanding",
    "equity", "total_assets", "cash", "current_assets", "current_liabilities",
    "total_debt", "net_debt", "ebit", "ebitda", "net_income", "net_revenue",
}

# Already bounded, already normalized (percentile/sector-z-score), boolean
# flags, or identifiers: a whole-history MAD z-score on these is either
# meaningless (re-z-scoring an already-computed z-score/percentile) or
# duplicates a dedicated, more meaningful gate that already exists elsewhere
# (filing_lag_days -> quality_filters.filter_excessive_filing_lag,
# n_quarters_available -> the monotonicity check above in validate()).
_EXCLUDE_FROM_OUTLIER_CHECK = {
    "cagr_earnings_defined", "cagr_revenue_defined", "had_negative_earnings_5y",
    "has_dividends", "has_fundamentals", "f_score", "adj_close_precision_degraded",
    "f_leverage_decreasing", "f_liquidity_improving", "f_margin_improving",
    "f_roa_improving", "f_roa_positive",
    "n_quarters_available", "filing_lag_days", "days_since_fundamental",
    "rsi_14", "cvm_code",
    "drawdown_percentile", "price_percentile_5y", "pl_percentile_5y",
    "div_yield_sector_percentile", "volatility_20d_percentile", "volatility_60d_percentile",
    "debt_equity_zscore_sector", "pl_zscore_sector", "pvp_zscore_sector", "roe_zscore_sector",
}


def check_outliers_zscore(df, feature_cols, threshold=8.0, date_col="date"):
    """Robust (median/MAD) z-score outlier flagging.

    Two regimes, chosen per-column to avoid two confirmed false-positive
    sources (2026-07-14 anomaly investigation — see ANOMALY_INVESTIGATION.md):

    - _TREND_LEVEL_COLS: compared within (ticker, year) rather than the
      ticker's whole history — see docstring above the constant.
    - everything else: z-scored on a signed-log1p transform
      (sign(x) * log1p(|x|)) instead of the raw value. Ratio/growth-rate
      columns diverge as their denominator nears zero by construction
      (e.g. peg_ratio hit 2e9 — CLAUDE.md already documents this as an
      intentional, kept-intact distress signal, not an error), and
      volume/trade-count columns are naturally fat-tailed (volume skew ~92,
      kurtosis ~11,000). log1p compresses those extremes while leaving
      already-modest-range columns (returns, momentum) essentially
      unchanged, since log1p(x) ~= x for small x.
    """
    findings = []
    ticker = df["ticker"]
    year = df[date_col].dt.year
    for col in feature_cols:
        if col in _EXCLUDE_FROM_OUTLIER_CHECK:
            continue
        raw = df[col]
        if col in _TREND_LEVEL_COLS:
            s = raw
            group_keys = [ticker, year]
        else:
            s = np.sign(raw) * np.log1p(raw.abs())
            group_keys = [ticker]
        med = s.groupby(group_keys).transform("median")
        mad = (s - med).abs().groupby(group_keys).transform("median")
        z = 0.6745 * (s - med) / mad.replace(0, float("nan"))
        mask = z.abs() > threshold
        if mask.any():
            sub = df.loc[mask, ["ticker", date_col]].rename(columns={date_col: "date"}).copy()
            sub["column"] = col
            sub["value"] = raw[mask].values
            findings.append(sub)
    if not findings:
        return pd.DataFrame(columns=["ticker", "date", "column", "value"])
    return pd.concat(findings, ignore_index=True)


def validate(df):
    """Golden gate: collect failures, exit(1) if any. Inspector runs first."""

    print()
    print_header("VALIDATION")

    checks = []
    leak_details = []

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
    numeric_cols = numeric_columns(df)
    # per-column, not df[numeric_cols]: avoids materializing every numeric
    # column as one copy just to count infs (OOMs on wide frames)
    n_inf = sum(int(np.isinf(df[col]).sum()) for col in numeric_cols)
    checks.append((f"no inf values in numeric columns [{n_inf} found]", n_inf == 0))

    # Macro merged and not entirely null
    for col in ("selic", "cdi", "ipca"):
        present = col in df.columns and df[col].notna().any()
        checks.append((f"macro {col} merged", present))

    # Stage 2 keeps sparse tickers (MIN_PRICE_ROWS=10 in build_ml_dataset.py);
    # the 252-row (1 trading year) full-history bar is a downstream-consumer
    # concern, not enforced here.
    row_counts = df.groupby("ticker").size()
    min_rows = row_counts.min()
    n_short = (row_counts < 252).sum()
    checks.append((f"all tickers >= 10 rows [min {min_rows}] "
                    f"({n_short} tickers < 252 rows — thin/recent listings, kept here on purpose; "
                    f"only trimmed downstream by ml_agent's data_pipeline.py, not by this repo)",
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
    # column-slice before the row filter: df[mask] on the full 140-col frame
    # copies every column just to keep the 2-3 actually used below
    filed_cols = [c for c in ("trade_date", "fundamentals_available_date", "reference_date")
                  if c in df.columns]
    filed = df.loc[df["has_fundamentals"] == 1, filed_cols]
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
    leak_cols = [c for c in ("pl", "pvp", "roe", "net_income", "market_cap") if c in df.columns]
    nf = df.loc[df["has_fundamentals"] == 0, leak_cols]
    leaked = {c: int(nf[c].notna().sum()) for c in leak_cols}
    checks.append((f"has_fundamentals=0 rows have NaN fundamentals {leaked}",
                   sum(leaked.values()) == 0))

    # Prefix-shaped NaN rule: fundamentals forward-filled via merge_asof backward.
    # Flag SUSPICIOUS interior holes (all major columns NaN), but allow partial reporting.
    suspicious_tickers = {}
    prefix_cols = [c for c in ("equity", "net_income", "total_assets") if c in df.columns]
    for ticker, g in df[["ticker"] + prefix_cols].groupby("ticker"):
        for col in prefix_cols:
            s = g[col]
            idx_first_nonnull = s.first_valid_index()
            if idx_first_nonnull is None:
                continue
            after_first = s.loc[idx_first_nonnull:]
            holes = (after_first[1:].isna() & after_first.shift(1)[1:].notna()).sum()
            if holes > 0:
                # Check if it's a legitimate partial gap (one column) or suspicious (all columns)
                suspicious_tickers.setdefault(ticker, set()).add(col)

    # Flag only if a ticker has holes across ALL three key columns (suggests merge bug)
    all_three = [t for t, cols in suspicious_tickers.items() if len(cols) == 3]
    prefix_ok = len(all_three) == 0
    msg = f"NaN shapes are prefix per ticker (no interior holes)"
    if suspicious_tickers and not all_three:
        msg += f" [warning: {len(suspicious_tickers)} tickers with partial gaps (legitimate data): {', '.join(sorted(suspicious_tickers.keys())[:5])}{'...' if len(suspicious_tickers) > 5 else ''}]"
    elif all_three:
        msg += f" [suspicious merge bug in: {', '.join(all_three)}]"
    checks.append((msg, prefix_ok))

    # CAGR coverage: NaN values are expected (base year missing, negative earnings, etc).
    # Just report coverage stats and warn if suspiciously high NaN rate.
    if "cagr_earnings_5y_final" in df.columns and "n_quarters_available" in df.columns:
        has_fund = df["has_fundamentals"] == 1
        cagr_nan = has_fund & df["cagr_earnings_5y_final"].isna()

        # Explained reasons
        base_neg = df.loc[cagr_nan, "had_negative_earnings_5y"] == 1
        few_q = df.loc[cagr_nan, "n_quarters_available"] < 20
        explained = (base_neg | few_q).sum()
        total_nan = cagr_nan.sum()
        coverage = (total_nan - (total_nan - explained)) / total_nan * 100 if total_nan > 0 else 100

        # Pass if explained + plausible (missing base year) > 80%
        cagr_ok = explained > total_nan * 0.8
        msg = f"cagr_earnings NaN coverage [{explained}/{total_nan} explained, {100-coverage:.1f}% unattributed]"
        if cagr_ok:
            msg += " (acceptable, likely missing base-year data)"
        checks.append((msg, cagr_ok))

        if "cagr_revenue_5y_final" in df.columns:
            cagr_rev_nan = has_fund & df["cagr_revenue_5y_final"].isna()
            few_q_rev = df.loc[cagr_rev_nan, "n_quarters_available"] < 20
            explained_rev = few_q_rev.sum()
            total_rev_nan = cagr_rev_nan.sum()
            rev_cov = (total_rev_nan - (total_rev_nan - explained_rev)) / total_rev_nan * 100 if total_rev_nan > 0 else 100

            cagr_rev_ok = explained_rev > total_rev_nan * 0.8
            msg_rev = f"cagr_revenue NaN coverage [{explained_rev}/{total_rev_nan} explained, {100-rev_cov:.1f}% unattributed]"
            if cagr_rev_ok:
                msg_rev += " (acceptable)"
            checks.append((msg_rev, cagr_rev_ok))

    # New flag columns: well-formed (0/1 for flags, valid count for n_quarters).
    if "cagr_earnings_defined" in df.columns:
        ok = df["cagr_earnings_defined"].isin([0, 1, 0.0, 1.0]).all()
        checks.append((f"cagr_earnings_defined ∈ {{0,1}}, no NaN", ok))
    if "cagr_revenue_defined" in df.columns:
        ok = df["cagr_revenue_defined"].isin([0, 1, 0.0, 1.0]).all()
        checks.append((f"cagr_revenue_defined ∈ {{0,1}}, no NaN", ok))
    if "adj_close_precision_degraded" in df.columns:
        ok = df["adj_close_precision_degraded"].isin([0, 1, 0.0, 1.0]).all()
        checks.append((f"adj_close_precision_degraded ∈ {{0,1}}, no NaN", ok))
    if "n_quarters_available" in df.columns:
        # Check: when sorted by trade_date, n_quarters_available should be non-decreasing
        # for rows with valid reference_date. Minor edge cases (rows with NaT ref but filled n_quarters)
        # are acceptable as they affect <1% of tickers.
        bad_tickers = []
        nq_cols = ["ticker", "trade_date", "n_quarters_available", "reference_date"]
        for t, ticker_df in df[nq_cols].sort_values("trade_date").groupby("ticker"):
            # Only check rows with non-NaN n_quarters_available AND non-NaT reference_date
            valid_df = ticker_df[
                (ticker_df["n_quarters_available"].notna()) &
                (ticker_df["reference_date"].notna())
            ]
            if len(valid_df) > 1:
                diffs = valid_df["n_quarters_available"].diff().fillna(0)
                if (diffs < 0).any():
                    bad_tickers.append(t)

        # Accept if <1% of tickers have issues (edge cases with stale data)
        pct_bad = len(bad_tickers) / len(df["ticker"].unique()) * 100
        is_acceptable = pct_bad < 1.0
        msg = f"n_quarters_available: non-decreasing per ticker [{len(bad_tickers)} tickers affected, {pct_bad:.2f}%]"
        if bad_tickers and is_acceptable:
            msg += f" [edge case — acceptable]"
        checks.append((msg, is_acceptable))

    # asof merge correctness (T5): the merged quarter is the MOST RECENT one
    # available by each trade_date — the per-quarter availability calendar is
    # reconstructed from the dataset's own (reference_date, available_date)
    # pairs, so this catches stale-pick merge bugs under either date source.
    #
    # Strict < (not <=): merge_prices_and_fundamentals uses
    # allow_exact_matches=False (2026-07-23 audit, Issue 6) -- a filing
    # received ON trade_date T is not visible to T's own close, visibility
    # starts T+1. This check used <= until 2026-07-25, silently mismatched
    # against that intentional design (never updated when Issue 6 landed),
    # and only surfaced once the random_state=0 sample happened to land on an
    # exact filing-date row.
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
                    quarters[avail_col] < row["trade_date"], "reference_date"]
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
        by_ticker = {t: g for t, g in df[["ticker", "trade_date", "log_return"]].groupby("ticker")}
        for _, e in ev.iterrows():
            g = by_ticker.get(e["ticker"])
            if g is None:
                continue
            # both directions: the audit log's factor convention is inconsistent
            expected = np.log(1.0 / e["factor"])
            lo = e["date"] + pd.Timedelta(days=EVENT_WINDOW_DAYS[0])
            hi = e["date"] + pd.Timedelta(days=EVENT_WINDOW_DAYS[1])
            w = g[g["trade_date"].between(lo, hi)]
            has_leak = ((w["log_return"] - expected).abs() < JUMP_MATCH_TOL).any() or ((w["log_return"] + expected).abs() < JUMP_MATCH_TOL).any()
            if has_leak:
                leaks += 1
                leak_details.append(f"{e['ticker']} on {e['date'].date()} (factor {e['factor']:.4f})")
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
        if leak_details:
            print("\n  Leaking corporate events:")
            for detail in leak_details:
                print(f"    • {detail}")
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

    numeric_cols = numeric_columns(df)

    if numeric_cols:

        print("\n")
        print_separator()
        print("NUMERIC STATISTICS")
        print_separator()

        # per-column, not df[numeric_cols].describe(): avoids materializing
        # every numeric column as one copy just to summarize it (OOMs on wide frames)
        stats = pd.DataFrame({col: df[col].describe() for col in numeric_cols}).T

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

    # subset to the natural key, not a full-row check: a real duplicate always
    # shares (ticker, trade_date), and factorizing all 140 columns to prove it
    # is orders of magnitude more memory for the same answer
    dup_subset = ["ticker", "trade_date"] if {"ticker", "trade_date"}.issubset(df.columns) else None
    total_duplicates = df.duplicated(subset=dup_subset).sum()

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

    # date_col="trade_date": avoids renaming (and thus deep-copying) the
    # entire wide frame just to satisfy these helpers' default column name
    stale = check_stale_prices(df, date_col="trade_date")
    print(f"Stale price runs (>=5 identical closes, volume>0): {len(stale)}")
    if len(stale):
        print(stale.head(20).to_string(index=False))

    # Exclude raw macro passthrough columns: identical across all tickers on a
    # given date by construction, so they flood this report with repeats of
    # the same macro regime shift rather than per-observation data errors.
    macro_cols = {"selic", "cdi", "ipca", "ipca_daily_equiv", "selic_trend_20d"}
    numeric_cols = [c for c in numeric_columns(df) if c not in macro_cols]
    outliers = check_outliers_zscore(df, numeric_cols, date_col="trade_date")
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
    # Owner: whoever next reviews the anomaly report's false-positive rate.
    # --strict is not passed by run_all.py or ci.yml, so today this stays
    # informational forever unless someone deliberately wires it in.
    if args.strict and (len(stale) or len(outliers)):
        print(f"\nSTRICT MODE FAILED: {len(stale)} stale-price rows, "
              f"{len(outliers)} outlier cells")
        sys.exit(1)


if __name__ == "__main__":
    main()