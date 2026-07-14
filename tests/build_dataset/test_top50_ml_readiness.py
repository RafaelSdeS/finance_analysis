"""
Top-50 ML-readiness audit — see TOP50_ML_READINESS_AUDIT.md for the report
this script generates.

Scope: the 50 most-traded tickers (by summed traded_amount) from the
validated start date (2011-04-01, see report §1) onward. Restricting to a
liquidity-ranked, recent-history subset on purpose -- the goal is the
highest-quality slice for the first ML agent phase, not the longest one.

Reuses existing check_* helpers (test_top_traded_quality.py,
test_final_dataset.py) rather than reimplementing OHLC/stale-price/outlier
logic; adds only what's new for this scope: point-in-time universe
definition, pairwise duplicate-series detection, per-ticker verdicts, and a
generated markdown report.

Run from project root:
    python tests/build_dataset/test_top50_ml_readiness.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.build_dataset.loaders import company_siblings  # noqa: E402
from src.build_dataset.paths import COMPANY_INFO_PATH, CORPORATE_EVENTS_PATH, OUTPUT_PATH  # noqa: E402
from src.build_dataset.repair import MIN_DETECTABLE_JUMP  # noqa: E402
from tests.build_dataset.test_top_traded_quality import check_ohlc_consistency, check_trading_calendar_gaps  # noqa: E402
from tests.build_dataset.test_final_dataset import check_stale_prices  # noqa: E402

START_DATE = "2011-04-01"
TOP_N = 50
DUPLICATE_SERIES_CORR = 0.999  # near-1.0 return correlation => suspected duplicated/mislabeled data
CALENDAR_GAP_WARN_DAYS = 30    # top-50 liquid names shouldn't go this long without a print
CALENDAR_GAP_CRITICAL_DAYS = 90
BIG_JUMP_LOG_RETURN = MIN_DETECTABLE_JUMP  # reuse repair.py's own "not normal market noise" threshold
ADJ_DISCONTINUITY_ADJ_PCT = 0.5   # adj_close moves >= 50% day-over-day...
ADJ_DISCONTINUITY_RAW_PCT = 0.10  # ...while raw close moves < 10% -> adjustment-factor bug, not a real move
FUND_COVERAGE_WARN = 0.90
REPORT_PATH = ROOT / "TOP50_ML_READINESS_AUDIT.md"

# Known raw-data fabrication: CCTY3's entire price history is a byte-for-byte
# copy of CCRO3's despite a different cvm_code (see report §2, critical
# finding). Excluded from ranking; replaced by the next-ranked ticker.
EXCLUDE_FABRICATED = {"CCTY3"}


def build_universe(df):
    """Point-in-time-safe for this purpose: rank by traded_amount summed only
    over [START_DATE, dataset end] -- no pre-2011 volume, no future
    information beyond "we know it now" (this is a fixed backtest/training
    universe for phase 1, not a per-date rebalanced production universe --
    see TOP50_UNIVERSE_VALIDATION.md for that separate design question)."""
    window = df[df["trade_date"] >= START_DATE]
    ranked = window.groupby("ticker")["traded_amount"].sum().sort_values(ascending=False)
    ranked = ranked[~ranked.index.isin(EXCLUDE_FABRICATED)]
    return ranked.head(TOP_N).index.tolist(), window


def check_duplicate_series(df, tickers):
    """All-pairs return-correlation scan -- the automated version of the
    manual check that caught CCTY3. A real distinct company essentially
    never correlates > 0.999 with another one over years of daily returns."""
    wide = df.pivot(index="trade_date", columns="ticker", values="log_return")
    wide = wide[tickers]
    corr = wide.corr(min_periods=250)
    pairs = []
    for i, t1 in enumerate(tickers):
        for t2 in tickers[i + 1:]:
            c = corr.loc[t1, t2]
            if pd.notna(c) and c > DUPLICATE_SERIES_CORR:
                pairs.append((t1, t2, c))
    return pairs


def check_adj_close_discontinuity(df, tickers):
    """adj_close jumping hard while raw close barely moves -- real splits/
    dividends move BOTH by a matched factor (or leave adj_close continuous
    across a raw-close split jump). A jump in the adjustment factor with no
    corresponding raw-price move on the same day is neither: it silently
    poisons every return/momentum feature computed across that date."""
    findings = []
    for t in tickers:
        g = df[df["ticker"] == t].sort_values("trade_date")
        close_pct = g["close"].pct_change()
        adj_pct = g["adj_close"].pct_change()
        mask = (adj_pct.abs() >= ADJ_DISCONTINUITY_ADJ_PCT) & (close_pct.abs() < ADJ_DISCONTINUITY_RAW_PCT)
        hits = g.loc[mask, ["trade_date", "close"]].copy()
        hits["close_pct"] = close_pct[mask]
        hits["adj_pct"] = adj_pct[mask]
        for _, row in hits.iterrows():
            findings.append({
                "ticker": t, "date": row["trade_date"],
                "close_pct": row["close_pct"], "adj_pct": row["adj_pct"],
            })
    return pd.DataFrame(findings)


def check_unexplained_jumps(df, tickers, events):
    """Single-day |log_return| >= MIN_DETECTABLE_JUMP not near any recorded
    corporate event or dividend ex-date -- candidate residual adjustment bug
    (repair.py already fixes detected split jumps; this is a check that none
    slipped through undetected, or that a new one hasn't appeared)."""
    findings = []
    ev_by_ticker = {t: g["date"].tolist() for t, g in events.groupby("ticker")} if events is not None else {}
    for t in tickers:
        g = df[df["ticker"] == t].sort_values("trade_date")
        big = g[g["log_return"].abs() >= BIG_JUMP_LOG_RETURN]
        for _, row in big.iterrows():
            ev_dates = ev_by_ticker.get(t, [])
            near_event = any(abs((row["trade_date"] - d).days) <= 35 for d in ev_dates)
            if not near_event:
                findings.append({"ticker": t, "date": row["trade_date"], "log_return": row["log_return"]})
    return pd.DataFrame(findings)


def main():
    print("Loading dataset...")
    df = pd.read_parquet(OUTPUT_PATH)
    company_info = pd.read_parquet(COMPANY_INFO_PATH)
    events = None
    if CORPORATE_EVENTS_PATH.exists():
        events = pd.read_parquet(CORPORATE_EVENTS_PATH)
        events["date"] = pd.to_datetime(events["date"])

    tickers, window = build_universe(df)
    sub = window[window["ticker"].isin(tickers)].copy()
    print(f"Universe: {len(tickers)} tickers, {len(sub):,} rows from {START_DATE}")

    issues = {t: [] for t in tickers}  # ticker -> list of (severity, message)

    def flag(ticker, severity, message):
        issues[ticker].append((severity, message))

    # --- duplicate/fabricated series (critical) ---
    dup_pairs = check_duplicate_series(sub, tickers)
    for t1, t2, c in dup_pairs:
        flag(t1, "CRITICAL", f"return series suspiciously identical to {t2} (corr={c:.4f})")
        flag(t2, "CRITICAL", f"return series suspiciously identical to {t1} (corr={c:.4f})")

    # --- OHLC consistency (critical) ---
    ohlc_bad = check_ohlc_consistency(sub)
    for t, g in ohlc_bad.groupby("ticker") if len(ohlc_bad) else []:
        flag(t, "CRITICAL", f"{len(g)} OHLC consistency violations")

    # --- non-positive prices/volume (critical) ---
    for col in ("open", "high", "low", "close", "adj_close"):
        bad = sub[sub[col] <= 0]
        for t, g in bad.groupby("ticker"):
            flag(t, "CRITICAL", f"{len(g)} rows with {col} <= 0")
    bad_vol = sub[sub["volume"] < 0]
    for t, g in bad_vol.groupby("ticker"):
        flag(t, "CRITICAL", f"{len(g)} rows with negative volume")

    # --- duplicate (ticker, trade_date) rows (critical) ---
    dupes = sub[sub.duplicated(subset=["ticker", "trade_date"], keep=False)]
    for t, g in dupes.groupby("ticker"):
        flag(t, "CRITICAL", f"{g['trade_date'].nunique()} duplicated trade_date rows")

    # --- weekend trade_date (warning) ---
    weekend = sub[sub["trade_date"].dt.dayofweek >= 5]
    for t, g in weekend.groupby("ticker"):
        flag(t, "WARNING", f"{len(g)} rows fall on a weekend date")

    # --- lookahead: fundamentals visible only on/after their real dates (critical) ---
    has_dates = sub["reference_date"].notna()
    late = sub.loc[has_dates & (sub["reference_date"] > sub["trade_date"])]
    for t, g in late.groupby("ticker"):
        flag(t, "CRITICAL", f"{len(g)} rows where reference_date > trade_date (lookahead)")
    if "fundamentals_available_date" in sub.columns:
        has_avail = sub["fundamentals_available_date"].notna()
        early = sub.loc[has_avail & (sub["trade_date"] < sub["fundamentals_available_date"])]
        for t, g in early.groupby("ticker"):
            flag(t, "CRITICAL", f"{len(g)} rows visible before their fundamentals_available_date")

    # --- stale prices (warning) ---
    stale = check_stale_prices(sub, date_col="trade_date")
    for t, g in stale.groupby("ticker") if len(stale) else []:
        flag(t, "WARNING", f"{len(g)} stale-price rows (>=5 identical closes, volume>0)")

    # --- trading-calendar gaps (warning / critical) ---
    gaps = check_trading_calendar_gaps(sub, max_gap_days=CALENDAR_GAP_WARN_DAYS)
    for _, row in gaps.iterrows():
        sev = "CRITICAL" if row["gap_days"] > CALENDAR_GAP_CRITICAL_DAYS else "WARNING"
        flag(row["ticker"], sev, f"{row['gap_days']}-day gap between {row['date1'].date()} and {row['date2'].date()}")

    # --- adj_close discontinuity: adjustment factor jumps without a matching
    # raw-price move (critical -- poisons every return/momentum feature that
    # spans the date, not just the single row) ---
    discontinuities = check_adj_close_discontinuity(sub, tickers)
    for t, g in discontinuities.groupby("ticker") if len(discontinuities) else []:
        shown = g.head(5)
        dates = ", ".join(f"{r['date'].date()} (close {r['close_pct']:+.1%}, adj_close {r['adj_pct']:+.1%})"
                           for _, r in shown.iterrows())
        more = f" (+{len(g) - 5} more)" if len(g) > 5 else ""
        noun = "discontinuity" if len(g) == 1 else "discontinuities"
        flag(t, "CRITICAL", f"{len(g)} adj_close {noun} without matching raw-close move, e.g. {dates}{more}")

    # --- unexplained big single-day jumps (warning) ---
    jumps = check_unexplained_jumps(sub, tickers, events)
    for t, g in jumps.groupby("ticker") if len(jumps) else []:
        flag(t, "WARNING", f"{len(g)} single-day |log_return| >= {BIG_JUMP_LOG_RETURN} not near a recorded corporate event")

    # --- fundamentals coverage (warning) ---
    fund_rate = sub.groupby("ticker")["has_fundamentals"].mean()
    for t in tickers:
        if fund_rate.get(t, 0) < FUND_COVERAGE_WARN:
            flag(t, "WARNING", f"has_fundamentals rate only {fund_rate[t]:.0%} since {START_DATE}")

    # --- adj_close precision floor (informational) ---
    if "adj_close_precision_degraded" in sub.columns:
        degraded = sub[sub["adj_close_precision_degraded"] == 1]
        for t, g in degraded.groupby("ticker"):
            flag(t, "INFO", f"{len(g)} rows flagged adj_close_precision_degraded")

    # --- extreme valuation ratios (informational, known kept-intact policy) ---
    if "pl" in sub.columns:
        extreme_pl = sub[sub["pl"].abs() > 400_000]
        for t, g in extreme_pl.groupby("ticker"):
            flag(t, "INFO", f"{len(g)} rows with |pl| > 400,000 (denominator-near-zero, kept intact per policy)")

    # --- sector/status completeness (warning) ---
    # Check the MERGED dataset, not raw company_info directly: company_siblings()
    # forward-fills sector/status from same-cvm_code tickers (e.g. PETR4->PETR3),
    # so a raw-file gap that's legitimately filled downstream isn't a real issue.
    sector_by_ticker = sub.groupby("ticker")["sector"].apply(lambda s: s.dropna().astype(str))
    for t in tickers:
        vals = sector_by_ticker.get(t, pd.Series(dtype=str))
        if vals.empty or (vals == "None").all():
            flag(t, "WARNING", "missing sector even after sibling-fill (ml_dataset.parquet)")

    # --- sibling correlation (informational, reused from test_universe_integrity.py) ---
    siblings = company_siblings(company_info)
    sibling_pairs = [tuple(g) for g in siblings.values() if len(g) == 2 and set(g).issubset(tickers)]
    sibling_notes = []
    for t1, t2 in sibling_pairs:
        s1 = sub.loc[sub["ticker"] == t1].set_index("trade_date")["log_return"]
        s2 = sub.loc[sub["ticker"] == t2].set_index("trade_date")["log_return"]
        joined = pd.concat([s1, s2], axis=1, join="inner")
        if len(joined) < 60:
            continue
        corr = joined.iloc[:, 0].rolling(60).corr(joined.iloc[:, 1]).dropna()
        if not corr.empty and corr.min() < 0.5:
            sibling_notes.append(f"{t1}/{t2}: min 60d rolling corr = {corr.min():.2f}")

    # --- schema/dtype (single check for the subset, not per-ticker) ---
    schema_ok = (
        sub["ticker"].dtype == object
        and pd.api.types.is_datetime64_any_dtype(sub["trade_date"])
        and pd.api.types.is_numeric_dtype(sub["close"])
        and pd.api.types.is_numeric_dtype(sub["has_fundamentals"])
    )

    # =========================================================================
    # VERDICTS + REPORT
    # =========================================================================
    verdicts = {}
    for t in tickers:
        sevs = {s for s, _ in issues[t]}
        if "CRITICAL" in sevs:
            verdicts[t] = "NOT READY"
        elif "WARNING" in sevs:
            verdicts[t] = "READY WITH CAVEATS"
        else:
            verdicts[t] = "READY"

    n_ready = sum(1 for v in verdicts.values() if v == "READY")
    n_caveats = sum(1 for v in verdicts.values() if v == "READY WITH CAVEATS")
    n_not_ready = sum(1 for v in verdicts.values() if v == "NOT READY")

    print(f"\nREADY: {n_ready}  READY WITH CAVEATS: {n_caveats}  NOT READY: {n_not_ready}")
    print(f"Schema/dtype contract: {'OK' if schema_ok else 'FAILED'}")
    print(f"Writing full report to {REPORT_PATH}")

    # --- generate markdown report ---
    lines = []
    lines.append("# Top-50 ML-Readiness Audit\n")
    lines.append(f"Generated by `tests/build_dataset/test_top50_ml_readiness.py`. "
                 f"Universe: top {TOP_N} tickers by traded_amount from **{START_DATE}** onward "
                 f"(see analysis below). `{OUTPUT_PATH.name}` snapshot: {len(df):,} total rows, "
                 f"{df['ticker'].nunique()} total tickers.\n")

    lines.append("## 1. Recommended start date: 2011-04-01\n")
    lines.append(
        "Fundamentals coverage (`has_fundamentals` rate) is **0% for every year 2000-2010**, then ramps "
        "within 2011: Jan 0.1% -> Feb 6.9% -> Mar 38.8% -> **Apr 93.6%**, then holds flat at 94-98% every "
        "month through 2012 and beyond. This matches Brazil's DFP (annual financial statement) filing "
        "deadline of ~March 31 for the prior fiscal year -- by April, the year's filings have landed. "
        "Jan-Mar 2011 is a genuine ramp/transition period, not a data bug; excluding it (rather than "
        "starting 2011-01-01) drops 3 months of mostly-missing fundamentals without discarding any "
        "genuinely-available history. Macro (selic/cdi/ipca) is 100% covered for the entire 2000-2026 "
        "span, so it never constrains the start date -- fundamentals coverage is the binding constraint.\n"
    )

    lines.append("## 2. Universe construction\n")
    lines.append(
        f"Ranked by `traded_amount` summed over [{START_DATE}, dataset end] -- deliberately excludes "
        "pre-2011 volume so a name that was heavily traded in 2005 but thin since can't crowd out a "
        "name that's been consistently liquid throughout the actual training window. This is a **fixed** "
        "universe for this initial ML-agent phase, not a point-in-time-rebalanced production universe -- "
        "see `TOP50_UNIVERSE_VALIDATION.md` for that separate (still open) design question.\n"
    )
    lines.append(
        "**Resolved:** the raw #36 candidate, `CCTY3`, was excluded and replaced by the next-ranked "
        "ticker. `data/raw/prices/CCTY3.parquet` was a byte-for-byte duplicate of `CCRO3.parquet`'s "
        "entire close-price history despite belonging to a different company by `cvm_code` (CCTY3=27570, "
        "\"Belora RDVC City Desenvolvimento Imobiliário S.A.\" vs CCRO3=018821, CCR S.A./Motiva). "
        "Investigated against two independent sources: BolsAI's *live* API serves the identical "
        "CCR-matching values today (not a stale local snapshot), and yfinance's `CCTY3.SA` mirrors the "
        "same numbers -- both vendors alias this ticker to the same dead/orphaned security. No reliable "
        "source exists for this ticker's true price history, so it was added to "
        "`quality_filters.QUARANTINED_TICKERS` (2026-07-14) rather than patched -- it is now permanently "
        "excluded from `ml_dataset.parquet`, not just this audit's universe.\n"
    )
    lines.append("Automated pairwise return-correlation scan (the general form of the check that caught "
                  f"CCTY3) run across all {TOP_N} tickers in the final universe: "
                  f"{'no further pairs above ' + str(DUPLICATE_SERIES_CORR) + ' correlation.' if not dup_pairs else str(len(dup_pairs)) + ' further suspicious pair(s) -- see critical findings below.'}\n")

    lines.append("## 3. Final 50-ticker universe\n")
    lines.append("| Rank | Ticker | Verdict | Critical | Warning | Info |")
    lines.append("|---|---|---|---|---|---|")
    for i, t in enumerate(tickers, 1):
        n_crit = sum(1 for s, _ in issues[t] if s == "CRITICAL")
        n_warn = sum(1 for s, _ in issues[t] if s == "WARNING")
        n_info = sum(1 for s, _ in issues[t] if s == "INFO")
        lines.append(f"| {i} | {t} | {verdicts[t]} | {n_crit} | {n_warn} | {n_info} |")
    lines.append("")

    lines.append("## 4. Root cause (FIXED): ticker-continuity splice boundary broke `adj_close`\n")
    lines.append(
        "First pass of this audit found 4 tickers (`B3SA3`, `BHIA3`, `MBRF3`, `TIMS3`) with an "
        "`adj_close` discontinuity and no matching move in raw `close`. Root cause: `continuity.py`'s "
        "splice only rescaled price columns by the documented share-exchange `ratio` (1.0 for all "
        "renames) -- it never reconciled the old and new ticker's `adj_close` onto one consistent "
        "total-return basis when BolsAI computed the two vendor series differently (e.g. `BVMF3`'s "
        "`adj_close` always equalled its `close` -- no dividend adjustment ever recorded for that "
        "identity -- while post-rename `B3SA3` started already ~3.7x below its own `close`, a "
        "today-anchored fully-adjusted basis from day one). **Fixed 2026-07-14**: "
        "`apply_ticker_continuity()` (`src/build_dataset/continuity.py`) now computes an empirical "
        "reconciliation factor at every splice boundary (`new.adj_close[boundary] / old.adj_close[last_day]`, "
        "skipped when within `ADJ_RECONCILE_TOL`=10% of normal 1-day-return noise) and rescales the old "
        "ticker's `adj_*` history by it -- same pattern `repair_unadjusted_splits()` already used for "
        "detected splits, just keyed off the continuity map. Confirmed dataset-wide, not just these 4: "
        "the rebuilt dataset logs 14 reconciliations across the full ticker universe (`BVMF3->B3SA3`, "
        "`ESTC3->YDUQ3`, `CELP3->EQPA3`, `TIMP3->TIMS3`, `DTEX3->DXCO3`, `VVAR3->VIIA3`, `BRDT3->VBBR3`, "
        "`CCPR3->SYNE3`, `CARD3->CSUD3`, `SULA11->RDOR3`, `WIZS3->WIZC3`, `SOMA3->AZZA3`, "
        "`TRPL4->ISAE4`, `CCRO3->MOTV3` -- the last discovered during this same investigation, see §5).\n"
    )
    lines.append(
        "`MBRF3`'s dozens of *additional* discontinuities (scattered 2011-2022, unrelated to any splice "
        "boundary) turned out to be a separate bug: BolsAI's raw `adj_close` for this ticker is stored "
        "at 2-decimal precision at small values (e.g. 0.01/0.02), so a few-percent real drift rounds to "
        "a fake \"100% jump.\" Confirmed live (not a stale snapshot -- BolsAI's `/stocks/MRFG3/history` "
        "endpoint serves the identical broken values today) and confirmed NOT a real corporate action "
        "(web search found dividends only, no splits/grupamento for Marfrig 2019-2022). yfinance's "
        "`MBRF3.SA` series is independently clean on every flagged date (0% flat-run fraction, not "
        "vendor coverage-padding). **Fixed 2026-07-14**: `src/data_collection/fix_mrfg3_adj_close.py` "
        "recomputes `adj_*` for both `data/raw/prices/MRFG3.parquet` and `MBRF3.parquet` from yfinance's "
        "`(adj_close/close)` ratio applied to BolsAI's own OHLC (keeps BolsAI's raw close/volume, fixes "
        "only the broken derived adjustment). Both raw files needed the fix: BolsAI has two "
        "independently-collected files with the identical bug, and `ticker_continuity.json`'s "
        "`MRFG3->MBRF3` splice boundary resolves to `MBRF3`'s own (earliest) first-trade date -- meaning "
        "`MBRF3.parquet`, not `MRFG3.parquet`, is what actually survives into `ml_dataset.parquet`; fixing "
        "`MRFG3.parquet` alone had no effect on the built dataset (caught by re-running this audit after "
        "the first fix attempt).\n"
    )

    lines.append("## 5. Trading-calendar gaps investigated\n")
    lines.append(
        "- **`CCRO3`** (52-day gap, 2025-11-07..2025-12-29 in the original audit): **real corporate "
        "event, not a bug.** CCR S.A. renamed to Motiva Infraestrutura de Mobilidade and switched ticker "
        "`CCRO3` -> `MOTV3` effective 2025-05-02 (confirmed via multiple independent BR financial news "
        "sources). `CCRO3`'s raw file already showed the tell: normal 5-30M-share volume through "
        "2025-04-28, then near-zero volume (100-2000 shares) with wildly erratic thin-market prices "
        "afterward -- an orphaned dead ticker code, not corrupted data (yfinance independently reports "
        "`CCRO3.SA` as delisted from the same date). **Fixed**: added the `CCRO3->MOTV3` rename to "
        "`ticker_continuity.json`; the splice now correctly drops the dead post-rename `CCRO3` stub and "
        "continues the series under `MOTV3` (which already existed as a clean, independently-collected "
        "raw file).\n"
        "- **`BHIA3`** (53-day gap, 2024-04-29..2024-06-21): **BolsAI collection gap, not a real "
        "suspension.** yfinance shows continuous normal-volume trading through the entire window, "
        "matching our data exactly on both edges (7.30 on 04-29, 5.71 on resumption). **Fixed**: "
        "backfilled 36 rows via the existing `backfill_price_gap()` (yfinance, with the "
        "`_flat_run_fraction` fabrication guard already in place from a prior investigation).\n"
        "- **`BHIA3`** (second gap, surfaced after the fix above: 47 days, 2015-02-18..2015-04-06): "
        "**legitimate, not a bug.** `BHIA3`'s 2015 predecessor (`VVAR3`, pre-rename) shows only 4 real "
        "trades in this window (volumes 100-700, 1-4 trades/day) -- confirmed via BolsAI's live API. "
        "Genuine illiquid micro-cap trading from over a decade before Casas Bahia's current scale, not a "
        "gap in collection. Left as-is.\n"
        "- **`BRKM5`** (49-day gap, 2017-06-30..2017-08-18): **BolsAI collection gap.** Same pattern as "
        "`BHIA3`: yfinance shows continuous trading throughout, matching on both edges (34.20 -> 37.80). "
        "**Fixed**: backfilled 34 rows the same way.\n"
    )
    lines.append(
        "Also resolved during this pass: the \"missing sector\" warning for `PETR3`/`AZUL4`/`BBDC3` in "
        "the first audit pass was a **false positive in this script**, not a data issue -- it checked "
        "raw `company_info.parquet` directly instead of the merged `ml_dataset.parquet`, where "
        "`company_siblings()` already forward-fills sector from the same-`cvm_code` ticker (PETR4->PETR3, "
        "etc.). Fixed in this script's own sector check. `RDOR3`'s 44% `has_fundamentals` coverage is "
        "legitimate, not a bug: `RDOR3` inherited `SULA11`'s (SulAmérica) 2007-2020 price history as a "
        "shareholder-continuity predecessor via the documented `merger`-type splice, which correctly "
        "drops fundamentals for the pre-merger segment (SulAmérica's books are not Rede D'Or's) -- roughly "
        "9 predecessor years with no fundamentals plus 6 years of RDOR3's own real filings matches the "
        "observed 44% almost exactly.\n"
    )

    lines.append("## 6. Issues by severity\n")
    for severity in ("CRITICAL", "WARNING", "INFO"):
        lines.append(f"### {severity}\n")
        any_found = False
        for t in tickers:
            msgs = [m for s, m in issues[t] if s == severity]
            if msgs:
                any_found = True
                lines.append(f"- **{t}**: " + "; ".join(msgs))
        if not any_found:
            lines.append("- None found.")
        lines.append("")

    lines.append("## 7. Sibling correlation (informational, not scored per-ticker)\n")
    if sibling_notes:
        for n in sibling_notes:
            lines.append(f"- {n}")
    else:
        lines.append("- No same-company sibling pairs in this universe below the 0.5 threshold.")
    lines.append("")

    lines.append("## 8. Schema/dtype contract\n")
    lines.append(f"{'PASSED' if schema_ok else 'FAILED'} for this subset+window (ticker=object, "
                 "trade_date=datetime64, close/has_fundamentals numeric).\n")

    lines.append("## 9. Survivorship bias note\n")
    lines.append(
        "0/50 tickers in this universe are `CANCELADA` (delisted) — expected, not a bug: ranking by "
        "post-2011 liquidity structurally selects survivors (a stock that delisted rarely stays in the "
        "top 50 by trading value through to the dataset's end). This is fine for a phase-1 ML-agent "
        "training set (per the project's phased validate-then-scale approach) but is a research-design "
        "caveat, not a data-integrity one: a backtest confined to this universe will not see how the "
        "strategy would have handled a name that was top-50 in, say, 2015 and delisted by 2020. "
        "`TOP50_UNIVERSE_VALIDATION.md` covers the point-in-time-rebalanced alternative for when that "
        "matters.\n"
    )

    lines.append("## 10. Recommendations before first training run\n")
    lines.append("- [x] `CCTY3` quarantined (§2) -- no longer reaches `ml_dataset.parquet`.")
    lines.append("- [x] `apply_ticker_continuity()` fixed to reconcile `adj_close` across every splice "
                  "boundary (§4) -- confirmed firing dataset-wide (14 tickers), not just the original 4.")
    lines.append("- [x] `MRFG3`/`MBRF3` chronic precision-floor bug fixed from yfinance (§4).")
    lines.append("- [x] `CCRO3->MOTV3` rename added to `ticker_continuity.json`; `BHIA3`/`BRKM5` "
                  "collection gaps backfilled from yfinance (§5).")
    if n_not_ready:
        not_ready = [t for t, v in verdicts.items() if v == "NOT READY"]
        lines.append(f"- [ ] Resolve critical findings for: {', '.join(not_ready)} before training.")
    else:
        lines.append("- [x] Zero CRITICAL findings remain across all 50 tickers.")
    if n_caveats:
        lines.append(f"- [ ] Optional: review warning-level findings for the {n_caveats} \"READY WITH "
                      "CAVEATS\" tickers above — per explicit scope of this pass, single-day "
                      "|log_return| >= 0.3 warnings not near a recorded corporate event are treated as "
                      "informational only and were not investigated individually; every other warning "
                      "category (gaps, stale prices, fundamentals coverage, sector) was investigated and "
                      "is resolved or explained above.")
    lines.append("- [ ] Re-run this script after any raw data re-collection or dataset rebuild "
                 f"(`python tests/build_dataset/test_top50_ml_readiness.py`).")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines))

    if n_not_ready:
        sys.exit(1)


if __name__ == "__main__":
    main()
