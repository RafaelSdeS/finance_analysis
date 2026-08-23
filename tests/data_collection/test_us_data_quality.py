"""
test_us_data_quality.py
========================
Whole-universe sanity sweep over data/raw/us/{prices,fundamentals,macro,sec} --
the regression net for the audit run 2026-07-30 (see docs/US_COLLECTOR_BUG_AUDIT.md
and US_COLLECTOR_FIX_PLAN.md for the full write-up). Reuses the same
validate.py gates the collectors themselves call at write time, applied here
across every file already on disk so a systemic regression (not just a single
bad collection run) gets caught.

Two real findings this test encodes as regression guards, not just narrative:
  1. item6.py's footnote-marker bug (originally fixed 2026-07-30, see
     test_sec_item6.py) produced accounting-impossible negative total_assets
     on NEM/ORCL, collected before that first fix. BOOM and ZION, found in
     the SAME run's negative-total_assets flags, turned out NOT to be stale
     pre-fix leftovers -- both were freshly regenerated in a run that
     happened AFTER the first fix, and root-causing them live (fetching the
     real EDGAR filings directly) turned up two more, DIFFERENT bugs in the
     same subsystem: ZION's Item 6 is incorporated by reference, so
     find_item6_table picked a business-segment fragment over the real
     table (fixed by ranking on year count first); BOOM's real "Total
     assets" row had a colspan-duplication artifact that defeated the
     footnote-marker guard (fixed by collapsing duplicate pairs first). Both
     now fixed (2026-07-30) alongside a third, unrelated bug found the same
     way: implausibly ancient `end` dates (TENX/NG/CLSK, e.g. end=1967) in
     the xbrl tier, from genuine val=0 placeholder XBRL contexts in the
     filers' own data -- companyfacts.py never had item6.py's equivalent
     plausibility bound; it does now. All 4 on-disk tickers (NEM, ORCL,
     BOOM, ZION) plus the ancient-date tickers are stale until a fresh
     `collect_fundamentals_us` pass regenerates them (no auto-refresh of
     already-collected tickers; see fix plan's pothole #8 note) -- this test
     doesn't re-fix on-disk data, only guards against a NEW, systemic
     regression. The rate ceiling below tolerates today's small known-stale
     count.
  2. Everything else checked here (Inf, OHLC bracket sanity, macro
     completeness) was already clean at audit time -- these are hard
     zero-tolerance assertions, not rate ceilings.
  3. validate.validate_prices gained a NaN-OHLC check 2026-08-16 (data
     integrity review, see docs/DATA_INTEGRITY_TEST_PLAN.md D10/D2/D3):
     every prior check there was a comparison, and NaN compares False in
     pandas, so a NaN bar silently passed every one of them. Found on BR's
     BOVA11 (the market-beta benchmark) and CAMB3; applying the new check
     retroactively here surfaces the same defect class on 111/9700 US price
     files (1.14%), all pre-dating the fix. Rate ceiling tolerates the
     backlog while still catching a NEW regression; other validate_prices
     errors (bracket violations, non-positive prices, ...) stay
     zero-tolerance, unaffected by this ceiling.
  4. Added 2026-08-23: data/raw/us/sec/ -- the crosswalk, universe roster,
     EDGAR filings index and SIC company_info. This file only ever globbed
     the one-file-per-ticker directories, so the four tables that gate which
     tickers exist at all, and that supply the manifest's survivorship
     denominator, were checked by nothing. All clean on measurement; see
     test_sec_artifacts_clean.

Skips gracefully (prints SKIP, still exits 0) if data/raw/us isn't collected
yet -- it's gitignored, unlike BR's git-tracked data/raw/.

Run from project root:
    python tests/data_collection/test_us_data_quality.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data_collection import validate  # noqa: E402

US_ROOT = ROOT / "data/raw/us"

# Known-stale pre-fix files as of 2026-07-30 (see module docstring, finding
# 1) -- collected before item6.py's footnote-marker fix, not yet
# regenerated. A rate ceiling (not an exact-zero assert) tolerates these
# while still catching a NEW, systemic negative-total_assets regression.
_NEGATIVE_ASSETS_RATE_CEILING = 0.01  # currently 3/2289 = 0.13%

# xbrl tier only -- item6 (>=1990) and ex27 (1995-2000) tiers have their own,
# earlier, legitimate floors; nothing in the xbrl tier should ever predate
# companyfacts.py's own _MIN_PLAUSIBLE_END guard (2026-07-30 fix). A rate
# ceiling tolerates today's known-stale pre-fix tickers (TENX/NG/CLSK/etc.)
# while still catching a NEW regression.
_MIN_PLAUSIBLE_XBRL_END = pd.Timestamp("1995-01-01")
_ANCIENT_END_RATE_CEILING = 0.02  # currently 48/4775 tickers = 1.01%

# validate.validate_prices gained a NaN-OHLC check 2026-08-16 (previously every
# check there was a comparison, and NaN compares False in pandas, so a NaN bar
# silently passed every one of them -- found via BR's BOVA11/CAMB3). Applied
# retroactively here, it surfaces a pre-existing collection gap rather than a
# regression: 111/9700 US price files (1.14%) carry it, all pre-dating the fix.
# Rate ceiling tolerates today's backlog while still catching a NEW regression;
# tighten (or drop, once recollected) after a fresh `collect_prices_us` pass.
_NAN_OHLC_RATE_CEILING = 0.02  # currently 111/9700 files = 1.14%
_NAN_OHLC_ERROR_RE = re.compile(r"^\d+ rows with NaN in \[")

# validate.validate_prices' day-over-day adj_close jump guard
# (MAX_PLAUSIBLE_DAILY_MOVE): the US build has no equivalent of BR's
# repair.py split-rescale step (build_us_dataset.py deliberately skips it,
# per CLAUDE.md), so any jump caught here survives into us_ml_dataset.parquet
# unrepaired, unlike BR. Measured 2026-08-22: 316/9700 files, almost all
# sub-$1 OTC-adjacent penny names on a 1-3 day gap, e.g. AAGH (0.10 -> 0.001,
# 1-day gap, real volume both sides) -- consistent with an unrepaired
# reverse split rather than corruption, but not verified against a
# corporate-actions feed the way BR's repair.py cross-checks its jumps.
# Rate ceiling tolerates today's backlog while catching a NEW regression.
_JUMP_WARNING_RE = re.compile(r"^\d+ day\(s\) with adj_close moving >")
_JUMP_RATE_CEILING = 0.035  # currently 316/9700 files = 3.26%

# sec/company_info.parquet is collected on its own cadence and is a superset
# of the crosswalk, so a small orphan share is ordinary drift between two
# runs, not corruption. Measured 2026-08-23: 78/10,545 = 0.74%.
_CROSSWALK_ORPHAN_RATE_CEILING = 0.05


def _has_inf(df: pd.DataFrame) -> bool:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return False
    return bool(np.isinf(numeric.to_numpy(dtype="float64", na_value=0.0)).any())


def test_prices_clean():
    price_dir = US_ROOT / "prices"
    files = sorted(price_dir.glob("*.parquet")) if price_dir.exists() else []
    if not files:
        print("SKIP  prices: data/raw/us/prices not collected yet")
        return True

    validate_errors = 0
    nan_ohlc_files = []
    other_error_files = []
    inf_files = []
    jump_warning_files = []
    for f in files:
        df = pd.read_parquet(f)
        r = validate.validate_prices(df)
        if not r.passed:
            validate_errors += 1
            print(f"FAIL  prices/{f.stem}: {r.errors}")
            if len(r.errors) == 1 and _NAN_OHLC_ERROR_RE.match(r.errors[0]):
                nan_ohlc_files.append(f.stem)
            else:
                other_error_files.append(f.stem)
        if any(_JUMP_WARNING_RE.match(w) for w in r.warnings):
            jump_warning_files.append(f.stem)
        if _has_inf(df):
            inf_files.append(f.stem)

    nan_ohlc_rate = len(nan_ohlc_files) / len(files)
    jump_rate = len(jump_warning_files) / len(files)
    ok = (not other_error_files and not inf_files
          and nan_ohlc_rate <= _NAN_OHLC_RATE_CEILING
          and jump_rate <= _JUMP_RATE_CEILING)
    if inf_files:
        print(f"FAIL  prices: Inf present in {len(inf_files)} file(s): {inf_files[:10]}")
    if other_error_files:
        print(f"FAIL  prices: non-NaN-OHLC validate_prices error(s) in {len(other_error_files)} "
              f"file(s): {other_error_files[:10]}")
    if nan_ohlc_rate > _NAN_OHLC_RATE_CEILING:
        print(f"FAIL  prices: NaN-OHLC rate {nan_ohlc_rate:.2%} exceeds "
              f"{_NAN_OHLC_RATE_CEILING:.0%} ceiling -- {nan_ohlc_files[:10]}")
    elif nan_ohlc_files:
        print(f"note  prices: {len(nan_ohlc_files)} known-stale file(s) with NaN OHLC "
              f"(pre-fix), within ceiling: {nan_ohlc_files[:10]}")
    if jump_rate > _JUMP_RATE_CEILING:
        print(f"FAIL  prices: implausible day-over-day adj_close jump rate {jump_rate:.2%} exceeds "
              f"{_JUMP_RATE_CEILING:.0%} ceiling -- {jump_warning_files[:10]}")
    elif jump_warning_files:
        print(f"note  prices: {len(jump_warning_files)} file(s) with a >{validate.MAX_PLAUSIBLE_DAILY_MOVE}x "
              f"day-over-day adj_close move, within ceiling: {jump_warning_files[:10]}")
    print(f"{'PASS' if ok else 'FAIL'}  prices: {len(files)} files, "
          f"{validate_errors} validate_prices errors ({len(nan_ohlc_files)} NaN-OHLC, "
          f"{len(other_error_files)} other), {len(inf_files)} with Inf, "
          f"{len(jump_warning_files)} jump-warning")
    return ok


def test_fundamentals_clean():
    fund_dir = US_ROOT / "fundamentals"
    files = sorted(fund_dir.glob("*.parquet")) if fund_dir.exists() else []
    if not files:
        print("SKIP  fundamentals: data/raw/us/fundamentals not collected yet")
        return True

    gate_errors = 0
    inf_files = []
    neg_assets_files = []
    ancient_end_files = []
    total_warnings = 0
    for f in files:
        df = pd.read_parquet(f)
        r = validate.validate_us_fundamentals(df)
        if not r.passed:
            gate_errors += 1
            print(f"FAIL  fundamentals/{f.stem}: {r.errors}")
        total_warnings += len(r.warnings)
        if _has_inf(df):
            inf_files.append(f.stem)
        if "total_assets" in df.columns and (df["total_assets"] < 0).any():
            neg_assets_files.append(f.stem)
        if "fundamentals_tier" in df.columns and "end" in df.columns:
            xbrl_ends = df.loc[df["fundamentals_tier"] == "xbrl", "end"]
            if (xbrl_ends < _MIN_PLAUSIBLE_XBRL_END).any():
                ancient_end_files.append(f.stem)

    neg_rate = len(neg_assets_files) / len(files)
    ancient_rate = len(ancient_end_files) / len(files)
    ok = (gate_errors == 0 and not inf_files
          and neg_rate <= _NEGATIVE_ASSETS_RATE_CEILING
          and ancient_rate <= _ANCIENT_END_RATE_CEILING)
    if inf_files:
        print(f"FAIL  fundamentals: Inf present in {len(inf_files)} file(s): {inf_files[:10]}")
    if neg_rate > _NEGATIVE_ASSETS_RATE_CEILING:
        print(f"FAIL  fundamentals: negative total_assets rate {neg_rate:.2%} exceeds "
              f"{_NEGATIVE_ASSETS_RATE_CEILING:.0%} ceiling -- {neg_assets_files}")
    elif neg_assets_files:
        print(f"note  fundamentals: {len(neg_assets_files)} known-stale file(s) with negative "
              f"total_assets, within ceiling: {neg_assets_files}")
    if ancient_rate > _ANCIENT_END_RATE_CEILING:
        print(f"FAIL  fundamentals: implausible xbrl-tier end date rate {ancient_rate:.2%} exceeds "
              f"{_ANCIENT_END_RATE_CEILING:.0%} ceiling -- {ancient_end_files[:10]}")
    elif ancient_end_files:
        print(f"note  fundamentals: {len(ancient_end_files)} known-stale file(s) with an "
              f"implausible xbrl-tier end date (pre-fix), within ceiling: {ancient_end_files[:10]}")
    print(f"{'PASS' if ok else 'FAIL'}  fundamentals: {len(files)} files, {gate_errors} gate errors, "
          f"{len(inf_files)} with Inf, {len(neg_assets_files)} with negative total_assets "
          f"({neg_rate:.2%}), {len(ancient_end_files)} with implausible xbrl end dates "
          f"({ancient_rate:.2%}), {total_warnings} total warnings")
    return ok


def test_macro_clean():
    macro_dir = US_ROOT / "macro"
    files = sorted(macro_dir.glob("*.parquet")) if macro_dir.exists() else []
    if not files:
        print("SKIP  macro: data/raw/us/macro not collected yet")
        return True

    ok = True
    for f in files:
        name = f.stem
        df = pd.read_parquet(f)
        r = validate.validate_macro(df, name)
        nan_rate = df[name].isna().mean() if name in df.columns else 1.0
        has_inf = _has_inf(df)
        file_ok = r.passed and nan_rate == 0.0 and not has_inf
        ok &= file_ok
        status = "PASS" if file_ok else "FAIL"
        print(f"{status}  macro/{name}: rows={len(df)} nan_rate={nan_rate:.1%} inf={has_inf} errors={r.errors}")
    return ok


def test_sec_artifacts_clean():
    """The four SEC reference tables in data/raw/us/sec/, swept by nothing
    before 2026-08-23.

    These are not incidental outputs -- they gate the whole US side upstream
    of any price or fundamental:

      cik_ticker_crosswalk -> decides which tickers get collected AT ALL
                              (survivor-only by construction, accepted
                              2026-07-29 per US_COLLECTOR_FIX_PLAN.md §4)
      us_universe_roster   -> the bias-free filer roster whose per-year CIK
                              count IS the manifest's `survivorship_coverage`
                              denominator -- the only number tracking US
                              survivorship bias at all
      edgar_10k10q_filings -> the filing index behind every point-in-time
                              `fundamentals_available_date`
      company_info         -> SIC -> sector for the US build

    A silent regression in any of them (a truncated crosswalk, a roster
    missing a year, a filings index with the wrong quarter labels) degrades
    the US panel without touching a single price file, so nothing else here
    would notice.

    Everything asserted at zero tolerance except the crosswalk-orphan rate --
    all measured clean 2026-08-23 against the tables on disk.
    """
    sec_dir = US_ROOT / "sec"
    if not sec_dir.exists():
        print("SKIP  sec: data/raw/us/sec not collected yet")
        return True

    ok = True
    crosswalk_ciks = set()

    path = sec_dir / "cik_ticker_crosswalk.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        crosswalk_ciks = set(df["cik"])
        dup = int(df["ticker"].duplicated().sum())
        bad_cik = int((df["cik"] <= 0).sum())
        file_ok = dup == 0 and bad_cik == 0 and len(df) > 0
        ok &= file_ok
        print(f"{'PASS' if file_ok else 'FAIL'}  sec/crosswalk: {len(df):,} rows, "
              f"dup ticker={dup}, cik<=0={bad_cik}, tiers={df['tier'].value_counts().to_dict()}")

    path = sec_dir / "company_info.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        dup = int(df["ticker"].duplicated().sum())
        # Orphans are a staleness signal, not corruption: company_info is
        # collected on its own cadence and is a superset of the crosswalk.
        # Measured 78/10,545 = 0.74%; ceiling catches a real divergence
        # (e.g. a crosswalk rebuild that dropped a whole tier) without
        # failing on ordinary drift between two collection runs.
        orphan = int((~df["cik"].isin(crosswalk_ciks)).sum()) if crosswalk_ciks else 0
        orphan_rate = orphan / len(df) if len(df) else 0.0
        file_ok = dup == 0 and orphan_rate <= _CROSSWALK_ORPHAN_RATE_CEILING
        ok &= file_ok
        print(f"{'PASS' if file_ok else 'FAIL'}  sec/company_info: {len(df):,} rows, "
              f"dup ticker={dup}, cik not in crosswalk={orphan} ({orphan_rate:.2%}, "
              f"ceiling {_CROSSWALK_ORPHAN_RATE_CEILING:.0%})")

    path = sec_dir / "us_universe_roster.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        dup = int(df.duplicated(subset=["cik", "year"]).sum())
        empty = int((df["n_filings"] <= 0).sum())
        years = set(df["year"].unique())
        # A hole in the year sequence silently zeroes that year's
        # survivorship_coverage denominator instead of failing loudly.
        gaps = sorted(set(range(int(df["year"].min()), int(df["year"].max()) + 1)) - years)
        file_ok = dup == 0 and empty == 0 and not gaps
        ok &= file_ok
        print(f"{'PASS' if file_ok else 'FAIL'}  sec/roster: {len(df):,} rows, "
              f"{int(df['year'].min())}-{int(df['year'].max())}, dup (cik,year)={dup}, "
              f"n_filings<=0={empty}, missing years={gaps}")

    path = sec_dir / "edgar_10k10q_filings.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        filed = pd.to_datetime(df["date_filed"])
        dup = int(df["filename"].duplicated().sum())
        bad_cik = int((df["cik"] <= 0).sum())
        future = int((filed > pd.Timestamp.today()).sum())
        unknown = sorted({f for f in df["form_type"].dropna().unique()
                          if not f.startswith(("10-K", "10-Q"))})
        # `quarter` is the EDGAR full-index partition the row came from; it
        # must agree with the row's own date_filed, or the index was stitched
        # together wrong and every filing-date lookup keyed on it is off.
        q = df["quarter"].str.extract(r"(\d{4})Q(\d)").astype(float)
        mismatch = int(((q[0] != filed.dt.year) | (q[1] != filed.dt.quarter)).sum())
        file_ok = dup == 0 and bad_cik == 0 and future == 0 and not unknown and mismatch == 0
        ok &= file_ok
        print(f"{'PASS' if file_ok else 'FAIL'}  sec/filings: {len(df):,} rows, "
              f"{filed.min().date()}-{filed.max().date()}, dup filename={dup}, cik<=0={bad_cik}, "
              f"future={future}, non-10K/Q forms={unknown}, quarter-label mismatch={mismatch}")

    return ok


if __name__ == "__main__":
    all_ok = (test_prices_clean()
              & test_fundamentals_clean()
              & test_sec_artifacts_clean()
              & test_macro_clean())
    sys.exit(0 if all_ok else 1)
