"""
test_us_data_quality.py
========================
Whole-universe sanity sweep over data/raw/us/{prices,fundamentals,macro} --
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
        if _has_inf(df):
            inf_files.append(f.stem)

    nan_ohlc_rate = len(nan_ohlc_files) / len(files)
    ok = not other_error_files and not inf_files and nan_ohlc_rate <= _NAN_OHLC_RATE_CEILING
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
    print(f"{'PASS' if ok else 'FAIL'}  prices: {len(files)} files, "
          f"{validate_errors} validate_prices errors ({len(nan_ohlc_files)} NaN-OHLC, "
          f"{len(other_error_files)} other), {len(inf_files)} with Inf")
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


if __name__ == "__main__":
    all_ok = (test_prices_clean()
              & test_fundamentals_clean()
              & test_macro_clean())
    sys.exit(0 if all_ok else 1)
