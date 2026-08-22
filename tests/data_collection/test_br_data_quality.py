"""
test_br_data_quality.py
========================
Whole-universe sanity sweep over data/raw/br/{prices,fundamentals,dividends,
macro} -- the BR analogue of test_us_data_quality.py, closing the gap noted
in docs/DATA_INTEGRITY_TEST_PLAN.md P1-a: nothing previously swept raw BR
data file-by-file (only the built ml_dataset.parquet was checked, by
test_final_dataset.py). Delegates to validate.validate_prices /
validate_fundamentals / validate_dividends -- the same write-time gates the
collectors themselves call -- instead of hand-rolling predicates, so a fix
to validate.py is inherited here automatically.

Real findings from writing this test, 2026-08-16:
  1. 103/612 (16.8%) BR fundamentals files contain Inf values. clean.py
     silently converts these to NaN downstream in the processed dataset, so
     the existing "no inf" check in test_final_dataset.py passes while this
     vendor-level defect goes unmeasured. Rate ceiling tolerates today's
     backlog while catching a NEW regression (validate_fundamentals doesn't
     check Inf itself, unlike validate_us_fundamentals -- checked directly
     here).
  2. LUXM4 has 289/5032 rows (2000-2005) where adj_close underflows to a
     literal 0.00 -- not NaN, an actual zero -- from BolsAI's 2-decimal
     float floor on a name with enough cumulative dividend-adjustment
     discount to fall below it. This is CLAUDE.md's documented
     "adj_close 2-decimal precision degraded" class (currently named
     quarantine: WDCN3, CAMB4, LLIS3, CCTY3), already masked before log()
     and flagged via adj_close_precision_degraded downstream -- LUXM4 just
     wasn't previously known BY NAME, because nothing swept raw BR prices
     file-by-file before this test. Rate-ceilinged like the CLAUDE.md-
     accepted class it is, not treated as a new bug to fix.
  3. BOVA11 and CAMB3 have a genuinely different, NEW defect: a NaN (not
     zero) raw OHLC bar -- see validate.py's NaN-OHLC check added the same
     day (D2/D3 in docs/DATA_INTEGRITY_TEST_PLAN.md). Zero-tolerance, not
     folded into the LUXM4-style ceiling: BOVA11 is the market-beta
     benchmark for the whole panel, this is not a name to quietly tolerate.
  4. Dividends (BR: 314 files, US: 4,243 files) were previously swept by
     NOTHING on either market. Both pass validate.validate_dividends
     cleanly today (measured 2026-08-16) -- written as one market-agnostic
     sweep function, called for both roots, so this closes the gap on both
     sides at once.
  5. Everything else checked here (dupes/bracket via validate.py, monotone
     dates, no weekend rows, macro completeness and series-identity) was
     already clean on real data -- hard zero-tolerance assertions.

Run from project root:
    python tests/data_collection/test_br_data_quality.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_collection import validate  # noqa: E402

BR_ROOT = ROOT / "data/raw/br"
US_ROOT = ROOT / "data/raw/us"

# CLAUDE.md's documented "adj_close 2-decimal precision degraded" class
# (BolsAI's float floor underflows adj_* to a literal 0.00 on names with
# enough cumulative dividend-adjustment discount) -- already masked before
# log() and flagged downstream via adj_close_precision_degraded. Rate
# ceiling tolerates today's known instances while catching a NEW,
# structurally different price defect (e.g. a genuinely broken raw price).
_ADJ_PRECISION_ERROR_RE = re.compile(r"^\d+ rows with non-positive adj_")
_ADJ_PRECISION_RATE_CEILING = 0.01  # currently 1/1328 files (LUXM4) = 0.08%

# validate.validate_prices' day-over-day adj_close jump guard
# (MAX_PLAUSIBLE_DAILY_MOVE): fires at collection time, before repair.py's
# Stage-2 split-rescale runs, so it's expected to catch a real backlog, not
# just noise. Measured 2026-08-22: 149/1328 files, almost all thinly-traded
# delisted/microcap names, e.g. BMEB3 (800.0 -> 15.5, 7-day gap, vol 3->10k)
# -- consistent with an unrepaired split/bonus-share event, the exact class
# repair.py exists to fix downstream. A few look like genuine decimal-entry
# corruption rather than a corporate action, e.g. AFLT3 (18.0 -> 0.018,
# 1-day gap, real volume both sides) -- an exact 1000x, not a plausible
# split ratio. Rate ceiling tolerates today's backlog while catching a NEW
# regression (e.g. a systemic scale bug spiking the rate well above today's).
_JUMP_WARNING_RE = re.compile(r"^\d+ day\(s\) with adj_close moving >")
_JUMP_RATE_CEILING = 0.12  # currently 149/1328 files = 11.22%

# validate_fundamentals (BR) has no Inf check of its own (unlike
# validate_us_fundamentals) -- measured directly here. Rate ceiling
# tolerates today's real backlog while catching a NEW systemic regression;
# tighten (or drop) once these are re-collected/repaired.
_INF_RATE_CEILING = 0.20  # currently 103/612 files = 16.83%

# selic/cdi: percent per TRADING DAY (see manifest.COLUMN_UNITS); ipca:
# percent per CALENDAR MONTH. Bounds are deliberately generous -- these
# catch a unit/scale regression (e.g. an annual rate landing in a daily
# column), not ordinary macro variance.
_MACRO_RANGES = {"selic": (0.0, 1.0), "cdi": (0.0, 1.0), "ipca": (-5.0, 10.0)}
# ipca is BCB SGS series 433 (monthly), not 432 (the annual meta target) --
# CLAUDE.md documents this as a caveat that's bitten before. A monthly
# series since 2000 has ~300+ rows; an annual one would have ~26.
_IPCA_MIN_ROWS = 200


def _has_inf(df: pd.DataFrame) -> bool:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return False
    return bool(np.isinf(numeric.to_numpy(dtype="float64", na_value=0.0)).any())


def test_prices_clean():
    price_dir = BR_ROOT / "prices"
    files = sorted(price_dir.glob("*.parquet")) if price_dir.exists() else []
    if not files:
        print("SKIP  prices: data/raw/br/prices not collected yet")
        return True

    adj_precision_files = []
    other_error_files = []
    unsorted_files = []
    weekend_files = []
    jump_warning_files = []
    for f in files:
        df = pd.read_parquet(f)
        r = validate.validate_prices(df)
        if not r.passed:
            print(f"FAIL  prices/{f.stem}: {r.errors}")
            if all(_ADJ_PRECISION_ERROR_RE.match(e) for e in r.errors):
                adj_precision_files.append(f.stem)
            else:
                other_error_files.append(f.stem)
        if any(_JUMP_WARNING_RE.match(w) for w in r.warnings):
            jump_warning_files.append(f.stem)
        d = pd.to_datetime(df["trade_date"])
        if not d.is_monotonic_increasing:
            unsorted_files.append(f.stem)
        if (d.dt.dayofweek >= 5).any():
            weekend_files.append(f.stem)

    adj_precision_rate = len(adj_precision_files) / len(files)
    jump_rate = len(jump_warning_files) / len(files)
    ok = (not other_error_files and not unsorted_files and not weekend_files
          and adj_precision_rate <= _ADJ_PRECISION_RATE_CEILING
          and jump_rate <= _JUMP_RATE_CEILING)
    if other_error_files:
        print(f"FAIL  prices: non-adj-precision validate_prices error(s) in {len(other_error_files)} "
              f"file(s): {other_error_files[:10]}")
    if adj_precision_rate > _ADJ_PRECISION_RATE_CEILING:
        print(f"FAIL  prices: adj-precision-underflow rate {adj_precision_rate:.2%} exceeds "
              f"{_ADJ_PRECISION_RATE_CEILING:.0%} ceiling -- {adj_precision_files[:10]}")
    elif adj_precision_files:
        print(f"note  prices: {len(adj_precision_files)} known file(s) with adj_close precision "
              f"underflow (CLAUDE.md-documented), within ceiling: {adj_precision_files[:10]}")
    if jump_rate > _JUMP_RATE_CEILING:
        print(f"FAIL  prices: implausible day-over-day adj_close jump rate {jump_rate:.2%} exceeds "
              f"{_JUMP_RATE_CEILING:.0%} ceiling -- {jump_warning_files[:10]}")
    elif jump_warning_files:
        print(f"note  prices: {len(jump_warning_files)} file(s) with a >{validate.MAX_PLAUSIBLE_DAILY_MOVE}x "
              f"day-over-day adj_close move, within ceiling: {jump_warning_files[:10]}")
    if unsorted_files:
        print(f"FAIL  prices: {len(unsorted_files)} file(s) not sorted by trade_date: {unsorted_files[:10]}")
    if weekend_files:
        print(f"FAIL  prices: {len(weekend_files)} file(s) with a weekend trade_date: {weekend_files[:10]}")
    print(f"{'PASS' if ok else 'FAIL'}  prices: {len(files)} files, {len(other_error_files)} other "
          f"validate_prices errors, {len(adj_precision_files)} adj-precision, {len(jump_warning_files)} "
          f"jump-warning, {len(unsorted_files)} unsorted, {len(weekend_files)} with weekend rows")
    return ok


def test_fundamentals_clean():
    fund_dir = BR_ROOT / "fundamentals"
    files = sorted(fund_dir.glob("*.parquet")) if fund_dir.exists() else []
    if not files:
        print("SKIP  fundamentals: data/raw/br/fundamentals not collected yet")
        return True

    validate_errors = []
    inf_files = []
    for f in files:
        df = pd.read_parquet(f)
        r = validate.validate_fundamentals(df)
        if not r.passed:
            validate_errors.append(f.stem)
            print(f"FAIL  fundamentals/{f.stem}: {r.errors}")
        if _has_inf(df):
            inf_files.append(f.stem)

    inf_rate = len(inf_files) / len(files)
    ok = not validate_errors and inf_rate <= _INF_RATE_CEILING
    if inf_rate > _INF_RATE_CEILING:
        print(f"FAIL  fundamentals: Inf rate {inf_rate:.2%} exceeds {_INF_RATE_CEILING:.0%} "
              f"ceiling -- {inf_files[:10]}")
    elif inf_files:
        print(f"note  fundamentals: {len(inf_files)} known file(s) with Inf, within ceiling: {inf_files[:10]}")
    print(f"{'PASS' if ok else 'FAIL'}  fundamentals: {len(files)} files, {len(validate_errors)} "
          f"validate_fundamentals errors, {len(inf_files)} with Inf ({inf_rate:.2%})")
    return ok


def _check_dividends_dir(root: Path, label: str) -> bool:
    div_dir = root / "dividends"
    files = sorted(div_dir.glob("*.parquet")) if div_dir.exists() else []
    if not files:
        print(f"SKIP  {label}/dividends: not collected yet")
        return True

    validate_errors = []
    for f in files:
        df = pd.read_parquet(f)
        r = validate.validate_dividends(df)
        if not r.passed:
            validate_errors.append(f.stem)
            print(f"FAIL  {label}/dividends/{f.stem}: {r.errors}")

    ok = not validate_errors
    print(f"{'PASS' if ok else 'FAIL'}  {label}/dividends: {len(files)} files, "
          f"{len(validate_errors)} validate_dividends errors")
    return ok


def test_dividends_clean():
    # One market-agnostic sweep, called for both roots: BR dividends had no
    # sweep at all before this test, and neither did US (4,243 files) --
    # test_us_data_quality.py only ever covered prices/fundamentals/macro.
    br_ok = _check_dividends_dir(BR_ROOT, "br")
    us_ok = _check_dividends_dir(US_ROOT, "us")
    return br_ok and us_ok


def test_macro_clean():
    macro_dir = BR_ROOT / "macro"
    files = sorted(macro_dir.glob("*.parquet")) if macro_dir.exists() else []
    if not files:
        print("SKIP  macro: data/raw/br/macro not collected yet")
        return True

    ok = True
    for f in files:
        name = f.stem
        df = pd.read_parquet(f)
        r = validate.validate_macro(df, name)
        nan_rate = df[name].isna().mean() if name in df.columns else 1.0
        has_inf = _has_inf(df)
        lo, hi = _MACRO_RANGES.get(name, (-np.inf, np.inf))
        in_range = df[name].between(lo, hi).all() if name in df.columns else False
        row_count_ok = len(df) >= _IPCA_MIN_ROWS if name == "ipca" else True

        file_ok = r.passed and nan_rate == 0.0 and not has_inf and in_range and row_count_ok
        ok &= file_ok
        status = "PASS" if file_ok else "FAIL"
        extra = ""
        if not in_range:
            extra += f" OUT-OF-RANGE(expected [{lo},{hi}])"
        if not row_count_ok:
            extra += f" TOO-FEW-ROWS(expected >={_IPCA_MIN_ROWS}, is series 433 not 432?)"
        print(f"{status}  macro/{name}: rows={len(df)} nan_rate={nan_rate:.1%} inf={has_inf} "
              f"errors={r.errors}{extra}")
    return ok


if __name__ == "__main__":
    all_ok = (test_prices_clean()
              & test_fundamentals_clean()
              & test_dividends_clean()
              & test_macro_clean())
    sys.exit(0 if all_ok else 1)
