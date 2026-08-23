"""
test_br_data_quality.py
========================
Whole-universe sanity sweep over data/raw/br/{prices,fundamentals,dividends,
macro,company_info,corporate_events,filing_dates} -- the BR analogue of
test_us_data_quality.py, closing the gap noted
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
     RESOLVED 2026-08-23: re-measured at 0/612 files after the CVM rebuild.
     Ceiling ratcheted to zero -- see _INF_RATE_CEILING.
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
  6. Added 2026-08-23: selic vs. cdi cross-series agreement. The per-file
     macro loop can only check a series against itself, so a series-ID swap
     landing inside both plausible ranges survives it. The two series track
     each other by construction, which makes their spread a much tighter
     instrument than either range -- see _SELIC_CDI_MAX_SPREAD.
  7. Added 2026-08-23: the four market-wide REFERENCE tables. This file only
     ever globbed the one-file-per-ticker directories, so company_info,
     corporate_events, filing_dates and sectors were swept by nothing --
     even though validate.py already shipped validate_company_info /
     validate_corporate_events / validate_sectors and simply had no caller.
     One real finding: company_info.cnpj is stored in TWO formats (388 bare
     14-digit, 306 punctuated), which reads as 578 distinct companies where
     there are 455. Harmless only because every current consumer normalises
     first. See test_reference_tables_clean.

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
# validate_us_fundamentals) -- measured directly here.
# RATCHETED TO ZERO 2026-08-23. This landed 2026-08-16 as a 0.20 ceiling
# against a measured 103/612 (16.83%) backlog. Re-measured after the CVM
# rebuild: 0/612 files contain Inf. The backlog is gone, so the ceiling is
# now pure slack -- it would stay green through a regression reintroducing
# Inf into a fifth of the universe. A rate ceiling only ever earns its keep
# while the backlog it names still exists.
_INF_RATE_CEILING = 0.0  # currently 0/612 files = 0.00%

# selic and cdi track each other by construction (the CDI floats within a
# hair of the SELIC target). Asserting their AGREEMENT is a far tighter guard
# than the independent range checks in _MACRO_RANGES: a series-ID swap that
# lands inside both ranges -- the 432-vs-433 class CLAUDE.md documents as
# having bitten before -- survives a range check and dies here.
# Measured 2026-08-23: max |selic - cdi| = 0.00764, p99 0.00077, mean 0.00016
# over 6,686 shared trading days. The ceiling is ~2.6x the observed max.
_SELIC_CDI_MAX_SPREAD = 0.02

# CVM's SIT vocabulary as it appears on disk 2026-08-23 (ATIVO 562,
# CANCELADA 122, SUSPENSO(A) - DECISAO ADM 10). Not cosmetic: these exact
# strings are compared against by test_universe_integrity's survivorship
# floor and terminal_events' delisting classification, so a new value is a
# silent accounting change, not a new label.
_KNOWN_STATUS = {"ATIVO", "CANCELADA", "SUSPENSO(A) - DECISÃO ADM"}
_KNOWN_REPORT_TYPES = {"ITR", "DFP"}
# Bare 14 digits, no punctuation -- the form cvm/filing_dates.py and
# cvm/delistings.py already normalise to. See test_reference_tables_clean.
_CNPJ_CANONICAL = r"\d{14}"

# selic/cdi: percent per TRADING DAY (see manifest.COLUMN_UNITS); ipca:
# percent per CALENDAR MONTH. Bounds are deliberately generous -- these
# catch a unit/scale regression (e.g. an annual rate landing in a daily
# column), not ordinary macro variance.
# -- shares.parquet (cvm/shares.py) ------------------------------------------
# Absolute plausibility band for a share count. Measured 2026-08-23 on
# data/raw/br/cvm/shares.parquet (5,398 rows / 1,194 cnpjs): 25 rows below
# 1,000 and 28 above 1e12, across 30 companies. Both ends are CVM filer
# scale errors, not real capital structures -- FRE's own Valor_Capital on
# those same rows implies a par value of R$0.0000237/share (cnpj
# 01258944000126, 2018-04-25: R$2.70bn of paid-in capital against 113.5
# trillion shares). Zero tolerance: nothing legitimate lives out here, and
# every one of these reaches the panel through cvm/ratios.py.
_SHARES_MIN, _SHARES_MAX = 1_000, 1e12

# Ceiling on the adjacent-event share-count ratio per cnpj. A real 1:1000
# inplit exists; a jump from a literal 100 shares to 569.6 million does not.
# Measured: 43/4,204 adjacent pairs (1.02%) exceed 1000x, max 5,695,983x.
_SHARES_MAX_EVENT_RATIO = 1_000.0

# effective_date comes from FRE's Data_Autorizacao_Aprovacao, which filers
# populate by hand. Measured: 1 row dated 2029-07-15 (future) and 1 dated
# 1890-09-12. Both are merge_asof anchors in cvm/ratios._shares_asof -- a
# future date silently never matches, a pre-1900 date anchors a company's
# entire history to a garbage snapshot.
_SHARES_MIN_DATE = pd.Timestamp("1950-01-01")

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


def test_reference_tables_clean():
    """The four BR reference tables the per-ticker sweeps above never touch.

    Added 2026-08-23. `data/raw/br/` has nine directories; this file previously
    swept four of them (prices, fundamentals, dividends, macro). The rest are
    market-wide reference tables rather than one-file-per-ticker, so they fell
    outside the glob -- and they are not incidental:

      company_info    -> `status` (the survivorship gate), `sector` (6 derived
                         cross-sectional features), `cnpj`, `cvm_code`
      corporate_events-> repair.py's split ground truth, AND the ground truth
                         test_final_dataset.py's own split-leak check compares
                         against. A corrupt event log silently weakens that test
                         instead of failing it.
      filing_dates    -> CVM DT_RECEB. The entire no-lookahead claim rests on
                         this table; test_final_dataset.py measures how often
                         it's USED, nothing checked it was internally sound.
      sectors         -> small sanity aggregate.

    validate.py already had validate_company_info / validate_corporate_events /
    validate_sectors and nothing called them on what's actually on disk -- so
    most of this is wiring, not new predicates.

    All measured 2026-08-23. One is RED: see _CNPJ_CANONICAL below.
    """
    ok = True

    # -- company_info ------------------------------------------------------
    path = BR_ROOT / "company_info/company_info.parquet"
    if not path.exists():
        print("SKIP  company_info: not collected yet")
    else:
        df = pd.read_parquet(path)
        r = validate.validate_company_info(df)
        # CVM's SIT vocabulary. A value outside this set is not a cosmetic
        # surprise: `status` drives the CANCELADA survivorship accounting in
        # test_universe_integrity and terminal_events' delisting classification,
        # both of which compare against these exact strings.
        unknown = sorted(set(df["status"].dropna()) - _KNOWN_STATUS)
        sector_nan = int(df["sector"].isna().sum())
        dup = int(df["ticker"].duplicated().sum())
        # cnpj must be ONE canonical form. Measured: 388/694 bare 14-digit,
        # 306 punctuated ("42.771.949/0001-35") -- 578 distinct strings for
        # 455 real companies, a 27% phantom-company over-count for anything
        # grouping on the raw value. Harmless TODAY only because every current
        # consumer normalises first (quality_filters.py:328, cvm/delistings.py:49,
        # cvm/filing_dates.py:50, cvm/sectors.py:35) -- and quality_filters' is
        # the single line standing between the panel and a 44% filing-date join
        # miss. That is a landmine, not a convention. Fix is one str.replace at
        # write time in cvm/company_info.py.
        noncanonical = int((~df["cnpj"].dropna().astype(str)
                            .str.fullmatch(_CNPJ_CANONICAL)).sum())
        file_ok = r.passed and not unknown and sector_nan == 0 and dup == 0 and noncanonical == 0
        ok &= file_ok
        print(f"{'PASS' if file_ok else 'FAIL'}  company_info: {len(df)} tickers, "
              f"errors={r.errors}, unknown status={unknown}, sector NaN={sector_nan}, "
              f"dup ticker={dup}, non-canonical cnpj={noncanonical}")

    # -- sectors -----------------------------------------------------------
    path = BR_ROOT / "company_info/sectors.parquet"
    if path.exists():
        r = validate.validate_sectors(pd.read_parquet(path))
        ok &= r.passed
        print(f"{'PASS' if r.passed else 'FAIL'}  sectors: errors={r.errors}")

    # -- corporate_events --------------------------------------------------
    path = BR_ROOT / "corporate_events/corporate_events.parquet"
    if not path.exists():
        print("SKIP  corporate_events: not collected yet")
    else:
        df = pd.read_parquet(path)
        r = validate.validate_corporate_events(df)
        # factor must be the stated ratio. Either orientation is accepted --
        # CLAUDE.md already documents this log's factor convention as
        # inconsistent, and test_final_dataset.py's split-leak check tries both.
        # rtol 1e-4, not tighter: `factor` is stored at 6 decimals, so 1:29
        # lands at 0.034483 vs 1/29 = 0.0344827586. At rtol 1e-6 that rounding
        # alone reads as 37 false mismatches; at 1e-5 and looser, zero.
        calc = df["ratio_to"] / df["ratio_from"]
        agree = (np.isclose(df["factor"], calc, rtol=1e-4)
                 | np.isclose(df["factor"], 1 / calc, rtol=1e-4))
        bad_ratio = int((~agree).sum())
        future = int((pd.to_datetime(df["date"]) > pd.Timestamp.today()).sum())
        file_ok = r.passed and bad_ratio == 0 and future == 0
        ok &= file_ok
        print(f"{'PASS' if file_ok else 'FAIL'}  corporate_events: {len(df)} events, "
              f"errors={r.errors}, factor!=ratio_to/ratio_from={bad_ratio}, future-dated={future}")

    # -- filing_dates ------------------------------------------------------
    path = BR_ROOT / "filing_dates/filing_dates.parquet"
    if not path.exists():
        print("SKIP  filing_dates: not collected yet")
    else:
        df = pd.read_parquet(path)
        ref = pd.to_datetime(df["reference_date"])
        rec = pd.to_datetime(df["received_date"])
        # A filing received on or before the period-end it reports is
        # impossible, and would inject real lookahead through merge_asof --
        # this is the tightest single guard on the no-lookahead claim.
        # Measured: 0 violations, minimum lag 1 day, median 45.
        impossible = int((rec <= ref).sum())
        dup = int(df.duplicated(subset=["cnpj", "reference_date", "report_type"]).sum())
        unknown = sorted(set(df["report_type"].dropna()) - _KNOWN_REPORT_TYPES)
        future = int((rec > pd.Timestamp.today()).sum())
        nan = int(df[["cnpj", "cvm_code", "received_date", "reference_date"]].isna().sum().sum())
        noncanonical = int((~df["cnpj"].astype(str).str.fullmatch(_CNPJ_CANONICAL)).sum())
        file_ok = (impossible == 0 and dup == 0 and not unknown and future == 0
                   and nan == 0 and noncanonical == 0)
        ok &= file_ok
        print(f"{'PASS' if file_ok else 'FAIL'}  filing_dates: {len(df):,} filings, "
              f"received<=reference={impossible}, dupes={dup}, unknown report_type={unknown}, "
              f"future={future}, NaN={nan}, non-canonical cnpj={noncanonical}")

    # -- cvm/shares.parquet ------------------------------------------------
    # The share count is the denominator of vpa, lpa and the numerator's
    # partner in market_cap. It is also the ONE input the unit-scale identity
    # suite structurally cannot check: every identity it asserts
    # (vpa*shares==equity, market_cap==close*shares, book_to_market*pvp==1)
    # divides the error back out again, so all 45 of them stay green on a
    # company carrying 210 trillion shares. Checked here, at the source.
    path = BR_ROOT / "cvm/shares.parquet"
    if not path.exists():
        print("SKIP  cvm/shares: not collected yet")
    else:
        df = pd.read_parquet(path)
        eff = pd.to_datetime(df["effective_date"])
        out_of_band = int((~df["shares"].between(_SHARES_MIN, _SHARES_MAX)).sum())
        future = int((eff > pd.Timestamp.today()).sum())
        ancient = int((eff < _SHARES_MIN_DATE).sum())
        tl = df.assign(effective_date=eff).sort_values(["cnpj", "effective_date"])
        prev = tl.groupby("cnpj")["shares"].shift()
        ratio = pd.concat([tl["shares"] / prev, prev / tl["shares"]], axis=1).max(axis=1)
        jumps = int((ratio > _SHARES_MAX_EVENT_RATIO).sum())
        bad_cnpjs = tl.loc[(~tl["shares"].between(_SHARES_MIN, _SHARES_MAX))
                           | (ratio > _SHARES_MAX_EVENT_RATIO), "cnpj"].nunique()
        file_ok = out_of_band == 0 and future == 0 and ancient == 0 and jumps == 0
        ok &= file_ok
        print(f"{'PASS' if file_ok else 'FAIL'}  cvm/shares: {len(df):,} rows / "
              f"{df['cnpj'].nunique()} cnpjs, outside [{_SHARES_MIN:,.0f}, {_SHARES_MAX:.0e}]"
              f"={out_of_band}, adjacent ratio >{_SHARES_MAX_EVENT_RATIO:.0f}x={jumps}, "
              f"future-dated={future}, pre-{_SHARES_MIN_DATE.year}={ancient}, "
              f"affected cnpjs={bad_cnpjs}")

    return ok


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

    # Cross-series identity: the per-file loop above can only ever check a
    # series against itself. See _SELIC_CDI_MAX_SPREAD.
    selic_path, cdi_path = macro_dir / "selic.parquet", macro_dir / "cdi.parquet"
    if selic_path.exists() and cdi_path.exists():
        both = (pd.read_parquet(selic_path)
                  .merge(pd.read_parquet(cdi_path), on="reference_date")
                  .dropna(subset=["selic", "cdi"]))
        spread = (both["selic"] - both["cdi"]).abs()
        pair_ok = len(both) > 0 and spread.max() <= _SELIC_CDI_MAX_SPREAD
        ok &= pair_ok
        print(f"{'PASS' if pair_ok else 'FAIL'}  macro/selic-vs-cdi: {len(both)} shared days, "
              f"max |spread| {spread.max():.5f} (ceiling {_SELIC_CDI_MAX_SPREAD})")

    return ok


if __name__ == "__main__":
    all_ok = (test_prices_clean()
              & test_fundamentals_clean()
              & test_dividends_clean()
              & test_reference_tables_clean()
              & test_macro_clean())
    sys.exit(0 if all_ok else 1)
