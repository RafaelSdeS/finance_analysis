"""US golden gate -- the D6 gap from docs/DATA_INTEGRITY_TEST_PLAN.md, closed
2026-08-23.

Until this file, `data/processed/us_ml_dataset.parquet` (5.5 GB, 2,903
tickers, 15.4M rows, 1962-2026) was validated by NOTHING at row level. The
only US-side coverage was manifest-vs-manifest (`test_manifest_drift.py`),
artifact timestamps (`test_artifact_coherence.py`), and the scale/anchor
identities (`test_unit_scale_invariants.py`). Every structural invariant BR's
`test_final_dataset.py` asserts -- no lookahead, no duplicate keys, NaN
shape, flag coherence -- was simply unchecked on the US panel.

WHY A SEPARATE FILE INSTEAD OF `--market us` ON test_final_dataset.py
---------------------------------------------------------------------
The plan originally proposed adding a `--market` flag there. Two things make
that the wrong shape:

  1. `test_final_dataset.py` reads the whole frame into pandas and uses
     whole-frame idioms throughout. On BR (626 MB) that is fine; on US it
     peaks well past this machine's headroom. Everything here streams in
     row-group batches -- peak RSS stays ~300 MB on a 5.5 GB file.
  2. Roughly a third of BR's checks are genuinely BR-only (split-jump leakage
     vs. `corporate_events.parquet`, continuity splices, `filing_lag_days`
     provenance, `close_price` being dropped, `status`/CVM columns). Bolting a
     market flag onto them buys branching, not coverage.

What this file deliberately does NOT re-check, because
`test_unit_scale_invariants.py` already does it for BOTH markets: the seven
scale/valuation-anchor identities and the percent-vs-fraction scale bands.

BASELINE MEASURED 2026-08-23 against the 2026-08-16 US build. Five checks
here are RED on landing. They are real defects, recorded rather than
tolerated -- each needs a code fix, listed against it below:

  * 6 rows with NaN `close` (AOS/FCPT/SCCO/SMA/VRTS 2026-07/08, HUBB 1977;
    all volume=0). BR has zero.
  * 1,054 rows with NaN `adj_close`, of which **1,054 are unflagged** --
    `adj_close_precision_degraded` is 0 on every one. BR closed exactly this
    gap in DATA_LAYER_CORRECTNESS_PLAN.md §2a; the US build never got it.
  * `reference_date` goes BACKWARDS as `trade_date` advances -- 245,537 rows
    (2.43%) across 1,454/2,903 tickers (50.1%), median 274 days back, max
    4,748. BR: literally zero rows. Root-caused 2026-08-23, see
    _REFERENCE_DATE_REGRESSION below; `n_quarters_available` being
    non-monotone is the symptom, not the disease.
  * 956,477 rows (9.47%) carry a fiscal period more than 400 days older than
    their own trade_date, and 871,193 (8.63%) claim `has_fundamentals=1`
    while `net_income` is NaN. BR: 0.11% and ~0%. Same root cause.
  * 37 tickers (1.27%) carry a narrow interior NaN hole in `equity`,
    `net_income` AND `total_assets` simultaneously -- BR's "merge bug" shape.

Clean at baseline and therefore asserted at zero tolerance: duplicate keys,
weekend rows, lookahead, filing-date ordering, has_fundamentals=0 leakage,
flag domains, P/L daily re-anchoring, minimum history.

Runtime is minutes, not seconds: the inf sweep needs every numeric column, so
this reads the whole 5.5 GB once. That single pass also feeds the per-ticker
checks, so there is no second read.

Run from project root:
    python tests/build_dataset/test_us_final_dataset.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.build_dataset.paths import OUTPUT_PATH, US_OUTPUT_PATH  # noqa: E402
from test_utils import print_header, print_check, print_separator  # noqa: E402

BATCH_ROWS = 200_000

# Columns the per-ticker pass needs. Kept narrow on purpose: these frames are
# buffered until a ticker's last row arrives, so the width here is what sets
# the buffer's memory ceiling (worst case ~16k rows for the longest history).
NARROW = ["ticker", "trade_date", "reference_date", "pl", "volume",
          "equity", "net_income", "total_assets"]

MIN_ROWS_PER_TICKER = 10          # build_ml_dataset.MIN_PRICE_ROWS; thin names kept on purpose
MIN_SUSPICIOUS_GAP_ROWS = 15      # same rationale as test_final_dataset.py: a legitimately
                                  # dropped quarter is 60+ trading days wide; a merge artifact
                                  # is a handful of rows
FROZEN_PL_RATE_CEILING = 0.01     # measured 0/84,568 eligible (ticker, quarter) groups

# _REFERENCE_DATE_REGRESSION -- root-caused 2026-08-23.
#
# `n_quarters_available` is the rank of `reference_date` within a ticker's
# sorted distinct quarters (features.py:735), so it can only go down if the
# MERGED reference_date goes down as trade_date advances. It does, for half
# the US universe, and the mechanism is not a US quirk of the counter -- it is
# the asof merge answering the wrong question:
#
#   merge_asof(direction="backward") on `fundamentals_available_date` picks the
#   most recently PUBLISHED fundamentals row. What the panel needs is the most
#   recent FISCAL PERIOD among the rows published by now. Those coincide only
#   while publication order matches fiscal order.
#
# On BR they always coincide (CVM's DT_RECEB arrives in fiscal order): 0 rows
# regress, out of 1,225,282. On US they routinely don't, because SEC XBRL
# facts are collected as-first-reported (sec/companyfacts.as_first_reported
# dedupes each (start, end) to its EARLIEST filing -- deliberate and correct)
# and a 10-K publishes several years of prior-period comparatives at once. The
# earliest XBRL filing to mention a 2006 period can therefore be a 2009 10-K.
# Measured on raw data: 60,207/359,402 rows (16.75%) across 7,108/8,283 files
# (85.8%) carry an `end` behind a period already available, spread across every
# year 2009-2026 rather than clustering at the XBRL onset. 28,708 of them
# (xbrl tier, 100% flowless) are pure balance-sheet comparatives -- e.g. AAPL's
# FY2009 10-K contributed exactly two rows to the panel, `end=2006-09-30` and
# `end=2007-09-29`, carrying equity and nothing else.
#
# The merge then makes those the panel's "current" fundamentals until the next
# 10-Q, which is where the 9.47%-stale and 8.63%-NaN-net_income figures come
# from. NOT lookahead -- every row was genuinely public before its trade_date;
# the panel is under-using information it already holds, not leaking future
# information. The fix is a cummax filter on `reference_date` in
# fundamentals_available_date order, before the merge: a no-op on BR, and it
# drops exactly the rows that can only ever make the US panel staler.
REF_REGRESSION_TICKER_CEILING = 0.01  # BR: 0.0%. MEASURED US: 50.09% -- RED, see above
STALE_FUNDAMENTAL_DAYS = 400          # a fiscal period older than this is a year+ out of date
STALE_FUNDAMENTAL_RATE_CEILING = 0.01  # BR: 0.11%. MEASURED US: 9.47% -- RED, same root cause

# CAGR NaN is expected (negative base year, short history); what must not
# happen is a growing share of NaN with no attributable reason. Floors set
# just under measured so a real regression trips them.
CAGR_EARNINGS_EXPLAINED_FLOOR = 0.85   # measured 0.900
CAGR_REVENUE_EXPLAINED_FLOOR = 0.68    # measured 0.731 (same CVM/SEC ceiling story as BR's 0.75)

LEAK_COLS = ["pl", "pvp", "roe", "net_income", "market_cap"]
FLAG_COLS = ["cagr_earnings_defined", "cagr_revenue_defined", "adj_close_precision_degraded"]


class Acc:
    """Plain counters accumulated across batches -- no per-row state retained."""

    def __init__(self):
        self.rows = 0
        self.nan = {"close": 0, "volume": 0}
        self.inf = 0
        self.inf_cols = set()
        self.weekend = 0
        self.lookahead = 0
        self.trade_before_avail = 0
        self.avail_before_ref = 0
        self.leak = dict.fromkeys(LEAK_COLS, 0)
        self.hf_rows = 0
        self.stale = 0
        self.bad_adj = 0
        self.bad_adj_unflagged = 0
        self.flag_domain = 0
        self.cagr = dict.fromkeys(
            ["earn_nan", "earn_explained", "rev_nan", "rev_explained"], 0)
        self.macro = []
        # per-ticker
        self.tickers = 0
        self.dupes = 0
        self.min_rows = None
        self.short_tickers = []
        self.frozen_pl = 0
        self.eligible_pl = 0
        self.ref_regress_tickers = []
        self.ref_regress_rows = 0
        self.prefix_suspicious = []


def _panel_checks(d: pd.DataFrame, acc: Acc) -> None:
    acc.rows += len(d)

    for col in acc.nan:
        if col in d.columns:
            acc.nan[col] += int(d[col].isna().sum())

    # Per-column, not a whole-frame select_dtypes(): that copies every numeric
    # column just to test them (the same OOM shape test_final_dataset.py's own
    # inf check calls out). float only -- np.isinf rejects datetimes.
    for col, dtype in d.dtypes.items():
        if dtype.kind != "f":
            continue
        n = int(np.isinf(d[col].to_numpy()).sum())
        if n:
            acc.inf += n
            acc.inf_cols.add(col)

    acc.weekend += int((d["trade_date"].dt.dayofweek >= 5).sum())

    ref = d["reference_date"]
    acc.lookahead += int((ref.notna() & (ref > d["trade_date"])).sum())

    hf = d["has_fundamentals"] == 1
    if "fundamentals_available_date" in d.columns:
        avail = d["fundamentals_available_date"]
        acc.trade_before_avail += int((hf & avail.notna() & (d["trade_date"] < avail)).sum())
        acc.avail_before_ref += int((hf & avail.notna() & ref.notna() & (avail < ref)).sum())

    acc.hf_rows += int(hf.sum())
    gap = (d.loc[hf, "trade_date"] - d.loc[hf, "reference_date"]).dt.days
    acc.stale += int((gap > STALE_FUNDAMENTAL_DAYS).sum())

    nf = d.loc[~hf]
    for col in acc.leak:
        if col in d.columns:
            acc.leak[col] += int(nf[col].notna().sum())

    if {"adj_close", "adj_close_precision_degraded"} <= set(d.columns):
        bad = d["adj_close"].isna() | (d["adj_close"] <= 0)
        acc.bad_adj += int(bad.sum())
        acc.bad_adj_unflagged += int((bad & (d["adj_close_precision_degraded"] != 1)).sum())

    for col in FLAG_COLS:
        if col in d.columns:
            acc.flag_domain += int((~d[col].isin([0, 1, 0.0, 1.0])).sum())

    if {"cagr_earnings_5y_final", "n_quarters_available", "had_negative_earnings_5y"} <= set(d.columns):
        m = hf & d["cagr_earnings_5y_final"].isna()
        acc.cagr["earn_nan"] += int(m.sum())
        acc.cagr["earn_explained"] += int(
            ((d.loc[m, "had_negative_earnings_5y"] == 1) | (d.loc[m, "n_quarters_available"] < 20)).sum())
        m2 = hf & d["cagr_revenue_5y_final"].isna()
        acc.cagr["rev_nan"] += int(m2.sum())
        acc.cagr["rev_explained"] += int((d.loc[m2, "n_quarters_available"] < 20).sum())

    if "selic" in d.columns:
        acc.macro.append(d[["trade_date", "selic"]].drop_duplicates("trade_date"))


def _ticker_checks(ticker: str, g: pd.DataFrame, acc: Acc) -> None:
    acc.tickers += 1
    acc.dupes += int(g.duplicated(subset=["ticker", "trade_date"]).sum())
    acc.min_rows = len(g) if acc.min_rows is None else min(acc.min_rows, len(g))
    if len(g) < MIN_ROWS_PER_TICKER:
        acc.short_tickers.append(ticker)

    # P/L must be re-anchored to the daily close, not frozen at the filing
    # price. Restricted to days the stock actually traded -- on a zero-volume
    # day `close` itself doesn't move, so a flat `pl` is correct, not stale
    # (root-caused for BR 2026-08-21, DATA_LAYER_FOLLOWUP_FINDINGS.md).
    traded = g[g["pl"].notna() & (g["volume"] > 0)]
    if len(traded):
        grp = traded.groupby("reference_date")["pl"]
        sizes, nun = grp.size(), grp.nunique()
        acc.eligible_pl += int((sizes >= 5).sum())
        acc.frozen_pl += int(((sizes >= 5) & (nun == 1)).sum())

    # Asserted on reference_date directly rather than on n_quarters_available:
    # the counter is only its rank, so this is the same invariant one derivation
    # earlier, and it survives the counter being renamed or recomputed.
    r = g[g["reference_date"].notna()].sort_values("trade_date")
    if len(r) > 1:
        regress = (r["reference_date"].cummax().shift() > r["reference_date"]).fillna(False)
        n = int(regress.sum())
        if n:
            acc.ref_regress_rows += n
            acc.ref_regress_tickers.append(ticker)

    narrow = 0
    for col in ("equity", "net_income", "total_assets"):
        s = g[col].reset_index(drop=True)
        first = s.first_valid_index()
        if first is None:
            continue
        is_na = s.loc[first:].isna()
        if not is_na.any():
            continue
        widths = is_na.groupby((~is_na).cumsum()).sum()
        widths = widths[widths > 0]
        if len(widths) and widths.min() < MIN_SUSPICIOUS_GAP_ROWS:
            narrow += 1
    if narrow == 3:
        acc.prefix_suspicious.append(ticker)


def _batches(pf: pq.ParquetFile, acc: Acc):
    """Yields the narrow per-ticker slice; runs the whole-panel checks on the
    full batch on the way past, so the 5.5 GB file is read exactly once."""
    for batch in pf.iter_batches(batch_size=BATCH_ROWS):
        d = batch.to_pandas()
        _panel_checks(d, acc)
        yield d[NARROW]


def _by_ticker(batches):
    """Regroup batches into whole tickers. Relies on the file being sorted by
    ticker (asserted below by the seen-set: a ticker reappearing after its
    block closed means the assumption is wrong and every per-ticker check
    silently under-reports)."""
    seen, name, buf = set(), None, []
    for d in batches:
        for ticker, g in d.groupby("ticker", sort=False):
            if ticker != name:
                if name is not None:
                    yield name, pd.concat(buf, ignore_index=True)
                    seen.add(name)
                if ticker in seen:
                    raise AssertionError(
                        f"{ticker} reappears after its block closed -- the parquet is not "
                        f"sorted by ticker, so per-ticker checks here are unsound")
                name, buf = ticker, []
            buf.append(g)
    if name is not None:
        yield name, pd.concat(buf, ignore_index=True)


def _macro_is_us(acc: Acc) -> tuple[bool, str]:
    """`build_us_dataset.merge_macro_us()` emits US T-bill/CPI data under the
    literal column names `selic`/`ipca` so BR's feature code works unchanged.
    That reuse is deliberate, and it is also exactly what makes a mis-wire
    invisible: if the US build ever joined BR's macro table instead, every
    downstream `excess_return`/`real_return` would silently become excess
    over the Brazilian CDI, and nothing would notice. Assert the two series
    actually differ on the dates they share."""
    if not acc.macro:
        return False, "no `selic` column in the US dataset"
    us = pd.concat(acc.macro).drop_duplicates("trade_date").dropna()
    if us.empty:
        return False, "`selic` is entirely null"
    if not OUTPUT_PATH.exists():
        return True, f"{len(us)} US macro days; BR dataset absent, cross-check skipped"
    br = (pd.read_parquet(OUTPUT_PATH, columns=["trade_date", "selic"])
          .drop_duplicates("trade_date").dropna())
    both = us.merge(br, on="trade_date", suffixes=("_us", "_br"))
    if both.empty:
        return True, f"{len(us)} US macro days; no overlap with BR calendar"
    same = float((both["selic_us"] == both["selic_br"]).mean())
    return same < 0.01, (f"{len(both)} overlapping days, {same:.1%} identical to BR selic "
                         f"(median ratio {(both['selic_us'] / both['selic_br']).median():.3f})")


def main() -> int:
    print_header("US GOLDEN GATE -- us_ml_dataset.parquet")

    if not US_OUTPUT_PATH.exists():
        print(f"SKIP: {US_OUTPUT_PATH} not found (US dataset not built)")
        return 0

    acc = Acc()
    pf = pq.ParquetFile(US_OUTPUT_PATH)
    for ticker, g in _by_ticker(_batches(pf, acc)):
        _ticker_checks(ticker, g, acc)

    checks: list[tuple[str, bool]] = []

    checks.append((f"rows read [{acc.rows:,} over {acc.tickers:,} tickers]", acc.rows > 0))
    for col, n in acc.nan.items():
        checks.append((f"no NaN in {col} [{n} found]", n == 0))
    checks.append((f"no inf in numeric columns [{acc.inf} found in "
                   f"{sorted(acc.inf_cols)[:5]}]", acc.inf == 0))
    checks.append((f"no weekend trade_date [{acc.weekend} found]", acc.weekend == 0))
    checks.append((f"no lookahead (reference_date <= trade_date) [{acc.lookahead} violations]",
                   acc.lookahead == 0))
    checks.append((f"fundamentals respect filing date [{acc.trade_before_avail} rows before "
                   f"availability, {acc.avail_before_ref} filed pre-quarter-end]",
                   acc.trade_before_avail == 0 and acc.avail_before_ref == 0))
    checks.append((f"has_fundamentals=0 rows have NaN fundamentals {acc.leak}",
                   sum(acc.leak.values()) == 0))
    checks.append((f"NaN/non-positive adj_close always flagged degraded=1 "
                   f"[{acc.bad_adj_unflagged} unflagged of {acc.bad_adj} such rows]",
                   acc.bad_adj_unflagged == 0))
    checks.append((f"flag columns are 0/1 with no NaN [{acc.flag_domain} violations]",
                   acc.flag_domain == 0))

    macro_ok, macro_detail = _macro_is_us(acc)
    checks.append((f"macro is US-sourced, not BR's [{macro_detail}]", macro_ok))

    earn_share = acc.cagr["earn_explained"] / max(acc.cagr["earn_nan"], 1)
    rev_share = acc.cagr["rev_explained"] / max(acc.cagr["rev_nan"], 1)
    checks.append((f"cagr_earnings NaN attributable [{earn_share:.1%} of "
                   f"{acc.cagr['earn_nan']:,}, floor {CAGR_EARNINGS_EXPLAINED_FLOOR:.0%}]",
                   earn_share >= CAGR_EARNINGS_EXPLAINED_FLOOR))
    checks.append((f"cagr_revenue NaN attributable [{rev_share:.1%} of "
                   f"{acc.cagr['rev_nan']:,}, floor {CAGR_REVENUE_EXPLAINED_FLOOR:.0%}]",
                   rev_share >= CAGR_REVENUE_EXPLAINED_FLOOR))

    checks.append((f"no duplicate (ticker, trade_date) [{acc.dupes} found]", acc.dupes == 0))
    checks.append((f"every ticker has >= {MIN_ROWS_PER_TICKER} rows [min {acc.min_rows}, "
                   f"{len(acc.short_tickers)} short]", not acc.short_tickers))

    frozen_rate = acc.frozen_pl / max(acc.eligible_pl, 1)
    checks.append((f"P/L varies daily within quarter on traded days "
                   f"[{acc.frozen_pl}/{acc.eligible_pl} frozen = {frozen_rate:.3%}]",
                   acc.eligible_pl > 0 and frozen_rate < FROZEN_PL_RATE_CEILING))

    regress_rate = len(acc.ref_regress_tickers) / max(acc.tickers, 1)
    checks.append((f"reference_date never goes backwards as trade_date advances "
                   f"[{acc.ref_regress_rows:,} rows over {len(acc.ref_regress_tickers)} tickers "
                   f"= {regress_rate:.2%}, ceiling {REF_REGRESSION_TICKER_CEILING:.0%}] "
                   f"e.g. {acc.ref_regress_tickers[:5]}",
                   regress_rate <= REF_REGRESSION_TICKER_CEILING))

    stale_rate = acc.stale / max(acc.hf_rows, 1)
    checks.append((f"fundamentals are not more than {STALE_FUNDAMENTAL_DAYS}d stale "
                   f"[{acc.stale:,}/{acc.hf_rows:,} = {stale_rate:.2%}, ceiling "
                   f"{STALE_FUNDAMENTAL_RATE_CEILING:.0%}]",
                   stale_rate <= STALE_FUNDAMENTAL_RATE_CEILING))

    checks.append((f"no narrow interior NaN hole across equity+net_income+total_assets "
                   f"[{len(acc.prefix_suspicious)} tickers] e.g. {acc.prefix_suspicious[:5]}",
                   not acc.prefix_suspicious))

    failed = 0
    for label, ok in checks:
        print_check(label, ok)
        failed += not ok

    print_separator()
    print(f"{len(checks) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
