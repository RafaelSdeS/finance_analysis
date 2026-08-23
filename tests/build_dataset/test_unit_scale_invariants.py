"""§1 guard from docs/DATA_LAYER_CORRECTNESS_PLAN.md, rewritten 2026-08-23:
money must be in one scale (full currency units, never thousands), every
price-linear valuation ratio must be anchored to the SAME close it is divided
by, and every ratio column must keep its declared percent-vs-fraction
convention.

WHAT CHANGED 2026-08-23, and why it matters
-------------------------------------------
The previous version aggregated each identity to a PER-TICKER median and
allowed a 10% band. That combination is blind to the single largest defect
class this file exists to catch: a per-QUARTER anchor error. When `pl` is
built against the quarter-end close but divided by the daily close, the error
is exactly the realized quarter-end -> filing-date return -- constant inside
each quarter (std ~1e-16) but straddling 1.0 across quarters, so the
per-ticker median lands back on ~1.0 and passes. Measured on the pre-fix BR
build: median relative error 7.4%, 82% of rows off by >2%, and every existing
check here green.

Two consequences, both landed here:

  1. The identities are now asserted ROW-LEVEL, not per-ticker-median. Worst
     offenders are still reported grouped by (ticker, reference_date) because
     that is the shape an anchor bug takes, but the pass/fail is per row.
  2. The band drops 0.10 -> 0.01, because these are ALGEBRAIC identities on a
     single frame, not cross-vendor comparisons. Nothing legitimately makes
     `market_cap` differ from `close * shares_outstanding`.

Calibration (2026-08-23, BR dataset_v7 + US build 2026-08-16): all seven
identities hold at **zero** row-level violations outside a 1% band, over
1.1M BR and 6.2M US valid rows each. That is why this lands as zero
tolerance rather than a rate ceiling -- there is no backlog to tolerate.
Two of the seven (`pl*lpa == close`, `earnings_yield*pl == 1`) were only
made true by the 2026-08-22 `rescale_price_linear_ratios` fix in
`merge.py`/`features.py`; freezing them here is the point.

The old `check_margin_scale()` pooled-median comparison is GONE, replaced by
`SCALE_BANDS` below, which subsumes it: pinning each margin's own absolute
level catches the ~86-100x `ebitda_margin` bug it was written for *and* the
whole percent-vs-fraction class it could not see (it only ever compared
margins against each other, so a uniform rescale of all of them cancelled).

Usage: python tests/build_dataset/test_unit_scale_invariants.py
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
from test_utils import print_header, print_check, print_section_end  # noqa: E402

BAND = 0.01          # algebraic identities on one frame; measured violations at this band: 0
MAX_EXAMPLES = 5     # worst offenders printed per failing identity
BATCH_ROWS = 300_000  # the US file is 5.5 GB / 15.4M rows -- never read whole (see run())
SAMPLE_PER_BATCH = 20_000  # per-column reservoir for the SCALE_BANDS medians

# label -> (columns needed, ratio that must equal 1.0 on every valid row).
#
# The first four are the original §1 scale identities (do the money columns
# agree on one unit?). The last three are the valuation-ANCHOR identities
# added 2026-08-23: they answer a different question -- is the price each
# ratio was divided by the same price stored in `close` on that row? -- and
# they are the ones that were silently false before the 2026-08-22 fix.
IDENTITIES = [
    ("vpa*shares == equity",
     ("vpa", "shares_outstanding", "equity"),
     lambda d: (d["vpa"] * d["shares_outstanding"]) / d["equity"]),
    ("lpa*shares == net_income",
     ("lpa", "shares_outstanding", "net_income"),
     lambda d: (d["lpa"] * d["shares_outstanding"]) / d["net_income"]),
    ("market_cap == close*shares",
     ("market_cap", "shares_outstanding", "close"),
     lambda d: d["market_cap"] / (d["close"] * d["shares_outstanding"])),
    ("book_to_market*pvp == 1",
     ("book_to_market", "pvp"),
     lambda d: d["book_to_market"] * d["pvp"]),
    ("pl*lpa == close",
     ("pl", "lpa", "close"),
     lambda d: (d["pl"] * d["lpa"]) / d["close"]),
    ("pvp*vpa == close",
     ("pvp", "vpa", "close"),
     lambda d: (d["pvp"] * d["vpa"]) / d["close"]),
    ("earnings_yield*pl == 1",
     ("earnings_yield", "pl"),
     lambda d: d["earnings_yield"] * d["pl"]),
]

# T2: the percent-vs-fraction convention is
# real, load-bearing (peg_ratio divides one by the other) and, until this
# table, asserted NOWHERE -- a source change emitting `roe` as a fraction
# instead of percent points passed every test in the suite.
#
# Band = median of |x| over non-zero, finite rows, roughly a decade either
# side of the measured value, so a 100x convention flip in either direction
# is caught on every column while ordinary cross-market level differences
# (US roa 1.6 vs BR roa 5.3) are not. `payout_ratio` gets its own wider band
# because it legitimately sits near 1.0 -- a shared "fraction family" band
# could not separate it from a 100x-flipped `log_return`.
# Measured 2026-08-23 (BR dataset_v7 / US 2026-08-16 build):
SCALE_BANDS = {
    # percent points
    "roe":                 (1.0, 200.0),    # BR 14.05  US  4.18
    "roa":                 (0.3, 100.0),    # BR  5.34  US  1.62
    "roic":                (0.5, 150.0),    # BR  9.45  US  4.01
    "gross_margin":        (3.0, 300.0),    # BR 31.71  US 43.51
    "net_margin":          (1.0, 200.0),    # BR 10.89  US  9.55
    "ebit_margin":         (1.0, 200.0),    # BR 14.68  US 13.78
    "ebitda_margin":       (1.0, 200.0),    # BR 18.21  US  n/a (empty_columns)
    "net_margin_q":        (1.0, 200.0),    # BR 10.79  BR-only
    "roe_q":               (0.3, 100.0),    # BR  3.59  BR-only
    # fractions
    "earnings_yield":      (0.002, 0.5),    # BR 0.098  US 0.0144
    "div_yield_12m":       (0.002, 0.5),    # BR 0.033  US 0.0171
    "payout_ratio":        (0.05, 10.0),    # BR 0.397  US 1.3798  <- straddles 1.0 on purpose
    "revenue_growth_yoy":  (0.01, 1.0),     # BR 0.141  US 0.1113
    "earnings_growth_yoy": (0.05, 5.0),     # BR 0.523  US 0.4598
    "volatility_20d":      (0.002, 0.2),    # BR 0.024  US 0.0194
    "volatility_60d":      (0.002, 0.2),    # BR 0.026  US 0.0205
    "log_return":          (0.002, 0.2),    # BR 0.016  US 0.0123
}
MIN_SCALE_ROWS = 1000  # below this a median isn't a convention signal


def _stream(path, columns):
    """Row-group batches of just `columns`. The US parquet is 5.5 GB and this
    machine has ~9 GB free -- a whole-file read of even 30 columns peaks
    around 4 GB and is exactly the kind of thing that takes the editor with
    it. Batching keeps peak RSS under ~100 MB regardless of market."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=columns):
        yield batch.to_pandas()


def run(path, market):
    print_header(f"UNIT-SCALE + VALUATION-ANCHOR INVARIANTS -- {market}")

    if not path.exists():
        print(f"  SKIPPED: {path} not found (dataset not built)")
        return True

    available = set(pq.read_schema(path).names)
    identities = [(lbl, cols, fn) for lbl, cols, fn in IDENTITIES
                  if set(cols) <= available]
    scale_cols = [c for c in SCALE_BANDS if c in available]

    needed = sorted(({"ticker", "reference_date"} & available)
                    | {c for _, cols, _ in identities for c in cols}
                    | set(scale_cols))

    valid = {lbl: 0 for lbl, _, _ in identities}
    violations = {lbl: 0 for lbl, _, _ in identities}
    examples = {lbl: [] for lbl, _, _ in identities}
    samples = {c: [] for c in scale_cols}

    for chunk in _stream(path, needed):
        for lbl, _, fn in identities:
            ratio = fn(chunk).replace([np.inf, -np.inf], np.nan)
            ok = ratio.notna()
            off = (ratio - 1.0).abs()
            bad = ok & (off > BAND)
            valid[lbl] += int(ok.sum())
            violations[lbl] += int(bad.sum())
            if bad.any() and len(examples[lbl]) < MAX_EXAMPLES:
                worst = off[bad].nlargest(MAX_EXAMPLES).index
                for i in worst[: MAX_EXAMPLES - len(examples[lbl])]:
                    # An anchor bug is constant within a quarter, so the
                    # (ticker, reference_date) pair is what identifies it --
                    # both are best-effort here, a fixture may carry neither.
                    who = " ".join(str(chunk.at[i, c]) for c in ("ticker", "reference_date")
                                   if c in chunk.columns)
                    examples[lbl].append(f"{who} ratio {ratio.at[i]:.4f}")
        for col in scale_cols:
            v = chunk[col].replace([np.inf, -np.inf], np.nan).dropna().abs()
            v = v[v > 0]
            if len(v):
                samples[col].append(v.sample(min(SAMPLE_PER_BATCH, len(v)), random_state=0))

    passed = failed = 0
    for lbl, _, _ in identities:
        n, bad = valid[lbl], violations[lbl]
        ok = n > 0 and bad == 0
        rate = bad / n if n else float("nan")
        print_check(f"{lbl}  [{bad}/{n} rows outside {BAND:.0%} band ({rate:.4%})]", ok)
        for e in examples[lbl]:
            print(f"      worst: {e}")
        passed, failed = passed + ok, failed + (not ok)

    skipped = [lbl for lbl, cols, _ in IDENTITIES if not set(cols) <= available]
    if skipped:
        print(f"      (not applicable to {market}: {', '.join(skipped)})")

    for col in scale_cols:
        if not samples[col]:
            print(f"      (scale: {col} has no non-zero finite rows, skipped)")
            continue
        s = pd.concat(samples[col])
        if len(s) < MIN_SCALE_ROWS:
            print(f"      (scale: {col} only {len(s)} rows sampled, skipped)")
            continue
        lo, hi = SCALE_BANDS[col]
        med = s.median()
        ok = lo <= med <= hi
        print_check(f"scale convention {col}  [median |x| = {med:.4f}, expected {lo}-{hi}]", ok)
        passed, failed = passed + ok, failed + (not ok)

    print_section_end(passed, failed)
    return failed == 0


def main():
    br_ok = run(OUTPUT_PATH, "BR")
    us_ok = run(US_OUTPUT_PATH, "US")
    if not (br_ok and us_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
