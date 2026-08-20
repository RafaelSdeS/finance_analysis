"""§1 guard from docs/DATA_LAYER_CORRECTNESS_PLAN.md: money must be in one
scale (full currency units, never thousands) and the `*_margin` columns must
share one convention (percent, never a mix of percent and fraction).

The four scale identities (vpa*shares==equity, etc.) are asserted PER TICKER,
worst offender reported -- not a pooled median. A pooled statistic over a
panel that mixes correctly-scaled and still-in-thousands tickers reads
deceptively close to correct (measured in the plan: a 497/115-ticker mixed
panel pooled to ~1.0 median and hid a clean 1000x split). See the plan's
tolerance banner for why every identity here uses a 10% band, not a tight one.

The margin-scale check is POOLED instead, deliberately -- see its own
docstring for why per-ticker doesn't work there.

Usage: python tests/build_dataset/test_unit_scale_invariants.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.build_dataset.paths import OUTPUT_PATH, US_OUTPUT_PATH  # noqa: E402
from test_utils import print_header, print_check, print_section_end  # noqa: E402

BAND = 0.10  # DATA_LAYER_CORRECTNESS_PLAN.md tolerance banner: vendors legitimately
             # disagree by a few percent; the real bugs this catches are off by 100-1000x.
MIN_ROWS = 5  # a median over fewer than this isn't a meaningful per-ticker signal

# Order-of-magnitude guard, NOT an identity, and POOLED (not per-ticker) --
# unlike the four scale identities above, a per-ticker check here produces
# false positives: a near-zero-net_revenue distress quarter sends ebit_margin/
# net_margin into the millions of percent (CLAUDE.md documents this as
# intentional, kept unclipped -- e.g. OBTC3 measured at -31,023,867%), which
# swamps a per-ticker spread check without being a scale bug at all. The
# original §2c bug's own signature was a UNIFORM panel-wide factor (ratio
# exactly 0.01 at every quantile p01-p99), so a pooled median -- computed only
# over rows where both margins are in a plausible range, filtering out the
# distress blowups -- reproduces exactly how that bug was actually found.
ANCHOR_MARGIN = "gross_margin"  # dense in BR (CLAUDE.md), assumed correctly scaled
COMPARE_MARGINS = ["ebit_margin", "net_margin", "ebitda_margin"]
REASONABLE_MARGIN_BOUND = 1000.0  # exclude near-zero-denominator distress blowups
MARGIN_RATIO_BAND = 20.0  # measured healthy cross-margin ratios stay under ~6x;
                          # the ebitda_margin bug produced ~86-100x
MARGIN_COLS = [ANCHOR_MARGIN, *COMPARE_MARGINS]

# label -> ratio(df) that should equal 1.0 on every valid row if the scale is right
IDENTITIES = [
    ("vpa*shares == equity",       lambda df: (df["vpa"] * df["shares_outstanding"]) / df["equity"]),
    ("lpa*shares == net_income",   lambda df: (df["lpa"] * df["shares_outstanding"]) / df["net_income"]),
    ("market_cap/shares == close", lambda df: (df["market_cap"] / df["shares_outstanding"]) / df["close"]),
    ("book_to_market*pvp == 1",    lambda df: df["book_to_market"] * df["pvp"]),
]


def check_identity(df, fn):
    """Per-ticker median ratio; fails if the WORST ticker sits outside the band."""
    ratio = fn(df)
    ratio = ratio.where(np.isfinite(ratio))
    stats = ratio.groupby(df["ticker"]).agg(["median", "count"])
    stats = stats[stats["count"] >= MIN_ROWS]
    if stats.empty:
        return True, "no ticker has enough valid rows to assess", []

    off = (stats["median"] - 1.0).abs()
    failing = off[off > BAND].sort_values(ascending=False)
    worst = [f"{t} (ratio {stats.loc[t, 'median']:.4f}, n={int(stats.loc[t, 'count'])})"
             for t in failing.index[:5]]
    detail = f"{len(failing)}/{len(stats)} tickers outside {BAND:.0%} band"
    return failing.empty, detail, worst


def check_margin_scale(df):
    """All *_margin columns must read as the same convention, checked pooled
    against the ANCHOR_MARGIN (see module docstring for why not per-ticker)."""
    if ANCHOR_MARGIN not in df.columns:
        return True, f"{ANCHOR_MARGIN} not in this dataset", []

    anchor = df[ANCHOR_MARGIN]
    in_bound = np.isfinite(anchor) & (anchor.abs().between(0.5, REASONABLE_MARGIN_BOUND))

    ok = True
    detail_parts, worst = [], []
    for col in COMPARE_MARGINS:
        if col not in df.columns:
            continue
        other = df[col]
        valid = in_bound & np.isfinite(other) & (other.abs().between(0.5, REASONABLE_MARGIN_BOUND))
        n = int(valid.sum())
        if n < 100:
            detail_parts.append(f"{col}: too few clean rows (n={n}), skipped")
            continue
        ratio = (other[valid] / anchor[valid]).median()
        pair_ok = (1.0 / MARGIN_RATIO_BAND) <= ratio <= MARGIN_RATIO_BAND
        ok &= pair_ok
        detail_parts.append(f"{col}/{ANCHOR_MARGIN}={ratio:.3f} (n={n})")
        if not pair_ok:
            worst.append(f"{col} vs {ANCHOR_MARGIN}: pooled median ratio {ratio:.4f} (n={n})")

    return ok, ", ".join(detail_parts), worst


def run(path, market):
    print_header(f"UNIT-SCALE INVARIANTS -- {market}")

    if not path.exists():
        print(f"  SKIPPED: {path} not found (dataset not built)")
        return True

    needed = {"ticker", "vpa", "lpa", "equity", "net_income", "shares_outstanding",
              "market_cap", "close", "book_to_market", "pvp", *MARGIN_COLS}
    df = pd.read_parquet(path, columns=sorted(needed))

    passed = failed = 0
    for label, fn in IDENTITIES:
        ok, detail, worst = check_identity(df, fn)
        print_check(f"{label}  [{detail}]", ok)
        for w in worst:
            print(f"      worst: {w}")
        passed, failed = passed + ok, failed + (not ok)

    ok, detail, worst = check_margin_scale(df)
    print_check(f"*_margin columns share one convention  [{detail}]", ok)
    for w in worst:
        print(f"      worst: {w}")
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
