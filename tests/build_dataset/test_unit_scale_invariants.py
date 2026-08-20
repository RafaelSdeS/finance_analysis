"""§1 guard from docs/DATA_LAYER_CORRECTNESS_PLAN.md: money must be in one
scale (full currency units, never thousands) and all four `*_margin` columns
must share one convention (percent, never a mix of percent and fraction).

Written FIRST, before the §1 normalization fix lands -- expected to FAIL on
BR today (book_to_market*pvp reads ~0.001, not 1.0) and pass once the raw
CVM fundamentals are rebuilt in real currency units. US already passes
(unit_scale=1 at the source).

Asserted per ticker, worst offender reported -- not a pooled median. A
pooled statistic over a panel that mixes 497 correctly-scaled tickers with
115 still-in-thousands tickers reads deceptively close to correct (measured
in the plan: pooled median ~1.0 hides a clean 1000x split). See the plan's
tolerance banner for why every identity here uses a 10% band, not a tight one.

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

# Order-of-magnitude guard, NOT an identity -- margins legitimately differ
# ticker to ticker, so this isn't held to the 10% band. Measured on this
# dataset's healthy siblings: worst normal pairwise spread is ~5x (gross vs
# net margin). The ebitda_margin bug this guards against (§2c) produced ~86x.
MARGIN_SPREAD_MAX = 10.0
MARGIN_COLS = ["gross_margin", "ebit_margin", "net_margin", "ebitda_margin"]

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
    """All four *_margin columns must read as the same convention per ticker."""
    present = [c for c in MARGIN_COLS if c in df.columns]
    med = df.groupby(df["ticker"])[present].median()
    cnt = df.groupby(df["ticker"])[present].count()

    failing = {}
    for ticker in med.index:
        vals = {c: abs(med.at[ticker, c]) for c in present
                 if cnt.at[ticker, c] >= MIN_ROWS
                 and pd.notna(med.at[ticker, c])
                 and abs(med.at[ticker, c]) > 1e-6}
        if len(vals) < 2:
            continue
        spread = max(vals.values()) / min(vals.values())
        if spread > MARGIN_SPREAD_MAX:
            failing[ticker] = spread

    worst_tickers = sorted(failing, key=failing.get, reverse=True)[:5]
    worst = [f"{t} (spread {failing[t]:.1f}x)" for t in worst_tickers]
    detail = f"{len(failing)}/{len(med)} tickers with margin spread > {MARGIN_SPREAD_MAX:.0f}x"
    return not failing, detail, worst


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
