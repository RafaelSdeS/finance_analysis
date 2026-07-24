"""
Universe integrity checks for ml_dataset.parquet — see
TOP50_UNIVERSE_VALIDATION.md §3 for the why/how/threshold behind each check.

Wired into run_all.py's DATA group: 3.1 (survivorship) is explicitly a
regression guard per its own docstring below ("silently reintroduces the
exact bias this whole analysis is about") -- that only holds if this file
actually runs on every build, not just when someone remembers to invoke it
by hand.

Run from project root:
    python tests/build_dataset/test_universe_integrity.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.build_dataset.paths import (  # noqa: E402
    COMPANY_INFO_PATH, CONTINUITY_PATH, CVM_CROSSWALK_PATH, OUTPUT_PATH, PRICES_DIR,
)
from src.build_dataset.loaders import company_siblings  # noqa: E402
from src.build_dataset.quality_filters import QUARANTINED_TICKERS  # noqa: E402
from test_utils import print_header, print_check, print_separator  # noqa: E402

# 3.2: columns the ml_agent branch is known to depend on, and their expected
# dtype family. Datetime columns checked separately (is_datetime64_any_dtype).
EXPECTED_DTYPES = {
    "ticker": "object",
    "close": "float",
    "adj_close": "float",
    "volume": "int",
    "has_fundamentals": "float",  # features.py:304 casts explicitly to float
    "has_dividends": "int",
}
EXPECTED_DATETIME_COLS = ["trade_date"]

# 3.1: floor ratio of raw CANCELADA tickers that must still be present in the
# final dataset with a usable amount of history. Generous on purpose --
# legitimate drops happen (delisted co. that never filed fundamentals).
SURVIVORSHIP_FLOOR_RATIO = 0.6
MIN_ROWS_PER_SURVIVING_TICKER = 10

# 3.4: informational only, not a failure condition.
SIBLING_CORR_WARN_THRESHOLD = 0.5


def _dtype_family(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    return "object"


def check_schema_contract(df):
    checks = []
    for col, expected in EXPECTED_DTYPES.items():
        if col not in df.columns:
            checks.append((f"column present: {col}", False))
            continue
        actual = _dtype_family(df[col].dtype)
        checks.append((f"{col} dtype is {expected} [got {df[col].dtype}]", actual == expected))

    for col in EXPECTED_DATETIME_COLS:
        if col not in df.columns:
            checks.append((f"column present: {col}", False))
            continue
        ok = pd.api.types.is_datetime64_any_dtype(df[col])
        checks.append((f"{col} is datetime64 [got {df[col].dtype}]", ok))

    return checks


def check_survivorship(df, company_info):
    checks = []
    cancelada_raw = set(company_info.loc[company_info["status"] == "CANCELADA", "ticker"])
    if not cancelada_raw:
        checks.append(("company_info has CANCELADA tickers to check against", False))
        return checks, pd.DataFrame()

    row_counts = df.groupby("ticker").size()
    surviving = {t: row_counts.get(t, 0) for t in cancelada_raw}
    present = {t: n for t, n in surviving.items() if n >= MIN_ROWS_PER_SURVIVING_TICKER}

    ratio = len(present) / len(cancelada_raw)
    checks.append((
        f"CANCELADA tickers surviving into ml_dataset with >= {MIN_ROWS_PER_SURVIVING_TICKER} rows "
        f"[{len(present)}/{len(cancelada_raw)} = {ratio:.0%}, floor {SURVIVORSHIP_FLOOR_RATIO:.0%}]",
        ratio >= SURVIVORSHIP_FLOOR_RATIO,
    ))

    dropped = pd.DataFrame(
        [(t, surviving[t]) for t in sorted(cancelada_raw) if surviving[t] < MIN_ROWS_PER_SURVIVING_TICKER],
        columns=["ticker", "rows_in_dataset"],
    )
    return checks, dropped


def check_sibling_correlation(df, company_info):
    """Informational: rolling 60d return correlation between same-cvm_code
    sibling tickers (e.g. PETR3/PETR4). Low correlation may flag a
    ticker-mapping/crosswalk bug -- not a hard failure, real share classes
    can legitimately diverge."""
    siblings = company_siblings(company_info)
    pairs = [tuple(tickers) for tickers in siblings.values() if len(tickers) == 2]

    if "log_return" not in df.columns or not pairs:
        return "No sibling pairs or no log_return column; skipping."

    lines = []
    for t1, t2 in pairs:
        s1 = df.loc[df["ticker"] == t1].set_index("trade_date")["log_return"]
        s2 = df.loc[df["ticker"] == t2].set_index("trade_date")["log_return"]
        joined = pd.concat([s1, s2], axis=1, join="inner")
        if len(joined) < 60:
            continue
        corr = joined.iloc[:, 0].rolling(60).corr(joined.iloc[:, 1]).dropna()
        if corr.empty:
            continue
        min_corr = corr.min()
        if min_corr < SIBLING_CORR_WARN_THRESHOLD:
            lines.append(f"  {t1}/{t2}: min 60d rolling corr = {min_corr:.2f} (n={len(joined)})")

    return "\n".join(lines) if lines else "  None below threshold."


def check_status_is_static(df):
    """Informational: confirm `status` (and `sector`) are collection-time
    snapshots held constant across a ticker's entire history, not real
    point-in-time state -- merge_company_info() joins company_info.parquet's
    CURRENT status onto every historical row.

    This means `status` deterministically reveals whether a company is still
    listed TODAY on every row, including rows from a decade before that was
    knowable -- a feature-level survivorship/lookahead trap distinct from the
    universe-selection-level bias 3.1 guards against. Not fixable here
    (downstream point-in-time universe construction, per
    TOP50_UNIVERSE_VALIDATION.md §1, needs this exact current-status column
    to identify delisted names) -- must not be fed to a model as a raw
    per-row feature. Not a hard gate: this documents the property (and
    catches it if a future change makes status genuinely time-varying,
    which would be a welcome improvement, not a regression)."""
    nunique = df.groupby("ticker")["status"].nunique()
    varying = nunique[nunique > 1]
    if len(varying):
        return (f"  {len(varying)}/{len(nunique)} tickers now have time-varying status "
                f"(status is no longer a pure current-snapshot join -- update this note "
                f"and CLAUDE.md's caveat if this is a deliberate improvement).")
    return (f"  status is constant across history for all {len(nunique)} tickers, as expected "
            f"for a current-snapshot join. Do not use `status`/`sector` as a raw per-row "
            f"training feature -- see CLAUDE.md caveat.")


def _cnpj_alias_pairs():
    """Ticker pairs that are the SAME legal entity (identical CNPJ) per the
    real CVM registry (company_info + the free CVM crosswalk, whichever has
    it) -- e.g. ALOS3/ALSO3 (Aliansce Sonae/Allos) and MEGA3/SRNA3 (Omega
    Energia/Serena Energia) both matched here, cnpj-for-cnpj, on the
    2026-07-24 audit. A same-CNPJ pair sharing raw price history is an
    expected vendor-alias duplicate (same company, two ticker mnemonics --
    continuity.py splices these once dated), not corruption -- distinct from
    a same-cnpj-absent pair like BAHI3/CGRA3 or ATOM3/MBLY3, which really are
    two different companies and a genuine data-integrity defect.

    Deliberately CNPJ-based (not just ticker_continuity.json's documented
    events): that json only has entries someone already investigated and
    dated; this catches the same fact pattern even before a continuity entry
    with a verified rename date has been added, using registry cnpj as the
    ground truth instead of the price-series near-duplication itself
    (circular -- the pattern we're trying to explain shouldn't also be the
    evidence excusing it)."""

    def _norm(s):
        return "".join(ch for ch in str(s) if ch.isdigit()) or None

    cnpj_by_ticker = {}
    if COMPANY_INFO_PATH.exists():
        ci = pd.read_parquet(COMPANY_INFO_PATH)
        for t, c in zip(ci["ticker"], ci["cnpj"]):
            n = _norm(c)
            if n:
                cnpj_by_ticker.setdefault(t, n)
    if CVM_CROSSWALK_PATH.exists():
        xw = pd.read_parquet(CVM_CROSSWALK_PATH)
        for t, c in zip(xw["ticker"], xw["cnpj"]):
            n = _norm(c)
            if n:
                cnpj_by_ticker.setdefault(t, n)

    by_cnpj = {}
    for t, c in cnpj_by_ticker.items():
        by_cnpj.setdefault(c, set()).add(t)

    pairs = set()
    for tickers in by_cnpj.values():
        if len(tickers) > 1:
            tl = sorted(tickers)
            pairs.update(frozenset((tl[i], tl[j])) for i in range(len(tl)) for j in range(i + 1, len(tl)))
    return pairs


def check_no_duplicate_price_series():
    """3.6: no two DISTINCT (different-CNPJ) tickers may share a
    near-identical raw OHLCV price series -- (ticker, trade_date) dedup
    elsewhere in the pipeline can't catch this, since it's duplication
    ACROSS tickers, not within one. Found 2026-07-24: BAHI3=CGRA3 (Bahema vs
    Grazziotin -- CGRA3's own dividend history independently corroborates
    the shared series as CGRA3's, so BAHI3's raw price file is the vendor
    copy), ATOM3=MBLY3(=LVTC3) (Atom vs Mobly, side not yet identified),
    GFTT3=GFTT4 (never reaches the final dataset, MIN_PRICE_ROWS-filtered) --
    each group's raw prices/*.parquet is a vendor copy of another distinct
    company's series (fundamentals differ), so every derived price/return/
    volatility/beta feature for the copied side is fabricated. Same failure
    class as the already-quarantined WDCN3/CCTY3, just at the cross-ticker
    level instead of within one file.

    Tolerance-based (not exact-hash): ARND3/PORT3 match to within ~5e-9
    (float32-rounding noise from some vendor-side conversion), not bit-for-
    bit -- an exact hash comparison missed this real duplicate.

    Same-CNPJ pairs (_cnpj_alias_pairs) are excluded: e.g. ALOS3/ALSO3 and
    MEGA3/SRNA3 are the SAME company under two ticker mnemonics (confirmed
    via the CVM registry, not just the price match itself) -- an expected,
    not-yet-continuity-dated vendor alias, not corruption. Documented
    ticker_continuity.json rename/merger events are excluded too, for
    companies where the splice is already dated and handled.

    Bucketed by (row count, first date, last date) before the tolerance
    check so this stays O(n) over ~500 tickers instead of an O(n^2) full
    pairwise scan -- coincidentally matching span+count is itself already
    a strong prior for two tickers being worth a closer look.
    """
    ohlcv_cols = ["trade_date", "open", "high", "low", "close", "volume"]
    frames = {}
    for f in sorted(PRICES_DIR.glob("*.parquet")):
        g = pd.read_parquet(f, columns=ohlcv_cols).sort_values("trade_date").reset_index(drop=True)
        if len(g) >= 2:
            frames[f.stem] = g

    aliased = _cnpj_alias_pairs()
    if CONTINUITY_PATH.exists():
        for e in json.loads(CONTINUITY_PATH.read_text()).get("events", []):
            if e.get("type") in ("rename", "merger") and e.get("old") and e.get("new"):
                aliased.add(frozenset((e["old"], e["new"])))

    buckets = {}
    for t, g in frames.items():
        key = (len(g), g["trade_date"].iloc[0], g["trade_date"].iloc[-1])
        buckets.setdefault(key, []).append(t)

    in_dataset = set(pd.read_parquet(OUTPUT_PATH, columns=["ticker"])["ticker"].unique())
    dupes = []
    for tickers in buckets.values():
        if len(tickers) < 2:
            continue
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                t1, t2 = tickers[i], tickers[j]
                if frozenset((t1, t2)) in aliased:
                    continue
                if t1 in QUARANTINED_TICKERS or t2 in QUARANTINED_TICKERS:
                    continue  # already documented + excluded from the built dataset
                g1, g2 = frames[t1], frames[t2]
                if (np.allclose(g1["close"], g2["close"], rtol=1e-5, atol=1e-4)
                        and np.allclose(g1["volume"], g2["volume"], rtol=1e-5, atol=1)):
                    tag = " [IN final dataset]" if t1 in in_dataset or t2 in in_dataset else ""
                    dupes.append(f"{t1}={t2}{tag}")

    label = ("no near-duplicate raw price series across distinct tickers"
              if not dupes else
              f"no near-duplicate raw price series across distinct tickers [found: {sorted(dupes)}]")
    return [(label, not dupes)]


def main():
    print_separator()
    print("UNIVERSE INTEGRITY TEST (survivorship, schema, sibling consistency)")
    print_separator()

    if not OUTPUT_PATH.exists():
        print(f"\nERROR: file not found:\n{OUTPUT_PATH}")
        sys.exit(1)
    if not COMPANY_INFO_PATH.exists():
        print(f"\nERROR: file not found:\n{COMPANY_INFO_PATH}")
        sys.exit(1)

    df = pd.read_parquet(OUTPUT_PATH)
    company_info = pd.read_parquet(COMPANY_INFO_PATH)

    print(f"\nFile      : {OUTPUT_PATH}")
    print(f"Total rows: {len(df):,}, tickers: {df['ticker'].nunique()}")

    print()
    print_header("3.1 SURVIVORSHIP-BIAS GUARD")
    survivorship_checks, dropped = check_survivorship(df, company_info)

    print()
    print_header("3.2 SCHEMA/DTYPE CONTRACT")
    schema_checks = check_schema_contract(df)

    print()
    print_header("3.6 DUPLICATE PRICE SERIES GUARD")
    dup_checks = check_no_duplicate_price_series()

    failed = 0
    for label, ok in survivorship_checks + schema_checks + dup_checks:
        print_check(label, ok)
        failed += not ok

    if len(dropped):
        print("\nCANCELADA tickers with insufficient history (first 15):")
        print(dropped.head(15).to_string(index=False))

    print()
    print_header("3.4 SIBLING-CORRELATION CHECK (informational)")
    print(check_sibling_correlation(df, company_info))

    print()
    print_header("3.5 STATUS/SECTOR STATIC-SNAPSHOT CHECK (informational)")
    print(check_status_is_static(df))

    print()
    if failed:
        print(f"VALIDATION FAILED: {failed} check(s)")
        sys.exit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
