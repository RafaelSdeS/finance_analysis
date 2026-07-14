"""
Universe integrity checks for ml_dataset.parquet — see
TOP50_UNIVERSE_VALIDATION.md §3 for the why/how/threshold behind each check.

Not part of the pipeline's regression suite (run_all.py) by design: these are
one-off audits of already-collected data, not gates on future rebuilds.

Run from project root:
    python tests/build_dataset/test_universe_integrity.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.build_dataset.paths import COMPANY_INFO_PATH, OUTPUT_PATH  # noqa: E402
from src.build_dataset.loaders import company_siblings  # noqa: E402
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

    failed = 0
    for label, ok in survivorship_checks + schema_checks:
        print_check(label, ok)
        failed += not ok

    if len(dropped):
        print("\nCANCELADA tickers with insufficient history (first 15):")
        print(dropped.head(15).to_string(index=False))

    print()
    print_header("3.4 SIBLING-CORRELATION CHECK (informational)")
    print(check_sibling_correlation(df, company_info))

    print()
    if failed:
        print(f"VALIDATION FAILED: {failed} check(s)")
        sys.exit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
