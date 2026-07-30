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
  1. item6.py's footnote-marker parsing bug (fixed 2026-07-30, see
     test_sec_item6.py) produced accounting-impossible negative total_assets
     for a small number of tickers (NEM, ORCL, ZION) collected before the
     fix. Those on-disk files are stale, not re-fixed by re-running this
     test -- a fresh `collect_fundamentals_us` pass is needed to actually
     regenerate them (collect_fundamentals_us has no auto-refresh of
     already-collected tickers; see fix plan's pothole #8 note). The rate
     ceiling below tolerates today's small known-stale count while still
     catching a systemic regression.
  2. Everything else checked here (Inf, OHLC bracket sanity, macro
     completeness) was already clean at audit time -- these are hard
     zero-tolerance assertions, not rate ceilings.

Skips gracefully (prints SKIP, still exits 0) if data/raw/us isn't collected
yet -- it's gitignored, unlike BR's git-tracked data/raw/.

Run from project root:
    python tests/data_collection/test_us_data_quality.py
"""

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
    inf_files = []
    for f in files:
        df = pd.read_parquet(f)
        r = validate.validate_prices(df)
        if not r.passed:
            validate_errors += 1
            print(f"FAIL  prices/{f.stem}: {r.errors}")
        if _has_inf(df):
            inf_files.append(f.stem)

    ok = validate_errors == 0 and not inf_files
    if inf_files:
        print(f"FAIL  prices: Inf present in {len(inf_files)} file(s): {inf_files[:10]}")
    print(f"{'PASS' if ok else 'FAIL'}  prices: {len(files)} files, "
          f"{validate_errors} validate_prices errors, {len(inf_files)} with Inf")
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

    neg_rate = len(neg_assets_files) / len(files)
    ok = gate_errors == 0 and not inf_files and neg_rate <= _NEGATIVE_ASSETS_RATE_CEILING
    if inf_files:
        print(f"FAIL  fundamentals: Inf present in {len(inf_files)} file(s): {inf_files[:10]}")
    if neg_rate > _NEGATIVE_ASSETS_RATE_CEILING:
        print(f"FAIL  fundamentals: negative total_assets rate {neg_rate:.2%} exceeds "
              f"{_NEGATIVE_ASSETS_RATE_CEILING:.0%} ceiling -- {neg_assets_files}")
    elif neg_assets_files:
        print(f"note  fundamentals: {len(neg_assets_files)} known-stale file(s) with negative "
              f"total_assets, within ceiling: {neg_assets_files}")
    print(f"{'PASS' if ok else 'FAIL'}  fundamentals: {len(files)} files, {gate_errors} gate errors, "
          f"{len(inf_files)} with Inf, {len(neg_assets_files)} with negative total_assets "
          f"({neg_rate:.2%}), {total_warnings} total warnings")
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
