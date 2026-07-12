"""
Test 1a (delisted price backfill): candidate-list filter + delisting-date anchors.

The candidate filter is pure code and always runs. The anchor checks need the
delisted parquets on disk (python -m src.data_collection.collect_delisted) and
SKIP gracefully until then — they are the regression net that catches the API
silently returning stale/extended data for a dead ticker.

Run from project root:
    python tests/data_collection/test_collect_delisted.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_collection.collect_delisted import candidate_tickers  # noqa: E402

# Known true last-trade dates, verified live against /stocks/{t}/history 2026-07-11
DELISTING_ANCHORS = {
    "SMLS3": "2021-06-04",   # Smiles: incorporated into GOL
    "LAME4": "2022-01-21",   # Lojas Americanas: combination into AMER3
    "HGTX3": "2021-09-17",   # Cia Hering: merged into Grupo Soma
}
ANCHOR_TOLERANCE_DAYS = 7


def test_candidate_filter():
    universe = ["PETR4", "SMLS3", "A1AP34", "HGLG11", "SULA11", "BOVA11", "XPTO3"]
    got = candidate_tickers(universe, existing=["PETR4"], crosswalk_tickers={"SULA11"})
    # BDR (A1AP34) out on shape; fund 11 (HGLG11) out — not in crosswalk;
    # unit 11 (SULA11) in via crosswalk; benchmark (BOVA11) always out
    assert got == ["SMLS3", "SULA11", "XPTO3"], got

    # without a crosswalk, no suffix-11 name may pass
    got = candidate_tickers(universe, existing=[], crosswalk_tickers=None)
    assert "SULA11" not in got and "HGLG11" not in got, got
    print("PASS  candidate filter")
    return True


def test_delisting_anchors():
    all_ok, skipped = True, 0
    for ticker, expected in DELISTING_ANCHORS.items():
        path = ROOT / f"data/raw/prices/{ticker}.parquet"
        if not path.exists():
            print(f"SKIP  {ticker}: not collected yet (run collect_delisted)")
            skipped += 1
            continue
        last = pd.read_parquet(path)["trade_date"].max()
        exp = pd.Timestamp(expected)
        if abs((last - exp).days) > ANCHOR_TOLERANCE_DAYS:
            print(f"FAIL  {ticker}: last trade {last.date()}, expected ~{expected}")
            all_ok = False
        else:
            print(f"PASS  {ticker}: last trade {last.date()} (anchor {expected})")
    if skipped == len(DELISTING_ANCHORS):
        print("note: all anchors skipped — backfill not run yet, filter test still counts")
    return all_ok


if __name__ == "__main__":
    ok = test_candidate_filter() & test_delisting_anchors()
    sys.exit(0 if ok else 1)
