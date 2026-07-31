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
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_collection import collect_delisted, collectors  # noqa: E402
from src.data_collection.collect_delisted import candidate_tickers  # noqa: E402

# Known true last-trade dates, verified live against /stocks/{t}/history 2026-07-11.
# These are point-in-time facts, not derived from code -- if a vendor later
# backfills more history for one of these tickers, this test starts failing
# with no corresponding code change nearby. That's the signal to re-verify
# the anchor live (not to widen the tolerance or delete the check).
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


def test_main_collects_all_tickers_in_one_call():
    """collect_prices() loads/saves its own checkpoint dict per call and isn't
    safe to invoke concurrently across ticker batches (each call's
    checkpoint.save() overwrites the others' in-memory view). main() must
    call it exactly once with the full candidate list, not batched/threaded
    -- guards against re-introducing the ThreadPoolExecutor split."""
    calls = []
    with mock.patch.object(collectors, "collect_prices",
                            side_effect=lambda t, m: calls.append((list(t), m))), \
         mock.patch.object(sys, "argv", ["collect_delisted", "--tickers", "AAAA3", "BBBB3", "CCCC3"]):
        collect_delisted.main()

    assert len(calls) == 1, f"expected exactly one collect_prices call, got {len(calls)}"
    tickers, mode = calls[0]
    assert tickers == ["AAAA3", "BBBB3", "CCCC3"], tickers
    assert mode == "full_scale"
    print("PASS  main() collects all tickers in a single call")
    return True


if __name__ == "__main__":
    ok = test_candidate_filter() & test_delisting_anchors() & test_main_collects_all_tickers_in_one_call()
    sys.exit(0 if ok else 1)
