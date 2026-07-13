"""
test_merge_save_new_rows_only.py
=================================
_merge_save must validate only the newly-fetched batch, not the full merged
history: a row already accepted onto disk in a previous run (e.g. a known
vendor data glitch from years ago) must not permanently block ingestion of
new, valid rows. Also checks validate_prices tolerates float noise at the
OHLC bracket boundary instead of flagging it as a violation.

Usage:
    python tests/data_collection/test_merge_save_new_rows_only.py
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection import validate
from src.data_collection.collectors import _merge_save


def _row(date, price=10.0):
    return {
        "ticker": "TEST3", "trade_date": date,
        "open": price, "high": price, "low": price, "close": price,
        "adj_open": price, "adj_high": price, "adj_low": price, "adj_close": price,
        "volume": 100, "volume_adjusted": 100, "traded_amount": 1000.0, "num_trades": 5.0,
    }


def test_old_bad_row_does_not_block_new_good_row():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "TEST3.parquet"

        # Old row already on disk with a non-positive open (real-world: a vendor
        # glitch accepted before this check existed, or before it was widened).
        bad_old = pd.DataFrame([_row(pd.Timestamp("2020-01-01"), price=10.0)])
        bad_old.loc[0, "open"] = 0.0
        bad_old.to_parquet(path)

        # New, valid row.
        good_new = pd.DataFrame([_row(pd.Timestamp("2026-01-02"), price=11.0)])

        saved = _merge_save(good_new, path, "trade_date", validate.validate_prices, "TEST3")
        assert saved is not None, "new valid row must save despite old bad row on disk"
        assert len(saved) == 2

    print("OK: old bad row on disk doesn't block a new valid row")


def test_bad_row_in_new_batch_still_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "TEST3.parquet"
        bad_new = pd.DataFrame([_row(pd.Timestamp("2026-01-02"), price=10.0)])
        bad_new.loc[0, "open"] = 0.0

        saved = _merge_save(bad_new, path, "trade_date", validate.validate_prices, "TEST3")
        assert saved is None, "a bad row in the freshly-fetched batch must still block"

    print("OK: bad row within the new batch itself is still caught")


def test_bracket_check_tolerates_float_noise():
    df = pd.DataFrame([_row(pd.Timestamp("2026-01-02"), price=10.0)])
    # Same value to float noise, not a real bracket violation.
    df.loc[0, "adj_open"] = 3.6921889781951910
    df.loc[0, "adj_close"] = 3.6921889781951904
    df.loc[0, ["adj_high", "adj_low"]] = 3.6921889781951904
    vr = validate.validate_prices(df)
    assert vr.passed, f"float noise should not fail validation: {vr.errors}"
    print("OK: bracket check tolerates float noise at the boundary")


if __name__ == "__main__":
    test_old_bad_row_does_not_block_new_good_row()
    test_bad_row_in_new_batch_still_blocks()
    test_bracket_check_tolerates_float_noise()
