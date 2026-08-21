"""
test_refresh_tail_only.py
==========================
Self-check for the fast-refresh path added 2026-08-12 (no network; mocks
every internal that touches yfinance/disk):

  - _prices_fetch_start(tail_only=True) fetches only since the last stored
    row, once the on-disk yfinance-era span is trusted (>= TRUSTED_MIN_YF_ROWS).
  - tail_only=False still reproduces the old full-yfinance-era-refetch
    behavior byte for byte (regression guard).
  - a THIN on-disk file (below TRUSTED_MIN_YF_ROWS) falls through to the deep
    floor even with tail_only=True -- the one branch that could silently
    regress and permanently truncate a ticker's history.
  - collect_dividends_yf's returned changed-set catches a split with NO
    dividend in the same window (the early "no new dividend rows" return
    must not skip the split check).

Usage: python tests/data_collection/test_refresh_tail_only.py
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_collection.yf import _common as common
from src.data_collection.yf import dividends as yfc


def _write_prices_fixture(path, dates, num_trades_nan=True):
    pd.DataFrame({
        "trade_date": pd.to_datetime(dates),
        "num_trades": [float("nan") if num_trades_nan else 1.0] * len(dates),
    }).to_parquet(path, index=False)


def test_tail_only_true_starts_after_last_stored_row():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "AAPL.parquet"
        dates = pd.bdate_range("2026-01-01", periods=common.TRUSTED_MIN_YF_ROWS + 5)
        _write_prices_fixture(path, dates)

        start = common._prices_fetch_start({}, "AAPL", path, tail_only=True)
        expected = (dates.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        assert start == expected, f"expected {expected}, got {start}"
    print("OK: tail_only=True starts the day after the last stored row")


def test_tail_only_false_still_starts_at_earliest_yfinance_row():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "AAPL.parquet"
        dates = pd.bdate_range("2026-01-01", periods=common.TRUSTED_MIN_YF_ROWS + 5)
        _write_prices_fixture(path, dates)

        start = common._prices_fetch_start({}, "AAPL", path)  # tail_only defaults False
        expected = str(dates.min().date())
        assert start == expected, f"expected {expected}, got {start}"
    print("OK: tail_only=False (default) is unchanged -- refetches the whole yfinance era")


def test_thin_file_ignores_tail_only_and_falls_back_to_floor():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "GRTX.parquet"
        dates = pd.bdate_range("2026-01-01", periods=common.TRUSTED_MIN_YF_ROWS - 1)  # below trust floor
        _write_prices_fixture(path, dates)

        start = common._prices_fetch_start({}, "GRTX", path, floor="1900-01-01", tail_only=True)
        assert start == "1900-01-01", \
            f"a thin/possibly-truncated file must ignore tail_only and use the deep floor, got {start}"
    print("OK: a thin on-disk span ignores tail_only=True and still refetches from the floor")


def test_dividends_changed_set_catches_split_with_no_dividend():
    hist_by_ticker = {
        # split only -- must still land in `changed`
        "SPLIT_ONLY": pd.DataFrame(
            {"Dividends": [0.0], "Stock Splits": [2.0]},
            index=pd.to_datetime(["2026-03-01"]),
        ),
        # neither -- must NOT land in `changed`
        "QUIET": pd.DataFrame(
            {"Dividends": [0.0], "Stock Splits": [0.0]},
            index=pd.to_datetime(["2026-03-01"]),
        ),
    }

    class FakeTicker:
        def __init__(self, symbol):
            self._ticker = symbol.replace(".SA", "")

        def history(self, **kwargs):
            return hist_by_ticker[self._ticker]

    with mock.patch.object(yfc, "checkpoint") as mock_cp, \
         mock.patch.object(yfc, "yf") as mock_yf, \
         mock.patch.object(yfc, "_retry", side_effect=lambda fn, *a, **k: fn()), \
         mock.patch.object(yfc, "_merge_save", side_effect=lambda df, *a, **k: df), \
         mock.patch.object(yfc, "sleep"):
        mock_cp.load.return_value = {}
        mock_yf.Ticker.side_effect = FakeTicker
        changed = yfc.collect_dividends_yf(["SPLIT_ONLY", "QUIET"], mode="test",
                                            dividend_dir=Path(tempfile.mkdtemp()),
                                            suffix="", floor="1900-01-01")

    assert changed == {"SPLIT_ONLY"}, f"expected only SPLIT_ONLY in changed set, got {changed}"
    print("OK: a split with no dividend still marks the ticker changed")


if __name__ == "__main__":
    test_tail_only_true_starts_after_last_stored_row()
    test_tail_only_false_still_starts_at_earliest_yfinance_row()
    test_thin_file_ignores_tail_only_and_falls_back_to_floor()
    test_dividends_changed_set_catches_split_with_no_dividend()
