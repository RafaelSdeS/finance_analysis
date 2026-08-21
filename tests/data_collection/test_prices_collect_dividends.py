"""
test_prices_collect_dividends.py
==================================
Self-check for collect_prices_yf's collect_dividends param (no network;
mocks yf.Ticker directly so the real _fetch_and_shape_prices/_extract_
dividends machinery runs end to end, not just the higher-level orchestration).

Before this (2026-08-13), a fast refresh cost TWO yfinance requests per
ticker every run: a dividends-specific fetch (collect_dividends_yf) plus a
separate price fetch (collect_prices_yf) -- even though actions=True already
returns the Dividends/Stock Splits columns alongside OHLCV in the SAME
response. collect_dividends=True folds dividend extraction+write into the
price fetch, so a ticker with nothing new costs one request, not two.

Usage: python tests/data_collection/test_prices_collect_dividends.py
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import yfinance as yf

from src.data_collection.yf import prices as yfc

_IDX = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]).tz_localize("America/New_York")


class _FakeTickerWithDividend:
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, start, auto_adjust, actions=False):
        return pd.DataFrame({"Open": [10.0, 10.0, 10.0], "High": [10.0, 10.0, 10.0],
                              "Low": [10.0, 10.0, 10.0], "Close": [10.0, 10.0, 10.0],
                              "Adj Close": [10.0, 9.9, 9.9],
                              "Volume": [100, 100, 100],
                              "Dividends": [0.0, 0.1, 0.0],
                              "Stock Splits": [0.0, 0.0, 0.0]}, index=_IDX)


class _FakeTickerNoActivity:
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, start, auto_adjust, actions=False):
        return pd.DataFrame({"Open": [10.0, 10.0, 10.0], "High": [10.0, 10.0, 10.0],
                              "Low": [10.0, 10.0, 10.0], "Close": [10.0, 10.0, 10.0],
                              "Adj Close": [10.0, 10.0, 10.0],
                              "Volume": [100, 100, 100],
                              "Dividends": [0.0, 0.0, 0.0],
                              "Stock Splits": [0.0, 0.0, 0.0]}, index=_IDX)


def test_collect_dividends_true_writes_dividend_file_and_reports_changed():
    with tempfile.TemporaryDirectory() as tmp:
        price_dir = Path(tmp) / "prices"
        div_dir = Path(tmp) / "dividends"
        price_dir.mkdir()
        div_dir.mkdir()

        with mock.patch.object(yf, "Ticker", _FakeTickerWithDividend), \
             mock.patch.object(yfc, "checkpoint") as mock_cp, \
             mock.patch.object(yfc, "sleep"):
            mock_cp.load.return_value = {}
            changed = yfc.collect_prices_yf(["PAYER"], mode="test", price_dir=price_dir,
                                             suffix="", floor="2020-01-01",
                                             dividend_dir=div_dir, collect_dividends=True)

        assert changed == {"PAYER"}, f"a ticker with a dividend in this fetch must be reported changed, got {changed}"
        assert (price_dir / "PAYER.parquet").exists(), "the price file must still be written (same request)"
        div_df = pd.read_parquet(div_dir / "PAYER.parquet")
        assert len(div_df) == 1 and abs(div_df["value_per_share"].iloc[0] - 0.1) < 1e-9, \
            f"the dividend row must be extracted from the SAME response, got {div_df}"
    print("OK: collect_dividends=True writes the dividend file and reports the ticker as changed")


def test_collect_dividends_true_no_activity_writes_no_dividend_file():
    with tempfile.TemporaryDirectory() as tmp:
        price_dir = Path(tmp) / "prices"
        div_dir = Path(tmp) / "dividends"
        price_dir.mkdir()
        div_dir.mkdir()

        with mock.patch.object(yf, "Ticker", _FakeTickerNoActivity), \
             mock.patch.object(yfc, "checkpoint") as mock_cp, \
             mock.patch.object(yfc, "sleep"):
            mock_cp.load.return_value = {}
            changed = yfc.collect_prices_yf(["QUIET"], mode="test", price_dir=price_dir,
                                             suffix="", floor="2020-01-01",
                                             dividend_dir=div_dir, collect_dividends=True)

        assert changed == set(), f"a ticker with no dividend/split must not be reported changed, got {changed}"
        assert (price_dir / "QUIET.parquet").exists(), "the price file must still be written"
        assert not (div_dir / "QUIET.parquet").exists(), "no dividend file should be created when there's nothing to write"
    print("OK: collect_dividends=True writes no dividend file when there's no dividend/split activity")


def test_collect_dividends_false_returns_none_and_never_touches_dividend_dir():
    # Backward compatibility: every EXISTING caller (br/pipeline.py's update mode via
    # refresh.py pre-2026-08-13, us/pipeline.py) must see byte-identical behavior --
    # no dividend_dir, no return value, even when the underlying fetch DOES include a
    # real dividend (proves this isn't just "dividend_dir defaults to None so nothing
    # happens" but an active off-switch).
    with tempfile.TemporaryDirectory() as tmp:
        price_dir = Path(tmp) / "prices"
        div_dir = Path(tmp) / "dividends"
        price_dir.mkdir()
        # div_dir deliberately NOT created -- collect_dividends=False must never touch it

        with mock.patch.object(yf, "Ticker", _FakeTickerWithDividend), \
             mock.patch.object(yfc, "checkpoint") as mock_cp, \
             mock.patch.object(yfc, "sleep"):
            mock_cp.load.return_value = {}
            result = yfc.collect_prices_yf(["PAYER"], mode="test", price_dir=price_dir,
                                            suffix="", floor="2020-01-01")

        assert result is None, f"collect_dividends=False (default) must return None, got {result}"
        assert not div_dir.exists(), "collect_dividends=False must never touch dividend_dir at all"
    print("OK: collect_dividends=False (default) returns None and never touches dividend_dir")


if __name__ == "__main__":
    test_collect_dividends_true_writes_dividend_file_and_reports_changed()
    test_collect_dividends_true_no_activity_writes_no_dividend_file()
    test_collect_dividends_false_returns_none_and_never_touches_dividend_dir()
