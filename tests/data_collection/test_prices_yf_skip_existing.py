"""
test_prices_yf_skip_existing.py
=================================
Self-check for collect_prices_yf's skip_existing param (no network; mocks
every internal that touches yfinance/disk).

Unlike BolsAI's collectors.py (tests/data_collection/test_skip_existing.py,
a DIFFERENT module -- always skips existing tickers unconditionally),
collect_prices_yf normally re-fetches EVERY ticker's entire span every run on
purpose (a dividend paid after collection needs its whole history's
adj_close revisited). skip_existing is opt-in, off by default, a narrow
escape hatch for resuming an interrupted first-time backfill -- added after
watching a real restarted run spend over an hour re-fetching ~2,090 tickers
already on disk from an earlier interrupted attempt, with zero new tickers
added in that time.

Usage: python tests/data_collection/test_prices_yf_skip_existing.py
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_collection.yf import prices as yfc


def test_skip_existing_true_skips_tickers_already_on_disk():
    with tempfile.TemporaryDirectory() as tmp:
        price_dir = Path(tmp)
        (price_dir / "AAPL.parquet").write_bytes(b"placeholder")  # already collected

        fetched = []
        with mock.patch.object(yfc, "checkpoint") as mock_cp, \
             mock.patch.object(yfc, "_prices_fetch_start", return_value="2020-01-01"), \
             mock.patch.object(yfc, "_bolsai_junction_date", return_value=None), \
             mock.patch.object(yfc, "_reconcile_yfinance_junction", side_effect=lambda t, p, df, j: df), \
             mock.patch.object(yfc, "_fetch_and_shape_prices",
                                side_effect=lambda t, *a, **k: fetched.append(t) or
                                pd.DataFrame({"trade_date": pd.to_datetime(["2026-01-01"]), "close": [1.0]})), \
             mock.patch.object(yfc, "_merge_save", side_effect=lambda df, *a, **k: df), \
             mock.patch.object(yfc, "sleep"):
            mock_cp.load.return_value = {}
            yfc.collect_prices_yf(["AAPL", "MSFT"], mode="test", price_dir=price_dir,
                                   suffix="", floor="1900-01-01", skip_existing=True)

        assert fetched == ["MSFT"], f"AAPL already on disk must be skipped entirely, got fetched={fetched}"
    print("OK: skip_existing=True skips a ticker whose file already exists, fetches new ones")


def test_skip_existing_false_still_refetches_everything_by_default():
    # The dangerous-by-default direction: skip_existing must default to False, so a normal
    # run (e.g. --mode update) still re-fetches every ticker's full span as designed.
    with tempfile.TemporaryDirectory() as tmp:
        price_dir = Path(tmp)
        (price_dir / "AAPL.parquet").write_bytes(b"placeholder")

        fetched = []
        with mock.patch.object(yfc, "checkpoint") as mock_cp, \
             mock.patch.object(yfc, "_prices_fetch_start", return_value="2020-01-01"), \
             mock.patch.object(yfc, "_bolsai_junction_date", return_value=None), \
             mock.patch.object(yfc, "_reconcile_yfinance_junction", side_effect=lambda t, p, df, j: df), \
             mock.patch.object(yfc, "_fetch_and_shape_prices",
                                side_effect=lambda t, *a, **k: fetched.append(t) or
                                pd.DataFrame({"trade_date": pd.to_datetime(["2026-01-01"]), "close": [1.0]})), \
             mock.patch.object(yfc, "_merge_save", side_effect=lambda df, *a, **k: df), \
             mock.patch.object(yfc, "sleep"):
            mock_cp.load.return_value = {}
            yfc.collect_prices_yf(["AAPL", "MSFT"], mode="test", price_dir=price_dir,
                                   suffix="", floor="1900-01-01")  # skip_existing defaults False

        assert fetched == ["AAPL", "MSFT"], f"default must still re-fetch AAPL despite existing file, got {fetched}"
    print("OK: skip_existing defaults to False, preserving the always-refetch design for --mode update")


if __name__ == "__main__":
    test_skip_existing_true_skips_tickers_already_on_disk()
    test_skip_existing_false_still_refetches_everything_by_default()
