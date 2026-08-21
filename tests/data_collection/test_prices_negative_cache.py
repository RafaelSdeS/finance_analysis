"""
test_prices_negative_cache.py
==============================
Self-check for collect_prices_yf's negative cache on dead/delisted tickers
(no network; mocks every internal that touches yfinance/disk).

Before this (2026-08-13), a genuinely delisted ticker got re-probed with a
full network request -- and paid _retry's empty-result backoff sleep -- on
EVERY run, forever (measured: 128 "possibly delisted" + 85 empty-result
retries in one US prices log). `empty_runs` (persisted per-ticker in the
checkpoint) tracks consecutive empty results; past EMPTY_RUNS_SKIP_THRESHOLD
the ticker is skipped outright except on every EMPTY_RUNS_REPROBE_INTERVAL-th
run, which still probes for real so a genuine re-listing isn't cached as
dead forever.

Usage: python tests/data_collection/test_prices_negative_cache.py
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_collection.yf import prices as yfc


def _fake_checkpoint_store():
    store: dict = {}

    def fake_load(name, mode):
        return store.get((name, mode), {})

    def fake_save(name, mode, data):
        store[(name, mode)] = dict(data)

    return store, fake_load, fake_save


def test_negative_cache_skips_after_threshold_and_reprobes():
    with tempfile.TemporaryDirectory() as tmp:
        price_dir = Path(tmp)
        store, fake_load, fake_save = _fake_checkpoint_store()
        call_count = {"n": 0}

        def fake_fetch(ticker, *a, **k):
            call_count["n"] += 1
            return None  # always "no coverage"

        with mock.patch.object(yfc, "checkpoint") as mock_cp, \
             mock.patch.object(yfc, "_prices_fetch_start", return_value="1900-01-01"), \
             mock.patch.object(yfc, "_bolsai_junction_date", return_value=None), \
             mock.patch.object(yfc, "_fetch_and_shape_prices", side_effect=fake_fetch), \
             mock.patch.object(yfc, "sleep"):
            mock_cp.load.side_effect = fake_load
            mock_cp.save.side_effect = fake_save

            # 11 successive "runs" (each a fresh collect_prices_yf call, like
            # separate --mode update invocations sharing one on-disk checkpoint).
            for _ in range(11):
                yfc.collect_prices_yf(["DEAD"], mode="test", price_dir=price_dir,
                                       suffix="", floor="1900-01-01")

        cp = store[("yf_prices", "test")]
        assert cp["DEAD"]["empty_runs"] == 11, f"expected empty_runs=11 after 11 dark runs, got {cp['DEAD']}"
        # Real attempts: the first 3 (ramp-up to the skip threshold) + run #11
        # (empty_runs read as 10 at its start, a multiple of EMPTY_RUNS_REPROBE_INTERVAL).
        assert call_count["n"] == 4, \
            f"expected exactly 4 real fetch attempts (3 ramp-up + 1 reprobe) across 11 runs, got {call_count['n']}"
    print("OK: negative cache skips a dead ticker past the threshold, still reprobes periodically")


def test_empty_runs_resets_when_ticker_saves_real_rows():
    with tempfile.TemporaryDirectory() as tmp:
        price_dir = Path(tmp)
        store, fake_load, fake_save = _fake_checkpoint_store()
        store[("yf_prices", "test")] = {"REVIVED": {"empty_runs": 2}}  # below skip threshold

        with mock.patch.object(yfc, "checkpoint") as mock_cp, \
             mock.patch.object(yfc, "_prices_fetch_start", return_value="2020-01-01"), \
             mock.patch.object(yfc, "_bolsai_junction_date", return_value=None), \
             mock.patch.object(yfc, "_reconcile_yfinance_junction", side_effect=lambda t, p, df, j: df), \
             mock.patch.object(yfc, "_fetch_and_shape_prices",
                                return_value=pd.DataFrame({"trade_date": pd.to_datetime(["2026-01-01"]),
                                                            "close": [1.0]})), \
             mock.patch.object(yfc, "_merge_save", side_effect=lambda df, *a, **k: df), \
             mock.patch.object(yfc, "sleep"):
            mock_cp.load.side_effect = fake_load
            mock_cp.save.side_effect = fake_save
            yfc.collect_prices_yf(["REVIVED"], mode="test", price_dir=price_dir,
                                   suffix="", floor="1900-01-01")

        cp = store[("yf_prices", "test")]
        assert "empty_runs" not in cp["REVIVED"], \
            f"a real successful fetch must clear empty_runs, not carry it forward, got {cp['REVIVED']}"
    print("OK: empty_runs is cleared once a ticker actually saves new rows again")


def test_empty_runs_resets_when_coverage_confirmed_but_nothing_new():
    with tempfile.TemporaryDirectory() as tmp:
        price_dir = Path(tmp)
        store, fake_load, fake_save = _fake_checkpoint_store()
        store[("yf_prices", "test")] = {"OK": {"empty_runs": 2, "last_date": "2026-01-01"}}

        with mock.patch.object(yfc, "checkpoint") as mock_cp, \
             mock.patch.object(yfc, "_prices_fetch_start", return_value="2026-01-02"), \
             mock.patch.object(yfc, "_bolsai_junction_date", return_value=None), \
             mock.patch.object(yfc, "_fetch_and_shape_prices",
                                return_value=pd.DataFrame({"trade_date": pd.to_datetime(["2026-01-02"]),
                                                            "close": [1.0]})), \
             mock.patch.object(yfc, "_reconcile_yfinance_junction",
                                side_effect=lambda t, p, df, j: df.iloc[0:0]), \
             mock.patch.object(yfc, "sleep"):
            mock_cp.load.side_effect = fake_load
            mock_cp.save.side_effect = fake_save
            yfc.collect_prices_yf(["OK"], mode="test", price_dir=price_dir,
                                   suffix="", floor="1900-01-01")

        cp = store[("yf_prices", "test")]
        assert cp["OK"].get("empty_runs", 0) == 0, \
            f"real yfinance coverage (even with nothing new past the junction) must reset empty_runs, got {cp['OK']}"
    print("OK: empty_runs resets once real coverage is confirmed, even with nothing new to save")


if __name__ == "__main__":
    test_negative_cache_skips_after_threshold_and_reprobes()
    test_empty_runs_resets_when_ticker_saves_real_rows()
    test_empty_runs_resets_when_coverage_confirmed_but_nothing_new()
