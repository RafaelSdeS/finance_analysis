"""
test_skip_existing.py
======================
Verifies collect_prices / collect_fundamentals / collect_dividends skip the
API call entirely when the ticker's parquet is already on disk AND COMPLETE
(see CLAUDE.md: BolsAI backfill is one-time; --mode update handles freshness),
and conversely that an on-disk file MISSING a required column is re-collected
rather than skipped forever (storage.is_complete).

Usage:
    python tests/data_collection/test_skip_existing.py
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection import config, validate
from src.data_collection.br import collectors
from src.data_collection.yf_collectors import FUND_FULL_COLS


def _write(path, cols, ticker, drop=None):
    """One row carrying every column in `cols` (minus `drop`) -- a COMPLETE
    on-disk file as far as storage.is_complete is concerned."""
    cols = [c for c in cols if c != drop]
    pd.DataFrame({c: [ticker if c == "ticker" else 1] for c in cols}).to_parquet(path)


def test_skip_existing():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        prices_dir, fund_dir, div_dir = tmp / "prices", tmp / "fundamentals", tmp / "dividends"
        for d in (prices_dir, fund_dir, div_dir):
            d.mkdir()

        ticker = "FAKE3"
        _write(prices_dir / f"{ticker}.parquet", validate.PRICE_COLS, ticker)
        _write(fund_dir / f"{ticker}.parquet", FUND_FULL_COLS, ticker)
        _write(div_dir / f"{ticker}.parquet", validate.DIVIDEND_COLS, ticker)

        def _boom(*a, **kw):
            raise AssertionError("API should not be called for an already-collected ticker")

        with mock.patch.object(config, "PRICES_DIR", prices_dir), \
             mock.patch.object(config, "FUND_DIR", fund_dir), \
             mock.patch.object(config, "DIVIDENDS_DIR", div_dir), \
             mock.patch.object(collectors.client, "get_json", side_effect=_boom), \
             mock.patch.object(collectors.client, "make_client", return_value=mock.MagicMock()), \
             mock.patch.object(collectors.checkpoint, "load", return_value={}), \
             mock.patch.object(collectors.checkpoint, "save"):
            collectors.collect_prices([ticker], mode="prototype")
            collectors.collect_fundamentals([ticker], mode="prototype")
            collectors.collect_dividends([ticker], mode="prototype")

    print("OK: prices/fundamentals/dividends all skipped an already-collected ticker")


def test_incomplete_file_is_recollected_not_skipped():
    """The `roic` bug (2026-08-15): 247 of 548 data/raw/br/fundamentals files were
    missing `roic` while 296 had it, and a `path.exists()` skip meant re-running
    full_scale could never repair them -- silently, since validate.FUND_COLS only
    requires a 9-column subset that doesn't include `roic`."""
    with tempfile.TemporaryDirectory() as tmp:
        fund_dir = Path(tmp) / "fundamentals"
        fund_dir.mkdir()

        ticker = "DRIFT3"
        _write(fund_dir / f"{ticker}.parquet", FUND_FULL_COLS, ticker, drop="roic")

        called = []

        def _fake_get_json(c, path, params=None):
            called.append(path)
            return {"history": []}

        with mock.patch.object(config, "FUND_DIR", fund_dir), \
             mock.patch.object(collectors.client, "get_json", side_effect=_fake_get_json), \
             mock.patch.object(collectors.client, "make_client", return_value=mock.MagicMock()), \
             mock.patch.object(collectors.checkpoint, "load", return_value={}), \
             mock.patch.object(collectors.checkpoint, "save"):
            collectors.collect_fundamentals([ticker], mode="full_scale")

        assert called, ("a file missing a required column (roic) must be re-collected, "
                        "not skipped as 'already collected'")

    print("OK: an on-disk file missing a required column is re-collected, not skipped")


def test_skip_list_reprobes_instead_of_excluding_forever():
    """A skip-listed ticker must get another real attempt every Nth run -- with
    MAX_RETRIES previously at 1, one transient BolsAI 503 blacklisted a live
    ticker permanently, and `_skip` cleared only by hand-editing the JSON."""
    from src.data_collection import checkpoint

    n = checkpoint.SKIP_REPROBE_INTERVAL

    # Legacy plain-list format (what's on disk today): count unknown -> due now.
    legacy = checkpoint.load_skip({"_skip": ["DEAD3"]})
    assert not checkpoint.should_skip(legacy, "DEAD3"), "legacy entry should be due for a re-probe"

    # Fresh failure -> skipped on the following runs, but not forever.
    skip = {"DEAD3": 1}
    assert checkpoint.should_skip(skip, "DEAD3")
    assert not checkpoint.should_skip({"DEAD3": n}, "DEAD3"), "must re-probe on the Nth run"
    assert checkpoint.should_skip({"DEAD3": n + 1}, "DEAD3"), "and go quiet again after"

    # A ticker that was never skip-listed is never skipped.
    assert not checkpoint.should_skip({}, "GOOD3")

    # A successful re-probe must reset the count, or it'd be skipped again next run.
    cp, skip = {}, {"DEAD3": n}
    with mock.patch.object(checkpoint, "save"):
        checkpoint.clear_skip("prices", "full_scale", cp, skip, "DEAD3")
    assert "DEAD3" not in skip
    assert not checkpoint.should_skip(skip, "DEAD3")

    print("OK: skip list re-probes every Nth run and resets on success")


def test_missing_file_ignores_stale_checkpoint():
    """A checkpoint entry must not be trusted once its parquet is gone -- otherwise
    collect_prices would fetch only a narrow incremental window from the stale
    last_date instead of rebuilding full history, silently truncating the ticker
    (e.g. after a corrupted file is manually reverted, see backfill_known_gaps.py,
    but its checkpoint entry survives)."""
    with tempfile.TemporaryDirectory() as tmp:
        prices_dir = Path(tmp) / "prices"
        prices_dir.mkdir()

        ticker = "GONE3"
        fetch_starts = []

        def _fake_get_json(c, path, params=None):
            fetch_starts.append(params["start"])
            return {"prices": []}

        stale_cp = {ticker: {"last_date": "2025-06-01", "rows": 100}}

        with mock.patch.object(config, "PRICES_DIR", prices_dir), \
             mock.patch.object(collectors.client, "get_json", side_effect=_fake_get_json), \
             mock.patch.object(collectors.client, "make_client", return_value=mock.MagicMock()), \
             mock.patch.object(collectors.checkpoint, "load", return_value=stale_cp), \
             mock.patch.object(collectors.checkpoint, "save"):
            collectors.collect_prices([ticker], mode="full_scale")

        assert fetch_starts, "expected at least one fetch window"
        assert fetch_starts[0] == config.START_DATE, (
            f"a missing file must trigger a full backfill from START_DATE, not the "
            f"stale checkpoint's last_date; got start={fetch_starts[0]}")

    print("OK: a stale checkpoint entry for a missing file doesn't truncate the refetch")


if __name__ == "__main__":
    test_skip_existing()
    test_incomplete_file_is_recollected_not_skipped()
    test_skip_list_reprobes_instead_of_excluding_forever()
    test_missing_file_ignores_stale_checkpoint()
