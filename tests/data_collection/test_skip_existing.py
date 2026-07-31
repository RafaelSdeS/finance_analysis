"""
test_skip_existing.py
======================
Verifies collect_prices / collect_fundamentals / collect_dividends skip the
API call entirely when the ticker's parquet already exists on disk (see
CLAUDE.md: BolsAI backfill is one-time; --mode update handles freshness).

Usage:
    python tests/data_collection/test_skip_existing.py
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection import collectors, config


def test_skip_existing():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        prices_dir, fund_dir, div_dir = tmp / "prices", tmp / "fundamentals", tmp / "dividends"
        for d in (prices_dir, fund_dir, div_dir):
            d.mkdir()

        ticker = "FAKE3"
        pd.DataFrame({"ticker": [ticker]}).to_parquet(prices_dir / f"{ticker}.parquet")
        pd.DataFrame({"ticker": [ticker]}).to_parquet(fund_dir / f"{ticker}.parquet")
        pd.DataFrame({"ticker": [ticker]}).to_parquet(div_dir / f"{ticker}.parquet")

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
    test_missing_file_ignores_stale_checkpoint()
