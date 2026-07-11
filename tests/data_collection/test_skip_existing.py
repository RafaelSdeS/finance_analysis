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


if __name__ == "__main__":
    test_skip_existing()
