"""
test_fred_collectors.py
========================
Self-check for fred_collectors.py: FRED's CSV uses "." for missing
observations (not blank) -- unhandled, pd.to_numeric would raise instead of
coercing to NaN, and a stray "." row would silently become a bad float if
coercion were skipped. Also verifies the collected series lands as
(reference_date, <name>) with the missing row dropped.

Usage:
    python tests/data_collection/test_fred_collectors.py
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection import config
from src.data_collection.us import fred_collectors


FAKE_CSV = "observation_date,FEDFUNDS\n1954-07-01,0.80\n1954-08-01,.\n1954-09-01,1.07\n"


def test_missing_observation_and_rename():
    with tempfile.TemporaryDirectory() as tmp:
        macro_dir = Path(tmp) / "macro_us"

        with mock.patch.object(config, "US_MACRO_DIR", macro_dir), \
             mock.patch.object(config, "FRED_SERIES", {"fed_funds": "FEDFUNDS"}), \
             mock.patch.object(fred_collectors.client, "get_text", return_value=FAKE_CSV), \
             mock.patch.object(fred_collectors.client, "make_client", return_value=mock.MagicMock()), \
             mock.patch.object(fred_collectors.checkpoint, "load", return_value={}), \
             mock.patch.object(fred_collectors.checkpoint, "save"):
            fred_collectors.collect_macro_us(mode="test")

        import pandas as pd
        df = pd.read_parquet(macro_dir / "fed_funds.parquet")
        assert list(df.columns) == ["reference_date", "fed_funds"]
        assert len(df) == 2, f"expected the '.' row dropped, got {len(df)} rows"
        assert set(df["fed_funds"]) == {0.80, 1.07}

    print("OK: FRED '.' missing-observation marker dropped, not coerced into a bad float")


if __name__ == "__main__":
    test_missing_observation_and_rename()
