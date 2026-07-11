"""
test_macro_bare_object.py
=========================
Verifies collect_macro() handles BCB returning a bare JSON object (instead of
a list) for a narrow date range with exactly one data point. Without the
isinstance(d, dict) guard, `rows += d` silently corrupts rows with the dict's
keys and crashes downstream with KeyError('data').

Usage:
    python tests/data_collection/test_macro_bare_object.py
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection import collectors, config


def test_bare_object_response():
    with tempfile.TemporaryDirectory() as tmp:
        macro_dir = Path(tmp) / "macro"
        macro_dir.mkdir()

        with mock.patch.object(config, "MACRO_DIR", macro_dir), \
             mock.patch.object(config, "BCB_SERIES", {"cdi": 12}), \
             mock.patch.object(collectors.client, "get_json",
                               return_value={"data": "11/07/2026", "valor": "0.0538"}), \
             mock.patch.object(collectors.client, "make_client", return_value=mock.MagicMock()), \
             mock.patch.object(collectors.checkpoint, "load", return_value={}), \
             mock.patch.object(collectors.checkpoint, "save"):
            collectors.collect_macro(mode="update")

        df = __import__("pandas").read_parquet(macro_dir / "cdi.parquet")
        assert len(df) == 1, f"expected 1 row, got {len(df)}"
        assert df.iloc[0]["cdi"] == 0.0538

    print("OK: bare-object BCB response is normalized into a single row, not corrupted")


if __name__ == "__main__":
    test_bare_object_response()
