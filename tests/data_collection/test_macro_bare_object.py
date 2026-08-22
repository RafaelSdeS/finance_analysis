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

from src.data_collection import config
from src.data_collection.br import macro


def test_bare_object_response():
    with tempfile.TemporaryDirectory() as tmp:
        macro_dir = Path(tmp) / "macro"
        macro_dir.mkdir()

        with mock.patch.object(config, "MACRO_DIR", macro_dir), \
             mock.patch.object(config, "BCB_SERIES", {"cdi": 12}), \
             mock.patch.object(macro.client, "get_json",
                               return_value={"data": "11/07/2026", "valor": "0.0538"}), \
             mock.patch.object(macro.client, "make_client", return_value=mock.MagicMock()), \
             mock.patch.object(macro.checkpoint, "load", return_value={}), \
             mock.patch.object(macro.checkpoint, "save"):
            macro.collect_macro(mode="update")

        df = __import__("pandas").read_parquet(macro_dir / "cdi.parquet")
        assert len(df) == 1, f"expected 1 row, got {len(df)}"
        assert df.iloc[0]["cdi"] == 0.0538

    print("OK: bare-object BCB response is normalized into a single row, not corrupted")


def test_series_ids_match_documented_bcb_codes():
    """CLAUDE.md's documented gotcha: ipca is BCB SGS series 433 (monthly), not
    432 (the annual meta target) -- a real mixup this has bitten before. Pure-code
    assert on the config dict itself, no network/data needed, so a series-ID swap
    is caught immediately rather than only downstream via test_br_data_quality.py's
    ipca row-count heuristic, which needs data collected first to notice at all."""
    assert config.BCB_SERIES == {"selic": 11, "cdi": 12, "ipca": 433}, config.BCB_SERIES
    print("OK: BCB_SERIES matches the documented selic=11/cdi=12/ipca=433 codes")


if __name__ == "__main__":
    test_bare_object_response()
    test_series_ids_match_documented_bcb_codes()
