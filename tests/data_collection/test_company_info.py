"""
test_company_info.py
=====================
Regression guard for cvm/company_info.py's synthesize_company_info() (0827f8b,
"feat: add single-quarter fundamentals + auto-refresh CVM crosswalk"): it must
call build_crosswalk() before reading CROSSWALK_PATH, so a ticker that just
IPO'd and filed its first FCA this year is discoverable without a manual
crosswalk rebuild or BolsAI's /stocks/ registry. Before this change, a brand-
new ticker present in CVM's CAD registry but absent from a stale on-disk
crosswalk would silently never appear in company_info.parquet.

Run from project root:
    python tests/data_collection/test_company_info.py
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection.cvm import company_info  # noqa: E402


def _delist_df():
    return pd.DataFrame({
        "ticker": ["OLDX3", "NEWX3"],
        "cnpj": ["11111111000111", "99999999000199"],
        "delist_date": [pd.NaT, pd.NaT],
        "motivo_cancel": [None, None],
        "sit": ["ATIVO", "ATIVO"],
    })


def _seed_existing(tmp) -> tuple[Path, Path]:
    crosswalk_path = Path(tmp) / "fca_crosswalk.parquet"
    pd.DataFrame({
        "ticker": ["OLDX3", "NEWX3"],
        "cnpj": ["11111111000111", "99999999000199"],
        "corporate_name": ["OLD CO SA", "NEW CO SA"],
        "cvm_code": ["1", "99999"],
    }).to_parquet(crosswalk_path, index=False)

    company_info_path = Path(tmp) / "company_info.parquet"
    pd.DataFrame({
        "ticker": ["OLDX3"], "ticker_primary": ["OLDX3"], "corporate_name": ["OLD CO SA"],
        "trade_name": [None], "cvm_code": ["1"], "cnpj": ["11111111000111"],
        "sector": [None], "status": ["ATIVO"],
    }).to_parquet(company_info_path, index=False)
    return crosswalk_path, company_info_path


def test_synthesize_calls_build_crosswalk():
    """build_crosswalk() must run before the crosswalk is read -- a stale
    on-disk crosswalk should never silently persist across a refresh."""
    with tempfile.TemporaryDirectory() as tmp:
        crosswalk_path, company_info_path = _seed_existing(tmp)
        build_crosswalk_mock = mock.Mock()

        with mock.patch.object(company_info, "build_crosswalk", build_crosswalk_mock), \
             mock.patch.object(company_info, "CROSSWALK_PATH", crosswalk_path), \
             mock.patch.object(company_info.config, "COMPANY_INFO_PATH", company_info_path), \
             mock.patch.object(company_info, "build_delist_events", return_value=_delist_df().iloc[:1]), \
             mock.patch.object(company_info, "sector_by_ticker", return_value=pd.Series(dtype=object)):
            company_info.synthesize_company_info()

        build_crosswalk_mock.assert_called_once()


def test_new_ipo_ticker_discovered_via_crosswalk():
    """A ticker present in CVM's CAD registry + crosswalk but not yet in
    company_info.parquet must be appended with its crosswalk-resolved
    cnpj/cvm_code -- simulates a same-run IPO -> first-FCA-filing -> pickup."""
    with tempfile.TemporaryDirectory() as tmp:
        crosswalk_path, company_info_path = _seed_existing(tmp)

        with mock.patch.object(company_info, "build_crosswalk", mock.Mock()), \
             mock.patch.object(company_info, "CROSSWALK_PATH", crosswalk_path), \
             mock.patch.object(company_info.config, "COMPANY_INFO_PATH", company_info_path), \
             mock.patch.object(company_info, "build_delist_events", return_value=_delist_df()), \
             mock.patch.object(company_info, "sector_by_ticker", return_value=pd.Series(dtype=object)):
            company_info.synthesize_company_info()

        out = pd.read_parquet(company_info_path)
        new_row = out[out["ticker"] == "NEWX3"]
        assert len(new_row) == 1, "NEWX3 must be appended as a new row"
        assert new_row.iloc[0]["cnpj"] == "99999999000199"
        assert new_row.iloc[0]["cvm_code"] == "99999"
        assert new_row.iloc[0]["status"] == "ATIVO"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
