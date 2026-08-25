#!/usr/bin/env python3
"""
Pins DEFECT-1a and DEFECT-1b (docs/BR_DATA_RECONSTRUCTION_PLAN.md §2): two
ways pipeline.py's ticker universe silently shrinks between "requested" and
"actually collected", found auditing the 2026-08-23 recollection that
regressed BR prices from 1,328 to 383 files.

DEFECT-1a: run()'s prices stage only ever gets `active` (status == ATIVO in
company_info). A ticker already collected once (a real, confirmed equity)
but now CANCELADA/SUSPENSO -- or never resolved by CVM at all -- is dropped
from prices every run, not just once. _recover_stale_company_info_tickers
already exists to add such tickers back, but only when mode == "update",
so full_scale (the mode that ran 2026-08-23) gets none of it.

DEFECT-1b: br/collectors.get_all_tickers()'s regex `^[A-Z0-9]{4}[3-8]$`
drops every suffix-11 ticker (FIIs/ETFs use it too), papered over by a
hand-maintained KNOWN_UNIT_TICKERS = {"BOVA11", "BPAC11"} allowlist that
was two entries out of date -- 13 real ATIVO operating-company units
(TAEE11, SANB11, KLBN11, ...) never made it into the requested list at all.

Run from project root: python tests/data_collection/test_universe_derivation.py
or: pytest tests/data_collection/test_universe_derivation.py -v
"""

import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection import config
from src.data_collection.br import collectors, pipeline


def _write_company_info(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["ticker", "status"]).to_parquet(path)


def test_full_scale_prices_include_already_collected_non_active_tickers(tmp_path, monkeypatch):
    """DEFECT-1a. AMER3 has a price file on disk (collected once, a real
    equity -- CLAUDE.md's documented BolsAI-registry gap) but company_info
    now reads CANCELADA. A full_scale run must still refresh its price
    series, the same way --mode update already does via
    _recover_stale_company_info_tickers -- that recovery must not be
    restricted to update mode."""
    monkeypatch.setattr(config, "COMPANY_INFO_PATH", tmp_path / "company_info.parquet")
    monkeypatch.setattr(config, "PRICES_DIR", tmp_path / "prices")
    monkeypatch.setattr(config, "BENCHMARK_TICKERS", [])
    (tmp_path / "prices").mkdir()
    (tmp_path / "prices" / "AMER3.parquet").touch()
    _write_company_info(config.COMPANY_INFO_PATH, [
        ("PETR4", "ATIVO"),
        ("AMER3", "CANCELADA"),
    ])

    with mock.patch.object(pipeline.macro, "collect_macro"), \
         mock.patch.object(pipeline.cvm_company_info, "synthesize_company_info"), \
         mock.patch.object(pipeline.cvm_sectors, "build_sectors"), \
         mock.patch.object(pipeline.yf_dividends, "collect_splits_yf"), \
         mock.patch.object(pipeline, "_collect") as collect_fn:
        pipeline.run("full_scale", ["PETR4", "AMER3"])

    prices_call = next(c for c in collect_fn.call_args_list if c.args[0] == "prices")
    assert set(prices_call.args[1]) == {"PETR4", "AMER3"}


def test_get_all_tickers_keeps_crosswalk_confirmed_units(tmp_path, monkeypatch):
    """DEFECT-1b. A suffix-11 ticker is a real operating-company unit (not a
    FII/ETF) when the FCA crosswalk resolves a CNPJ for it -- the same test
    collect_delisted.py's candidate_tickers() already uses (reading
    config.CVM_DIR / "fca_crosswalk.parquet" directly). get_all_tickers()
    must use that same file, not a hand-maintained KNOWN_UNIT_TICKERS set
    that silently drifts out of date (TAEE11/SANB11/KLBN11/... were missing)."""
    raw = ["PETR4", "TAEE11", "HGLG11", "BOVA11"]  # HGLG11 is a real FII, must stay excluded
    monkeypatch.setattr(collectors, "get_all_tickers_raw", lambda: raw)
    monkeypatch.setattr(config, "CVM_DIR", tmp_path)
    pd.DataFrame({"ticker": ["TAEE11", "PETR4"], "cnpj": ["1", "2"]}) \
        .to_parquet(tmp_path / "fca_crosswalk.parquet")

    result = set(collectors.get_all_tickers())

    assert "TAEE11" in result
    assert "HGLG11" not in result


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
