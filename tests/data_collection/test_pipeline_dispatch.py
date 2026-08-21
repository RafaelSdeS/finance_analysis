#!/usr/bin/env python3
"""
Dispatch routing in pipeline.py's _collect(): per-data-type source switch
(BolsAI vs yfinance, config.DATA_SOURCE) plus the YFINANCE_ONLY_TICKERS
override for benchmark ETFs (e.g. BOVA11, not on BolsAI at all).

No test previously touched this at all -- a bug here would silently route
real API calls to the wrong collector (or the wrong tickers) rather than
failing loudly, and the mistake would only surface downstream as a
confusing data-shape error.

Run from project root: python tests/data_collection/test_pipeline_dispatch.py
or: pytest tests/data_collection/test_pipeline_dispatch.py -v
"""

import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection import config
from src.data_collection.br import pipeline


def test_dispatches_to_bolsai_when_configured(monkeypatch) -> None:
    monkeypatch.setitem(config.DATA_SOURCE, "prices", "bolsai")
    monkeypatch.setattr(config, "YFINANCE_ONLY_TICKERS", set())
    with mock.patch.object(pipeline.collectors, "collect_prices") as bolsai_fn, \
         mock.patch.object(pipeline.yf_prices, "collect_prices_yf") as yf_fn:
        pipeline._collect("prices", ["PETR4", "VALE3"], "prototype")

    bolsai_fn.assert_called_once_with(["PETR4", "VALE3"], "prototype")
    yf_fn.assert_not_called()


def test_dispatches_to_yfinance_when_configured(monkeypatch) -> None:
    monkeypatch.setitem(config.DATA_SOURCE, "fundamentals", "yfinance")
    monkeypatch.setattr(config, "YFINANCE_ONLY_TICKERS", set())
    with mock.patch.object(pipeline.collectors, "collect_fundamentals") as bolsai_fn, \
         mock.patch.object(pipeline.yf_fundamentals, "collect_fundamentals_yf") as yf_fn:
        pipeline._collect("fundamentals", ["PETR4"], "update")

    yf_fn.assert_called_once_with(["PETR4"], "update")
    bolsai_fn.assert_not_called()


def test_yfinance_only_tickers_bypass_data_source_and_split_from_batch(monkeypatch) -> None:
    """BOVA11 (a benchmark ETF, not on BolsAI) always goes through yfinance
    regardless of config.DATA_SOURCE, split out from the rest of the batch --
    which still follows the global source setting."""
    monkeypatch.setitem(config.DATA_SOURCE, "prices", "bolsai")
    monkeypatch.setattr(config, "YFINANCE_ONLY_TICKERS", {"BOVA11"})
    with mock.patch.object(pipeline.collectors, "collect_prices") as bolsai_fn, \
         mock.patch.object(pipeline.yf_prices, "collect_prices_yf") as yf_fn:
        pipeline._collect("prices", ["PETR4", "BOVA11"], "full_scale")

    bolsai_fn.assert_called_once_with(["PETR4"], "full_scale")
    yf_fn.assert_called_once_with(["BOVA11"], "full_scale")


def test_defaults_to_bolsai_when_data_type_unconfigured(monkeypatch) -> None:
    """During `--mode update`, a data type missing from config.DATA_SOURCE
    entirely falls back to bolsai (the dict.get default inside _collect),
    not a KeyError. Uses mode="update" specifically so this exercises the
    dict.get fallback rather than the mode-forces-bolsai path below."""
    monkeypatch.setattr(config, "DATA_SOURCE", {})
    monkeypatch.setattr(config, "YFINANCE_ONLY_TICKERS", set())
    with mock.patch.object(pipeline.collectors, "collect_dividends") as bolsai_fn, \
         mock.patch.object(pipeline.yf_dividends, "collect_dividends_yf") as yf_fn:
        pipeline._collect("dividends", ["PETR4"], "update")

    bolsai_fn.assert_called_once_with(["PETR4"], "update")
    yf_fn.assert_not_called()


def test_dispatches_to_cvm_when_configured(monkeypatch) -> None:
    monkeypatch.setitem(config.DATA_SOURCE, "fundamentals", "cvm")
    monkeypatch.setattr(config, "YFINANCE_ONLY_TICKERS", set())
    with mock.patch.object(pipeline.cvm_ratios, "collect_fundamentals_cvm") as cvm_fn, \
         mock.patch.object(pipeline.collectors, "collect_fundamentals") as bolsai_fn, \
         mock.patch.object(pipeline.yf_fundamentals, "collect_fundamentals_yf") as yf_fn:
        pipeline._collect("fundamentals", ["PETR4"], "update")

    cvm_fn.assert_called_once_with(["PETR4"], "update")
    bolsai_fn.assert_not_called()
    yf_fn.assert_not_called()


def test_default_fundamentals_source_is_never_the_broken_yfinance_path() -> None:
    """Regression guard for BUG-1 (BOLSAI_EXIT_PLAN.md): yfinance's BR fundamentals
    are wrong in level (point-in-time balance-sheet items falling ~5x quarter over
    quarter), not just thin. "cvm" is a free superset of BolsAI's own depth -- the
    repo's default must never regress back to "yfinance" for this data type."""
    assert config.DATA_SOURCE["fundamentals"] != "yfinance"


def test_full_scale_honors_data_source_same_as_update(monkeypatch) -> None:
    """All modes (full_scale/prototype/update alike) now dispatch purely off
    config.DATA_SOURCE -- the free sources (yfinance prices/dividends, CVM
    fundamentals) match or exceed BolsAI's own depth (BOLSAI_EXIT_PLAN.md Task 5),
    so full_scale no longer needs a mode-based override forcing BolsAI."""
    monkeypatch.setitem(config.DATA_SOURCE, "fundamentals", "cvm")
    monkeypatch.setattr(config, "YFINANCE_ONLY_TICKERS", set())
    with mock.patch.object(pipeline.cvm_ratios, "collect_fundamentals_cvm") as cvm_fn, \
         mock.patch.object(pipeline.collectors, "collect_fundamentals") as bolsai_fn:
        pipeline._collect("fundamentals", ["PETR4"], "full_scale")

    cvm_fn.assert_called_once_with(["PETR4"], "full_scale")
    bolsai_fn.assert_not_called()


def test_recover_stale_company_info_tickers_picks_up_on_disk_orphans(tmp_path, monkeypatch) -> None:
    """A ticker with an existing raw price file but missing/non-ATIVO in
    company_info (BolsAI's own /companies/ registry structurally omits some
    real, still-trading distressed names -- confirmed 2026-08-16 on
    AMER3/Americanas and 14 others) must still be recovered for the free
    yfinance --mode update refresh. A ticker with NO existing file must NOT
    be recovered here -- that's collect_delisted.py's job."""
    monkeypatch.setattr(config, "PRICES_DIR", tmp_path)
    (tmp_path / "AMER3.parquet").touch()
    (tmp_path / "PETR4.parquet").touch()  # already ATIVO -- in prices_tickers already

    recovered = pipeline._recover_stale_company_info_tickers(
        requested=["AMER3", "PETR4", "NEVERCOLLECTED3"],
        prices_tickers={"PETR4"},
    )

    assert recovered == {"AMER3"}


def test_recover_stale_company_info_tickers_empty_when_nothing_orphaned(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "PRICES_DIR", tmp_path)
    (tmp_path / "PETR4.parquet").touch()

    recovered = pipeline._recover_stale_company_info_tickers(
        requested=["PETR4"], prices_tickers={"PETR4"},
    )

    assert recovered == set()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
