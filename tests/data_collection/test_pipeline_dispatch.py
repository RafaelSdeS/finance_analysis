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

from src.data_collection import config, pipeline


def test_dispatches_to_bolsai_when_configured(monkeypatch) -> None:
    monkeypatch.setitem(config.DATA_SOURCE, "prices", "bolsai")
    monkeypatch.setattr(config, "YFINANCE_ONLY_TICKERS", set())
    with mock.patch.object(pipeline.collectors, "collect_prices") as bolsai_fn, \
         mock.patch.object(pipeline.yf_collectors, "collect_prices_yf") as yf_fn:
        pipeline._collect("prices", ["PETR4", "VALE3"], "prototype")

    bolsai_fn.assert_called_once_with(["PETR4", "VALE3"], "prototype")
    yf_fn.assert_not_called()


def test_dispatches_to_yfinance_when_configured(monkeypatch) -> None:
    monkeypatch.setitem(config.DATA_SOURCE, "fundamentals", "yfinance")
    monkeypatch.setattr(config, "YFINANCE_ONLY_TICKERS", set())
    with mock.patch.object(pipeline.collectors, "collect_fundamentals") as bolsai_fn, \
         mock.patch.object(pipeline.yf_collectors, "collect_fundamentals_yf") as yf_fn:
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
         mock.patch.object(pipeline.yf_collectors, "collect_prices_yf") as yf_fn:
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
         mock.patch.object(pipeline.yf_collectors, "collect_dividends_yf") as yf_fn:
        pipeline._collect("dividends", ["PETR4"], "update")

    bolsai_fn.assert_called_once_with(["PETR4"], "update")
    yf_fn.assert_not_called()


def test_non_update_modes_force_bolsai_regardless_of_data_source(monkeypatch) -> None:
    """Regression test: full_scale/prototype are the one-time historical
    backfill and must always use BolsAI's deep history, even if
    config.DATA_SOURCE says yfinance (which governs `--mode update` only).
    This bug previously routed full_scale silently through yfinance's
    shallow ~5-quarter fundamentals depth instead of BolsAI's ~80-quarter
    backfill (caught via BPAC11 getting incomplete fundamentals)."""
    monkeypatch.setitem(config.DATA_SOURCE, "fundamentals", "yfinance")
    monkeypatch.setattr(config, "YFINANCE_ONLY_TICKERS", set())
    with mock.patch.object(pipeline.collectors, "collect_fundamentals") as bolsai_fn, \
         mock.patch.object(pipeline.yf_collectors, "collect_fundamentals_yf") as yf_fn:
        pipeline._collect("fundamentals", ["PETR4"], "full_scale")

    bolsai_fn.assert_called_once_with(["PETR4"], "full_scale")
    yf_fn.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
