#!/usr/bin/env python3
"""
load_dividends()'s implausible-value_per_share sanity ceiling.

Run from project root: python tests/build_dataset/test_loaders.py
or: pytest tests/build_dataset/test_loaders.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_dataset import loaders  # noqa: E402


def test_load_dividends_drops_implausible_value_per_share(tmp_path, monkeypatch) -> None:
    """A real BRL per-share dividend is at most low tens even in extreme
    cases. Regression test for the PDGR3 raw-data bug (vendor unit/labeling
    error, value_per_share in the hundreds of millions -- inflated
    div_yield_12m to 154,600%, docs/TOP50_UNIVERSE_ML_READINESS_AUDIT.md
    §1.4): rows above the sanity ceiling must be dropped, real rows kept.
    """
    monkeypatch.setattr(loaders, "DIVIDENDS_DIR", tmp_path)

    good = pd.DataFrame({
        "ticker": ["AAAA3", "AAAA3"],
        "ex_date": pd.to_datetime(["2020-01-01", "2020-06-01"]),
        "value_per_share": [0.5, 1.2],
    })
    good.to_parquet(tmp_path / "AAAA3.parquet")

    bad = pd.DataFrame({
        "ticker": ["PDGR3"],
        "ex_date": pd.to_datetime(["2012-05-09"]),
        "value_per_share": [168_557_520.0],
    })
    bad.to_parquet(tmp_path / "PDGR3.parquet")

    result = loaders.load_dividends()

    assert len(result) == 2
    assert set(result["ticker"]) == {"AAAA3"}
    assert sorted(result["value_per_share"]) == [0.5, 1.2]


def test_load_prices_tickers_filter_loads_only_matching_files(tmp_path, monkeypatch) -> None:
    """tickers=None (default) must keep loading everything -- only a US-scale
    caller passing an explicit set should skip files, so BR (and every
    existing caller) sees no behavior change. Regression guard for
    build_us_dataset.py's gate-before-load fix (§8.0): loading only
    qualifying tickers' files, not the full universe then filtering after."""
    monkeypatch.setattr(loaders, "PRICES_DIR", tmp_path)

    for ticker in ("AAAA3", "BBBB3", "CCCC3"):
        pd.DataFrame({
            "ticker": [ticker], "trade_date": pd.to_datetime(["2020-01-01"]),
        }).to_parquet(tmp_path / f"{ticker}.parquet")

    result = loaders.load_prices(tickers={"AAAA3", "CCCC3"})
    assert set(result["ticker"]) == {"AAAA3", "CCCC3"}

    result_all = loaders.load_prices()
    assert set(result_all["ticker"]) == {"AAAA3", "BBBB3", "CCCC3"}


def test_load_fundamentals_drops_provenance_columns(tmp_path, monkeypatch) -> None:
    """US fundamentals files carry per-line-item filing dates and XBRL/item6/
    EX-27 source-document metadata that nothing in build_dataset reads --
    dropping them trims the width that gets forward-filled onto the much
    larger daily panel (docs/US_DATASET_BUILD_PLAN.md §8.0). fds_multiplier
    (the actually-applied rescale factor, real data) must survive; only
    fds_multiplier_explicit (a flag about it) is provenance."""
    monkeypatch.setattr(loaders, "FUNDAMENTALS_DIR", tmp_path)

    pd.DataFrame({
        "end": pd.to_datetime(["2020-03-31"]),
        "net_income": [100.0],
        "net_income_filed": pd.to_datetime(["2020-05-01"]),
        "item6_filename": ["foo.htm"],
        "item6_form": ["10-K"],
        "fds_multiplier": [1000.0],
        "fds_multiplier_explicit": [True],
    }).to_parquet(tmp_path / "AAAA.parquet")

    result = loaders.load_fundamentals()

    for dropped in ("net_income_filed", "item6_filename", "item6_form", "fds_multiplier_explicit"):
        assert dropped not in result.columns
    assert result["fds_multiplier"].iloc[0] == 1000.0


def test_load_fundamentals_optimize_dtypes_downcasts_numeric_keeps_cik(tmp_path, monkeypatch) -> None:
    """optimize_dtypes=False (default) must leave BR's float64 precision
    untouched -- existing CAGR tests assert output to 1e-6 tolerance against
    real BR figures, which float32 rounding on large monetary values would
    break. optimize_dtypes=True (US only) downcasts numeric columns to
    float32 but must not touch cik, an identifier, not a value."""
    monkeypatch.setattr(loaders, "FUNDAMENTALS_DIR", tmp_path)

    pd.DataFrame({
        "end": pd.to_datetime(["2020-03-31"]),
        "net_income": [100.0],
        "cik": [320193],
    }).to_parquet(tmp_path / "AAAA.parquet")

    default = loaders.load_fundamentals()
    assert default["net_income"].dtype == "float64"

    optimized = loaders.load_fundamentals(optimize_dtypes=True)
    assert optimized["net_income"].dtype == "float32"
    assert optimized["cik"].dtype == "int64"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
