#!/usr/bin/env python3
"""
write_manifest's lookahead_tainted_columns and column_units fields.

Run from project root: python tests/build_dataset/test_manifest.py
or: pytest tests/build_dataset/test_manifest.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_dataset import manifest as bmd


def test_manifest_records_lookahead_tainted_columns_present_in_dataset(tmp_path, monkeypatch) -> None:
    """status is a current-day snapshot joined onto every historical row (see
    merge_company_info) -- a model trained on it at a 2012 row would see 2026
    knowledge of whether the company survived. write_manifest must record this
    mechanically (not just as a CLAUDE.md caveat) so any future training/
    scaling consumer can check it programmatically."""
    monkeypatch.setattr(bmd, "OUTPUT_PATH", tmp_path / "ml_dataset.parquet")

    dataset = pd.DataFrame({
        "ticker": ["A", "A", "B"],
        "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
        "status": ["ATIVO", "ATIVO", "CANCELADA"],
        "pl": [10.0, 11.0, 5.0],
    })

    manifest = bmd.write_manifest(dataset)

    assert manifest["lookahead_tainted_columns"] == ["status"]


def test_manifest_flags_sector_derived_columns_as_tainted(tmp_path, monkeypatch) -> None:
    """The 6 cross_sectional.py columns engineered from the same static,
    current-day `sector` join (pl_zscore_sector, div_yield_sector_percentile,
    momentum_vs_sector_*) must be recorded alongside `status` -- dropping
    only raw status/sector isn't enough, since these numeric features carry
    the same taint laundered into a clean-looking z-score/percentile/
    momentum figure (2026-07-24 audit)."""
    monkeypatch.setattr(bmd, "OUTPUT_PATH", tmp_path / "ml_dataset.parquet")

    dataset = pd.DataFrame({
        "ticker": ["A", "B"],
        "trade_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "status": ["ATIVO", "CANCELADA"],
        "pl_zscore_sector": [0.5, -0.5],
        "div_yield_sector_percentile": [0.2, 0.8],
        "momentum_vs_sector_1m": [0.01, -0.01],
        "pl": [10.0, 5.0],
    })

    manifest = bmd.write_manifest(dataset)

    assert set(manifest["lookahead_tainted_columns"]) == {
        "status", "pl_zscore_sector", "div_yield_sector_percentile", "momentum_vs_sector_1m",
    }


def test_manifest_lookahead_tainted_columns_empty_when_absent(tmp_path, monkeypatch) -> None:
    """A dataset that never joined company_info (e.g. a narrow test fixture)
    must not claim status is tainted if the column doesn't even exist."""
    monkeypatch.setattr(bmd, "OUTPUT_PATH", tmp_path / "ml_dataset.parquet")

    dataset = pd.DataFrame({
        "ticker": ["A"],
        "trade_date": pd.to_datetime(["2020-01-01"]),
        "pl": [10.0],
    })

    manifest = bmd.write_manifest(dataset)

    assert manifest["lookahead_tainted_columns"] == []


def test_manifest_records_dropped_no_fundamentals_when_provided(tmp_path, monkeypatch) -> None:
    """quality_filters.filter_tickers_with_no_fundamentals's dropped_report
    must be threaded into the manifest so this source of universe/
    survivorship bias is queryable, not just printed to a build log that
    gets thrown away (2026-07-23 audit finding)."""
    monkeypatch.setattr(bmd, "OUTPUT_PATH", tmp_path / "ml_dataset.parquet")

    dataset = pd.DataFrame({
        "ticker": ["A"], "trade_date": pd.to_datetime(["2020-01-01"]), "pl": [10.0],
    })
    dropped = {"gap_unexplained": ["ZZZZ3"], "delisted_stale": ["DEAD3"]}

    manifest = bmd.write_manifest(dataset, dropped_no_fundamentals=dropped)

    assert manifest["dropped_no_fundamentals"] == dropped


def test_manifest_dropped_no_fundamentals_defaults_to_not_tracked(tmp_path, monkeypatch) -> None:
    """Callers that don't run the real Stage 2 filter pipeline (most tests,
    ad-hoc scripts) shouldn't need to fabricate a dropped_report -- the
    field must say so explicitly rather than silently claiming zero drops."""
    monkeypatch.setattr(bmd, "OUTPUT_PATH", tmp_path / "ml_dataset.parquet")

    dataset = pd.DataFrame({
        "ticker": ["A"], "trade_date": pd.to_datetime(["2020-01-01"]), "pl": [10.0],
    })

    manifest = bmd.write_manifest(dataset)

    assert manifest["dropped_no_fundamentals"] == "not tracked"


def test_manifest_records_macro_column_units_present_in_dataset(tmp_path, monkeypatch) -> None:
    """selic/cdi are daily-percent rates, ipca is a monthly-percent rate --
    a unit mismatch here previously caused a Critical audit finding
    (excess_return/real_return silently treated a daily rate as annual).
    Recorded mechanically so a future direct consumer of these columns
    doesn't repeat it."""
    monkeypatch.setattr(bmd, "OUTPUT_PATH", tmp_path / "ml_dataset.parquet")

    dataset = pd.DataFrame({
        "ticker": ["A"],
        "trade_date": pd.to_datetime(["2020-01-01"]),
        "selic": [0.05], "cdi": [0.04], "ipca": [0.4],
        "pl": [10.0],
    })

    manifest = bmd.write_manifest(dataset)

    assert set(manifest["column_units"]) == {"selic", "cdi", "ipca"}
    assert "month" in manifest["column_units"]["ipca"]
    assert "trading day" in manifest["column_units"]["selic"]


def test_manifest_column_units_empty_when_absent(tmp_path, monkeypatch) -> None:
    """A dataset that never merged macro series must not list units for
    columns that don't exist."""
    monkeypatch.setattr(bmd, "OUTPUT_PATH", tmp_path / "ml_dataset.parquet")

    dataset = pd.DataFrame({
        "ticker": ["A"],
        "trade_date": pd.to_datetime(["2020-01-01"]),
        "pl": [10.0],
    })

    manifest = bmd.write_manifest(dataset)

    assert manifest["column_units"] == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
