#!/usr/bin/env python3
"""
write_manifest's lookahead_tainted_columns field.

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
