#!/usr/bin/env python3
"""
dataset_v{N} snapshot versioning (sync_dataset_version).

Run from project root: python tests/build_dataset/test_dataset_versioning.py
or: pytest tests/build_dataset/test_dataset_versioning.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_dataset import manifest as bmd

BASE_MANIFEST = {
    "rows": 100,
    "tickers": 2,
    "date_min": "2020-01-01",
    "date_max": "2020-04-01",
    "columns": ["ticker", "trade_date", "close"],
    "column_stats": {"close": {"mean": 10.0}},
}


def _write_current_build(tmp_path, manifest):
    output_path = tmp_path / "ml_dataset.parquet"
    output_path.write_text("fake parquet bytes")
    (tmp_path / "ml_dataset.manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "split_config.json").write_text(json.dumps({"train_end": "2020-03-01"}))
    return output_path


@pytest.fixture(autouse=True)
def _isolate_scaler_dir(tmp_path, monkeypatch):
    # Points at a directory that doesn't exist by default (sync_dataset_version
    # must skip it, not error) -- isolates tests from this machine's real
    # data/processed/scalers/ (paths.SCALER_DIR), which would otherwise get
    # silently copied into every test's tmp_path version dir.
    monkeypatch.setattr(bmd, "SCALER_DIR", tmp_path / "scalers")


def test_first_build_creates_v1(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bmd, "OUTPUT_PATH", _write_current_build(tmp_path, BASE_MANIFEST))
    monkeypatch.setattr(bmd, "SPLIT_CONFIG_PATH", tmp_path / "split_config.json")

    bmd.sync_dataset_version(dict(BASE_MANIFEST, built_at="t1", git_commit="a"))

    assert (tmp_path / "dataset_v1" / "ml_dataset.parquet").exists()
    assert (tmp_path / "dataset_v1" / "split_config.json").exists()


def test_unchanged_rerun_is_skipped(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bmd, "OUTPUT_PATH", _write_current_build(tmp_path, BASE_MANIFEST))
    monkeypatch.setattr(bmd, "SPLIT_CONFIG_PATH", tmp_path / "split_config.json")

    bmd.sync_dataset_version(dict(BASE_MANIFEST, built_at="t1", git_commit="a"))
    bmd.sync_dataset_version(dict(BASE_MANIFEST, built_at="t2", git_commit="b"))  # same content, rerun

    versions = sorted(p.name for p in tmp_path.glob("dataset_v*"))
    assert versions == ["dataset_v1"]  # no dataset_v2 created


def test_content_change_creates_v2(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bmd, "OUTPUT_PATH", _write_current_build(tmp_path, BASE_MANIFEST))
    monkeypatch.setattr(bmd, "SPLIT_CONFIG_PATH", tmp_path / "split_config.json")
    bmd.sync_dataset_version(dict(BASE_MANIFEST, built_at="t1", git_commit="a"))

    changed_manifest = dict(BASE_MANIFEST, rows=200)
    _write_current_build(tmp_path, changed_manifest)
    bmd.sync_dataset_version(dict(changed_manifest, built_at="t2", git_commit="b"))

    versions = sorted(p.name for p in tmp_path.glob("dataset_v*"))
    assert versions == ["dataset_v1", "dataset_v2"]


def test_scalers_snapshotted_into_version_dir(tmp_path, monkeypatch) -> None:
    """sync_dataset_version must copy scalers/ into dataset_v{N}/ too -- so an
    experiment citing dataset_v{N} gets the scaler it actually used, not just
    the parquet (previously a gap: scalers were never versioned)."""
    monkeypatch.setattr(bmd, "OUTPUT_PATH", _write_current_build(tmp_path, BASE_MANIFEST))
    monkeypatch.setattr(bmd, "SPLIT_CONFIG_PATH", tmp_path / "split_config.json")
    scaler_dir = tmp_path / "scalers"
    scaler_dir.mkdir()
    (scaler_dir / "feature_scaler.joblib").write_text("fake scaler bytes")
    (scaler_dir / "scaler_metadata.json").write_text("{}")
    monkeypatch.setattr(bmd, "SCALER_DIR", scaler_dir)

    bmd.sync_dataset_version(dict(BASE_MANIFEST, built_at="t1", git_commit="a"))

    assert (tmp_path / "dataset_v1" / "scalers" / "feature_scaler.joblib").read_text() == "fake scaler bytes"
    assert (tmp_path / "dataset_v1" / "scalers" / "scaler_metadata.json").exists()


def test_nan_regressions_detects_increase() -> None:
    """nan_regressions reports columns whose nan_pct rose by >threshold."""
    prev = {
        "column_stats": {
            "col_a": {"nan_pct": 10.0},
            "col_b": {"nan_pct": 5.0},
            "col_c": {"nan_pct": 20.0},
        }
    }
    curr = {
        "column_stats": {
            "col_a": {"nan_pct": 11.5},  # +1.5 pp, below threshold
            "col_b": {"nan_pct": 8.0},   # +3.0 pp, exceeds threshold (2.0)
            "col_c": {"nan_pct": 20.0},  # no change
        }
    }
    regressions = bmd.nan_regressions(prev, curr, threshold=2.0)
    assert len(regressions) == 1
    assert "col_b" in regressions[0]


def test_nan_regressions_ignores_new_columns() -> None:
    """nan_regressions doesn't report columns only in the new manifest (not a regression)."""
    prev = {"column_stats": {"col_a": {"nan_pct": 10.0}}}
    curr = {
        "column_stats": {
            "col_a": {"nan_pct": 10.0},
            "col_b": {"nan_pct": 99.0},  # new column, shouldn't be reported
        }
    }
    regressions = bmd.nan_regressions(prev, curr, threshold=2.0)
    assert len(regressions) == 0


def test_nan_regressions_empty_when_no_increase() -> None:
    """nan_regressions returns empty list when no column exceeds threshold."""
    prev = {"column_stats": {"col_a": {"nan_pct": 10.0}}}
    curr = {"column_stats": {"col_a": {"nan_pct": 11.0}}}  # +1.0 pp
    regressions = bmd.nan_regressions(prev, curr, threshold=2.0)
    assert regressions == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
