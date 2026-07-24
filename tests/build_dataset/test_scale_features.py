#!/usr/bin/env python3
"""
Feature scaler: train-only fit, reindexing, and no-leak/no-scale-drift gates.

Run from project root: python tests/build_dataset/test_scale_features.py
or: pytest tests/build_dataset/test_scale_features.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_dataset import scale_features as sf


def _synthetic_dataset(n=40):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "ticker": ["AAAA"] * n,
        "trade_date": dates,
        "pl": rng.normal(10, 2, n),
        "roe": rng.normal(0.15, 0.05, n),
        "pl_zscore_sector": rng.normal(0, 1, n),  # already scaled -> must pass through
        "drawdown_percentile": rng.uniform(0, 1, n),  # already bounded -> must pass through
    })


def test_transform_restores_original_column_order() -> None:
    df = _synthetic_dataset()
    ct = sf.build_scaler(ratio_columns=["pl", "roe"])
    ct.fit(df)

    out = sf.transform_features(ct, df)

    assert list(out.columns) == list(df.columns)


def test_ratio_columns_scaled_others_untouched() -> None:
    df = _synthetic_dataset()
    ct = sf.build_scaler(ratio_columns=["pl", "roe"])
    ct.fit(df)

    out = sf.transform_features(ct, df)

    assert not np.allclose(out["pl"].to_numpy(), df["pl"].to_numpy())
    assert np.allclose(out["pl_zscore_sector"].to_numpy(), df["pl_zscore_sector"].to_numpy())
    assert np.allclose(out["drawdown_percentile"].to_numpy(), df["drawdown_percentile"].to_numpy())
    assert (out["ticker"] == df["ticker"]).all()


def test_nan_preserved_not_imputed() -> None:
    df = _synthetic_dataset()
    df.loc[0, "pl"] = np.nan
    ct = sf.build_scaler(ratio_columns=["pl", "roe"])
    ct.fit(df)

    out = sf.transform_features(ct, df)

    assert pd.isna(out.loc[0, "pl"])


def _synthetic_dataset_full_ratio_columns(n=40):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"ticker": ["AAAA"] * n, "trade_date": dates})
    for col in sf.RATIO_COLUMNS:
        df[col] = rng.normal(1, 0.3, n)
    return df


def test_refit_on_train_split_is_reproducible(tmp_path, monkeypatch) -> None:
    df = _synthetic_dataset_full_ratio_columns()
    split_config = tmp_path / "split_config.json"
    split_config.write_text(json.dumps({"train_end": str(df["trade_date"].iloc[29].date())}))
    monkeypatch.setattr(sf, "SPLIT_CONFIG_PATH", split_config)

    ct_a = sf.fit_scaler_on_train_split(df)
    ct_b = sf.fit_scaler_on_train_split(df)  # re-fit on the same train rows

    robust_a = ct_a.named_transformers_["robust"]
    robust_b = ct_b.named_transformers_["robust"]
    assert np.allclose(robust_a.center_, robust_b.center_)
    assert np.allclose(robust_a.scale_, robust_b.scale_)


def test_fit_honors_arbitrary_window() -> None:
    """fit_scaler must depend only on rows inside the injected FitWindow, not
    on any hardcoded boundary -- the evaluation methodology (single split,
    rolling, expanding, multi-fold) is entirely the caller's choice of
    windows (docs/PER_TICKER_SCALING_PLAN.md §3.5)."""
    df = _synthetic_dataset_full_ratio_columns(n=60)

    # Window A: expanding from the start through row 29.
    window_a = sf.FitWindow(fold_id="a", fit_start=None, fit_end=df["trade_date"].iloc[29])
    # Window B: a rolling window covering only rows 30-59 (excludes what A used).
    window_b = sf.FitWindow(
        fold_id="b", fit_start=df["trade_date"].iloc[29], fit_end=df["trade_date"].iloc[59]
    )

    ct_a = sf.fit_scaler(df, window_a)
    ct_b = sf.fit_scaler(df, window_b)

    # Reproduces today's single-split config exactly (fit_scaler_on_train_split
    # filters trade_date <= train_end with no lower bound).
    ct_direct = sf.build_scaler()
    ct_direct.fit(df[df["trade_date"] <= window_a.fit_end])
    robust_a = ct_a.named_transformers_["robust"]
    robust_direct = ct_direct.named_transformers_["robust"]
    assert np.allclose(robust_a.center_, robust_direct.center_)

    # Disjoint windows over different (independently drawn) rows must not
    # coincidentally fit to the same statistics.
    robust_b = ct_b.named_transformers_["robust"]
    assert not np.allclose(robust_a.center_, robust_b.center_)


def test_write_scaler_metadata_records_fit_window(tmp_path) -> None:
    """Metadata must record the FitWindow that produced the artifact -- so a
    params/split mismatch is detectable at load time instead of silently
    transforming with stale boundaries under a changed evaluation config."""
    df = _synthetic_dataset_full_ratio_columns(n=40)
    window = sf.FitWindow(fold_id="fold_a", fit_start=None, fit_end=df["trade_date"].iloc[29])
    ct = sf.fit_scaler(df, window)

    metadata_path = tmp_path / "scaler_metadata.json"
    sf.write_scaler_metadata(ct, window, metadata_path)

    metadata = json.loads(metadata_path.read_text())
    assert metadata["fit_window"]["fold_id"] == "fold_a"
    assert metadata["fit_window"]["fit_start"] is None
    assert metadata["fit_window"]["fit_end"] == str(window.fit_end.date())


def test_write_scaler_metadata_flags_lookahead_tainted_columns_present() -> None:
    """status passes through the scaler untouched (not a ratio column) but
    must never be fed to a model as a feature -- merge_company_info() joins
    TODAY's status onto every historical row (survivorship leakage). The
    scaler metadata is the closest thing this repo has to a feature spec, so
    it must record this mechanically, not rely on CLAUDE.md prose alone."""
    df = _synthetic_dataset_full_ratio_columns(n=40)
    df["status"] = "ATIVO"
    window = sf.FitWindow(fold_id="full", fit_start=None, fit_end=df["trade_date"].iloc[29])
    ct = sf.fit_scaler(df, window)

    metadata = {}
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        metadata_path = Path(tmp) / "scaler_metadata.json"
        sf.write_scaler_metadata(ct, window, metadata_path)
        metadata = json.loads(metadata_path.read_text())

    assert "status" in metadata["passthrough_columns"]
    assert metadata["lookahead_tainted_columns"] == ["status"]


def test_ratio_columns_never_include_already_bounded_columns() -> None:
    # Guards the real module-level column list: percentiles/z-scores/binary
    # flags are already [0,1] or already unit-scaled and must stay passthrough.
    bounded_examples = {
        "volatility_20d_percentile", "volatility_60d_percentile",
        "price_percentile_5y", "pl_percentile_5y", "drawdown_percentile",
        "div_yield_sector_percentile", "has_fundamentals", "f_score",
        "f_roa_positive", "had_negative_earnings_5y",
        "pl_zscore_sector", "pvp_zscore_sector", "roe_zscore_sector",
        "debt_equity_zscore_sector",
    }
    assert bounded_examples.isdisjoint(sf.RATIO_COLUMNS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
