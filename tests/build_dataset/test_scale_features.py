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
