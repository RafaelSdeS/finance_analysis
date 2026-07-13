#!/usr/bin/env python3
"""
Sector/market-relative features. Mirrors src/build_dataset/cross_sectional.py.

Run from project root: python tests/build_dataset/test_cross_sectional.py
or: pytest tests/build_dataset/test_cross_sectional.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_dataset.cross_sectional import compute_cross_sectional_features


def _fill_advanced_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Constant-fill every column compute_advanced_features touches but this
    test doesn't care about, so callers only need to set up the columns
    relevant to what they're testing."""
    defaults = {
        "div_value_recent": 0.5, "lpa": 1.0, "ebitda": 100.0, "shares_outstanding": 1000.0,
        "net_revenue": 500.0, "net_income": 50.0, "revenue_growth_yoy": 0.05,
        "earnings_growth_yoy": 0.03, "volatility_20d": 0.1, "volatility_60d": 0.1,
        "adj_close": 100.0, "pl": 10.0, "drawdown": 0.0, "pvp": 2.0, "roe": 0.15,
        "debt_equity": 0.5, "div_yield_12m": 0.03, "return_1m": 0.01, "return_3m": 0.02,
        "return_12m": 0.05, "net_margin": 0.1, "roa": 0.05, "selic": 0.1,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    """Approximate equality allowing for floating-point rounding."""
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) < tol


def test_momentum_vs_sector_nan_for_sector_of_one() -> None:
    """A ticker alone in its sector (e.g. B3SA3 in "Bolsas de Valores") has no
    peer to compare against: 'return - mean(return of itself)' trivially gives
    0.0, which reads as 'moved exactly with its sector' — a lie, since there
    is no sector. Must be NaN instead, same treatment as *_zscore_sector.

    These are cross-sectional features (compute_cross_sectional_features), not
    compute_advanced_features -- they must run on the full universe, never on a
    per-ticker batch (see compute_cross_sectional_features docstring)."""
    date = pd.Timestamp("2026-01-01")
    df = pd.DataFrame({
        "ticker": ["LONE", "PEER1", "PEER2"],
        "sector": ["Solo", "Multi", "Multi"],
        "trade_date": [date] * 3,
        "reference_date": [date] * 3,
        "return_1m": [0.05, 0.05, 0.03],
        "return_3m": [0.05, 0.05, 0.03],
        "return_12m": [0.05, 0.05, 0.03],
        "div_yield_12m": [0.02, 0.02, 0.04],
        "pl": [10.0, 12.0, 8.0],
    })
    df = _fill_advanced_feature_columns(df)
    result = compute_cross_sectional_features(df).set_index("ticker")

    lone = result.loc["LONE"]
    assert pd.isna(lone["momentum_vs_sector_1m"])
    assert pd.isna(lone["momentum_vs_sector_3m"])
    assert pd.isna(lone["momentum_vs_sector_12m"])
    assert pd.isna(lone["div_yield_sector_percentile"])
    assert pd.isna(lone["pl_zscore_sector"])

    # PEER1/PEER2 share a real sector -> real (non-NaN) relative values
    peer_mean = (0.05 + 0.03) / 2
    assert approx(result.loc["PEER1", "momentum_vs_sector_1m"], 0.05 - peer_mean)
    assert approx(result.loc["PEER2", "momentum_vs_sector_1m"], 0.03 - peer_mean)
    assert not pd.isna(result.loc["PEER1", "pl_zscore_sector"])
    assert not pd.isna(result.loc["PEER2", "pl_zscore_sector"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
