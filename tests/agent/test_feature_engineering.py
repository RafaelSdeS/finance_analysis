#!/usr/bin/env python3
"""
Feature engineering: returns computation from split-adjusted prices.

Run from project root: python tests/agent/test_feature_engineering.py
or: pytest tests/agent/test_feature_engineering.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.feature_engineering import compute_returns, MAX_ABS_LOG_RETURN


def test_basic_log_returns() -> None:
    """Basic log-return: [100, 102, 101] → [NaN, log(1.02), log(101/102)]."""
    df = pd.DataFrame({
        "ticker": ["A", "A", "A"],
        "adj_close": [100.0, 102.0, 101.0]
    })
    result = compute_returns(df, price_col="adj_close", ticker_col="ticker")

    assert pd.isna(result.iloc[0]["returns"]), "first row should be NaN (no prior price)"
    assert np.isclose(result.iloc[1]["returns"], np.log(102.0 / 100.0), atol=1e-10)
    assert np.isclose(result.iloc[2]["returns"], np.log(101.0 / 102.0), atol=1e-10)


def test_non_positive_prices() -> None:
    """Non-positive prices → NaN returns (no crashes, no fake huge returns)."""
    df = pd.DataFrame({
        "ticker": ["A", "A", "A", "A"],
        "adj_close": [100.0, 0.0, -50.0, 101.0]
    })
    result = compute_returns(df)

    assert pd.isna(result.iloc[0]["returns"]), "first row: no prior price"
    assert pd.isna(result.iloc[1]["returns"]), "zero price → NaN"
    assert pd.isna(result.iloc[2]["returns"]), "negative price → NaN"
    # Row 3: prior price is NaN (row 2), so log(NaN/NaN) → NaN
    assert pd.isna(result.iloc[3]["returns"])


def test_extreme_returns_clipped() -> None:
    """Extreme returns (|log r| > 1.0) → NaN (split residues, data glitches)."""
    df = pd.DataFrame({
        "ticker": ["A", "A", "A"],
        "adj_close": [100.0, 100.0 * np.exp(1.5), 100.0]  # +150% return, then crash
    })
    result = compute_returns(df)

    assert pd.isna(result.iloc[0]["returns"])
    assert pd.isna(result.iloc[1]["returns"]), f"|1.5| > {MAX_ABS_LOG_RETURN} → should be NaN"
    # Row 2: prior price has NaN marker, returns is NaN
    assert pd.isna(result.iloc[2]["returns"])


def test_ticker_grouping() -> None:
    """Multiple tickers don't cross-contaminate returns (grouped computation)."""
    df = pd.DataFrame({
        "ticker": ["A", "A", "B", "B"],
        "adj_close": [100.0, 102.0, 50.0, 52.0]
    })
    result = compute_returns(df)

    # Ticker A: [NaN, log(1.02)]
    assert pd.isna(result.iloc[0]["returns"])
    assert np.isclose(result.iloc[1]["returns"], np.log(102.0 / 100.0))

    # Ticker B: [NaN, log(1.04)], NOT log(52/102) cross-ticker
    assert pd.isna(result.iloc[2]["returns"])
    assert np.isclose(result.iloc[3]["returns"], np.log(52.0 / 50.0))


def test_constant_prices() -> None:
    """Prices unchanged → returns ≈ 0."""
    df = pd.DataFrame({
        "ticker": ["A", "A", "A"],
        "adj_close": [100.0, 100.0, 100.0]
    })
    result = compute_returns(df)

    assert pd.isna(result.iloc[0]["returns"])
    assert np.isclose(result.iloc[1]["returns"], 0.0, atol=1e-10)
    assert np.isclose(result.iloc[2]["returns"], 0.0, atol=1e-10)


def test_single_row_per_ticker() -> None:
    """Single row per ticker → returns all NaN (no prior price for any)."""
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "adj_close": [100.0, 50.0, 75.0]
    })
    result = compute_returns(df)

    assert result["returns"].isna().all(), "all rows have no prior price"


def test_empty_dataframe() -> None:
    """Empty df → empty df with returns column (no crashes)."""
    df = pd.DataFrame({"ticker": [], "adj_close": []})
    result = compute_returns(df)

    assert len(result) == 0
    assert "returns" in result.columns


def test_preserves_other_columns() -> None:
    """compute_returns adds 'returns' but preserves other columns."""
    df = pd.DataFrame({
        "ticker": ["A", "A"],
        "date": ["2026-01-01", "2026-01-02"],
        "adj_close": [100.0, 102.0]
    })
    result = compute_returns(df)

    assert "date" in result.columns
    assert "returns" in result.columns
    assert result.iloc[0]["date"] == "2026-01-01"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
