#!/usr/bin/env python3
"""
Final cleaning pass: dedupe, inf->NaN, sort. Mirrors src/build_dataset/clean.py.

Previously only exercised as a byproduct of the end-to-end comparison in
test_compute_features_chunked.py and of validating the real production
dataset in test_final_dataset.py -- never given a fixture that isolates
each branch (exact duplicate vs. near-duplicate, a genuine inf value).

Run from project root: python tests/build_dataset/test_clean.py
or: pytest tests/build_dataset/test_clean.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_dataset.clean import clean_dataset


def test_exact_duplicate_row_removed() -> None:
    """A row that's byte-for-byte identical to another (every column, not
    just the natural key) is a real duplicate and must be dropped."""
    df = pd.DataFrame({
        "ticker": ["A", "A", "B"],
        "trade_date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-01"]),
        "close": [10.0, 10.0, 20.0],
    })

    result = clean_dataset(df)

    assert len(result) == 2
    assert set(result["ticker"]) == {"A", "B"}


def test_near_duplicate_survives() -> None:
    """Same (ticker, trade_date) but a genuinely different value elsewhere is
    NOT a duplicate -- clean_dataset must never collapse it away just because
    it shares the natural key with another row."""
    df = pd.DataFrame({
        "ticker": ["A", "A"],
        "trade_date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        "close": [10.0, 10.5],  # differs -> not a true duplicate
    })

    result = clean_dataset(df)

    assert len(result) == 2


def test_inf_replaced_with_nan_other_columns_untouched() -> None:
    """Literal inf/-inf (division-by-zero in a ratio or growth rate) must
    become NaN so it never reaches training/inference; finite values in
    other numeric columns must be untouched."""
    df = pd.DataFrame({
        "ticker": ["A", "B"],
        "trade_date": pd.to_datetime(["2026-01-02", "2026-01-01"]),
        "pl": [np.inf, 12.5],
        "hl_ratio": [0.05, -np.inf],
    })

    result = clean_dataset(df).set_index("ticker")

    assert pd.isna(result.loc["A", "pl"])
    assert result.loc["B", "pl"] == 12.5
    assert result.loc["A", "hl_ratio"] == 0.05
    assert pd.isna(result.loc["B", "hl_ratio"])


def test_sorted_by_ticker_then_trade_date_with_reset_index() -> None:
    """Output must be sorted (ticker, trade_date) ascending with a clean
    0..n-1 index, regardless of input order."""
    df = pd.DataFrame({
        "ticker": ["B", "A", "A"],
        "trade_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01"]),
        "close": [1.0, 2.0, 3.0],
    })

    result = clean_dataset(df)

    assert list(result["ticker"]) == ["A", "A", "B"]
    assert list(result["trade_date"]) == list(pd.to_datetime(
        ["2026-01-01", "2026-01-02", "2026-01-01"]
    ))
    assert list(result.index) == [0, 1, 2]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
