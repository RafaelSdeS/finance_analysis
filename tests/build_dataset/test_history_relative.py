#!/usr/bin/env python3
"""
Per-ticker own-history z-scores (compute_history_relative_features, R1 in
docs/PER_TICKER_SCALING_PLAN.md): no-lookahead, warm-up prefix shape,
quarterly (not daily-redundant) fundamentals rolling, and NaN/IQR=0 handling.

Run from project root: python tests/build_dataset/test_history_relative.py
or: pytest tests/build_dataset/test_history_relative.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_dataset.features import (
    DAILY_ZHIST_COLS,
    DAILY_ZHIST_MIN_DAYS,
    FUND_ZHIST_COLS,
    FUND_ZHIST_MIN_QUARTERS,
    compute_history_relative_features,
)


_FUND_LOC_SCALE = {
    "pl": (10.0, 2.0), "pvp": (1.5, 0.3), "roe": (0.15, 0.03),
    "net_margin": (0.1, 0.02), "ebitda_margin": (0.2, 0.03),
    "debt_equity": (0.5, 0.1), "net_debt_ebitda": (2.0, 0.4),
    "earnings_yield": (0.08, 0.02), "book_to_market": (0.6, 0.1),
    "current_ratio": (1.5, 0.2), "asset_turnover": (0.7, 0.1),
}


def _history_relative_fixture(n_quarters=12, days_per_quarter=65, ticker="A", seed=0):
    """One ticker, n_quarters distinct filings (days_per_quarter daily rows
    each, forward-filled like the real merged panel), plus mildly noisy daily
    liquidity columns so the daily-window zhist has real variance to measure.
    """
    quarter_ends = pd.date_range("2020-03-31", periods=n_quarters, freq="QE")
    rng = np.random.default_rng(seed)
    fund_series = {col: rng.normal(loc, scale, n_quarters) for col, (loc, scale) in _FUND_LOC_SCALE.items()}

    rows = []
    for qi, qe in enumerate(quarter_ends):
        for d in pd.date_range(qe, periods=days_per_quarter):
            row = {"ticker": ticker, "trade_date": d, "reference_date": qe}
            for col in _FUND_LOC_SCALE:
                row[col] = fund_series[col][qi]
            row["amihud_illiquidity"] = abs(rng.normal(1e-6, 2e-7))
            row["turnover_ratio"] = abs(rng.normal(0.01, 0.002))
            rows.append(row)
    return pd.DataFrame(rows)


def _zhist_cols():
    return [f"{c}_zhist_5y" for c in FUND_ZHIST_COLS + DAILY_ZHIST_COLS]


def test_zhist_truncation_invariance() -> None:
    """A row's zhist value must not depend on any row after it -- the same
    no-lookahead property already guarded for volatility_20d_percentile etc.
    (test_features.py::test_volatility_percentile_no_lookahead), extended to
    the new own-history columns."""
    df = _history_relative_fixture(n_quarters=12, days_per_quarter=65)
    cutoff_quarter_end = sorted(df["reference_date"].unique())[7]  # keep the first 8 quarters
    truncated = df[df["reference_date"] <= cutoff_quarter_end].copy()

    full = compute_history_relative_features(df.copy()).set_index(["ticker", "trade_date"]).sort_index()
    trunc = compute_history_relative_features(truncated).set_index(["ticker", "trade_date"]).sort_index()

    for col in _zhist_cols():
        pd.testing.assert_series_equal(
            full.loc[trunc.index, col], trunc[col], check_names=False,
            obj=f"{col} differs between full and truncated computation -- lookahead",
        )


def test_zhist_warmup_is_prefix_nan() -> None:
    """First non-NaN value must land exactly at the min_periods threshold --
    no interior holes, no early leakage from an under-filled window."""
    n_quarters = 12
    days_per_quarter = 65
    df = _history_relative_fixture(n_quarters=n_quarters, days_per_quarter=days_per_quarter)

    result = compute_history_relative_features(df)

    # Fundamentals: quarters are 0-indexed; first FUND_ZHIST_MIN_QUARTERS-1
    # quarters must be entirely NaN, quarter index FUND_ZHIST_MIN_QUARTERS-1
    # (the 8th quarter) onward must be non-NaN.
    quarter_ends = sorted(df["reference_date"].unique())
    col = f"{FUND_ZHIST_COLS[0]}_zhist_5y"
    for qi, qe in enumerate(quarter_ends):
        block = result.loc[result["reference_date"] == qe, col]
        if qi < FUND_ZHIST_MIN_QUARTERS - 1:
            assert block.isna().all(), f"quarter {qi}: expected NaN warm-up, got a value"
        else:
            assert block.notna().all(), f"quarter {qi}: expected a value past warm-up"

    # Daily liquidity column: first DAILY_ZHIST_MIN_DAYS-1 rows NaN, then non-NaN.
    daily_col = f"{DAILY_ZHIST_COLS[0]}_zhist_5y"
    ordered = result.sort_values("trade_date").reset_index(drop=True)
    assert ordered.loc[: DAILY_ZHIST_MIN_DAYS - 2, daily_col].isna().all()
    assert ordered.loc[DAILY_ZHIST_MIN_DAYS - 1 :, daily_col].notna().all()


def test_zhist_uses_quarterly_observations_not_daily_rows() -> None:
    """Fundamentals are forward-filled ~65 daily rows/quarter -- rolling
    directly on the daily panel would be ~65x redundant and the window would
    advance per row instead of per real filing. Every daily row within one
    quarter must share the exact same zhist value (mirrors
    test_features.py::test_trend_4q_uses_real_quarters_not_daily_rows)."""
    df = _history_relative_fixture(n_quarters=12, days_per_quarter=65)

    result = compute_history_relative_features(df)

    for col in FUND_ZHIST_COLS:
        per_quarter_values = result.groupby("reference_date")[f"{col}_zhist_5y"].nunique(dropna=False)
        assert (per_quarter_values <= 1).all(), f"{col}_zhist_5y varies within a single quarter's rows"


def test_zhist_nan_preserved_and_iqr_zero_is_nan() -> None:
    """NaN input stays NaN (no imputation); a perfectly constant window
    (IQR == 0) yields NaN, not +-inf.

    Needs >= FUND_ZHIST_WINDOW_QUARTERS (20) constant quarters so the
    trailing window at the last quarter is ENTIRELY inside the constant
    stretch -- a shorter constant tail would still have older, non-constant
    quarters inside the window and IQR wouldn't be exactly 0.
    """
    n_quarters = 30
    df = _history_relative_fixture(n_quarters=n_quarters, days_per_quarter=10)
    quarter_ends = sorted(df["reference_date"].unique())

    const_from = quarter_ends[10]  # quarters 10..29 = 20 constant quarters, == the window size
    df.loc[df["reference_date"] >= const_from, "pl"] = 7.5

    # A genuine NaN fundamental in an unrelated quarter/column.
    nan_quarter = quarter_ends[20]
    df.loc[df["reference_date"] == nan_quarter, "roe"] = np.nan

    result = compute_history_relative_features(df)

    last_quarter = quarter_ends[-1]
    pl_last = result.loc[result["reference_date"] == last_quarter, "pl_zhist_5y"]
    assert pl_last.isna().all(), "perfectly constant trailing window must yield NaN, not +-inf"

    roe_nan_rows = result.loc[result["reference_date"] == nan_quarter, "roe_zhist_5y"]
    assert roe_nan_rows.isna().all(), "NaN input must propagate to NaN output, not get imputed"


def test_zhist_continuous_across_ticker_history_boundary() -> None:
    """A rename/merger splice (continuity.py::apply_ticker_continuity) runs
    before Pass 1 and produces one continuous ticker label spanning what were
    two vendor legs -- compute_history_relative_features groups only by
    ticker, so it must build on the full combined history rather than
    resetting to a cold NaN prefix right after the splice point."""
    first_leg = _history_relative_fixture(n_quarters=9, days_per_quarter=20, seed=1)
    second_leg = _history_relative_fixture(n_quarters=3, days_per_quarter=20, seed=2)
    second_leg["reference_date"] = second_leg["reference_date"] + pd.DateOffset(years=3)
    second_leg["trade_date"] = second_leg["trade_date"] + pd.DateOffset(years=3)
    spliced = pd.concat([first_leg, second_leg], ignore_index=True)  # one ticker label throughout

    result = compute_history_relative_features(spliced)

    first_quarter_after_splice = sorted(second_leg["reference_date"].unique())[0]
    block = result.loc[result["reference_date"] == first_quarter_after_splice, "pl_zhist_5y"]
    assert block.notna().all(), (
        "first quarter past the splice boundary should already have >= "
        f"{FUND_ZHIST_MIN_QUARTERS} quarters of combined history and must not be NaN"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
