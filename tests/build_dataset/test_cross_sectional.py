#!/usr/bin/env python3
"""
Sector/market-relative features. Mirrors src/build_dataset/cross_sectional.py.

Run from project root: python tests/build_dataset/test_cross_sectional.py
or: pytest tests/build_dataset/test_cross_sectional.py -v
"""

import sys
from pathlib import Path

import numpy as np
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
        "log_return": 0.001,
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


def test_cross_sectional_values_hand_computed_multi_peer() -> None:
    """The singleton-sector test only proves NaN-vs-lie in the degenerate
    case; this checks the normal multi-peer case actually computes the right
    numbers, not just something internally consistent between chunked and
    unchunked runs (test_compute_features_chunked.py proves consistency, not
    correctness -- a shared bug would pass both).

    4 tickers, 2 sectors of 2 (X: T1/T2, Y: T3/T4), one date, hand-computed
    expected z-scores (pandas default ddof=1 std), percentiles, and momentum
    vs. market (all 4 tickers) vs. sector (2 peers each)."""
    date = pd.Timestamp("2026-01-01")
    df = pd.DataFrame({
        "ticker": ["T1", "T2", "T3", "T4"],
        "sector": ["X", "X", "Y", "Y"],
        "trade_date": [date] * 4,
        "reference_date": [date] * 4,
        "pl": [10.0, 14.0, 8.0, 12.0],
        "pvp": [10.0, 14.0, 8.0, 12.0],
        "roe": [0.10, 0.14, 0.08, 0.12],
        "debt_equity": [10.0, 14.0, 8.0, 12.0],
        "div_yield_12m": [0.02, 0.05, 0.03, 0.01],
        "return_1m": [0.02, 0.04, -0.01, 0.03],
        "return_3m": [0.05, 0.09, -0.02, 0.06],
        "return_12m": [0.02, 0.04, -0.01, 0.03],
        "log_return": [0.001, 0.002, -0.0005, 0.0015],
    })
    result = compute_cross_sectional_features(df).set_index("ticker")

    # z-score: pandas groupby std uses ddof=1 -> sector X [10,14]: mean=12, std=sqrt(8)
    std_x = ((10.0 - 12.0) ** 2 + (14.0 - 12.0) ** 2) ** 0.5  # ddof=1, n=2 -> /1
    assert approx(result.loc["T1", "pl_zscore_sector"], (10.0 - 12.0) / std_x)
    assert approx(result.loc["T2", "pl_zscore_sector"], (14.0 - 12.0) / std_x)
    # roe uses the same formula on independent values -> proportional result
    std_roe_x = ((0.10 - 0.12) ** 2 + (0.14 - 0.12) ** 2) ** 0.5
    assert approx(result.loc["T1", "roe_zscore_sector"], (0.10 - 0.12) / std_roe_x)

    # div_yield percentile within sector: rank(pct=True), default 'average' method
    assert approx(result.loc["T1", "div_yield_sector_percentile"], 0.5)   # lower of the pair
    assert approx(result.loc["T2", "div_yield_sector_percentile"], 1.0)   # higher of the pair
    assert approx(result.loc["T4", "div_yield_sector_percentile"], 0.5)   # sector Y: T4 (0.01) lower
    assert approx(result.loc["T3", "div_yield_sector_percentile"], 1.0)

    # momentum vs market: subtract the mean return_1m of the OTHER 3 tickers
    # (self-excluded, 2026-07-23 fix -- a ticker's own return no longer pulls
    # its own benchmark toward itself). (total - self) / (n - 1).
    total_1m = 0.02 + 0.04 - 0.01 + 0.03
    market_mean_1m_excl_t1 = (total_1m - 0.02) / 3
    market_mean_1m_excl_t3 = (total_1m - (-0.01)) / 3
    assert approx(result.loc["T1", "momentum_vs_market_1m"], 0.02 - market_mean_1m_excl_t1)
    assert approx(result.loc["T3", "momentum_vs_market_1m"], -0.01 - market_mean_1m_excl_t3)

    # momentum vs sector: subtract the mean within the 2-ticker sector only.
    # Unchanged by the self-exclusion fix -- only momentum_vs_MARKET/beta_1y
    # were flagged as self-inclusive (2026-07-23 audit); sector momentum's
    # self-inclusive mean matches this codebase's z-score convention
    # elsewhere (a stock's sector stat includes itself, standard semantics)
    # and wasn't part of that finding.
    sector_x_mean_1m = (0.02 + 0.04) / 2
    sector_y_mean_1m = (-0.01 + 0.03) / 2
    assert approx(result.loc["T1", "momentum_vs_sector_1m"], 0.02 - sector_x_mean_1m)
    assert approx(result.loc["T4", "momentum_vs_sector_1m"], 0.03 - sector_y_mean_1m)

    # return_3m uses independent values -> proves the 1m/3m/12m columns aren't aliased
    total_3m = 0.05 + 0.09 - 0.02 + 0.06
    market_mean_3m_excl_t2 = (total_3m - 0.09) / 3
    sector_x_mean_3m = (0.05 + 0.09) / 2
    assert approx(result.loc["T2", "momentum_vs_market_3m"], 0.09 - market_mean_3m_excl_t2)
    assert approx(result.loc["T2", "momentum_vs_sector_3m"], 0.09 - sector_x_mean_3m)


def _beta_fixture(n_days: int = 80, seed: int = 0):
    """3 tickers, n_days of independent log_return, one sector (sector isn't
    relevant to beta -- only market_log_return, the mean across ALL tickers
    on each date, is)."""
    dates = pd.date_range("2026-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    returns = {
        "A": rng.normal(0, 0.010, n_days),
        "B": rng.normal(0, 0.015, n_days),
        "C": rng.normal(0, 0.008, n_days),
    }
    rows = []
    for t, ret in returns.items():
        for i, d in enumerate(dates):
            rows.append({
                "ticker": t, "sector": "X", "trade_date": d, "reference_date": d,
                "log_return": ret[i],
            })
    df = pd.DataFrame(rows)
    return _fill_advanced_feature_columns(df), returns, dates


def test_beta_vs_market_matches_direct_computation() -> None:
    """beta_1y = rolling_cov(ticker_return, market_return) / rolling_var(market_return),
    market_return = mean log_return of the OTHER tickers in the full universe
    per date -- self-EXCLUDED (2026-07-23 fix: a ticker's own return
    previously pulled its own benchmark toward itself, artificially shrinking
    its measured beta). With 3 tickers, ticker A's market series is the mean
    of B and C only (a DIFFERENT series per ticker), not one shared market
    series across all three. Checked against an independently-built reference
    market series and pandas rolling cov/var call -- not the same code path
    as the per-ticker groupby/index-alignment loop in the implementation, so
    this catches misalignment/windowing bugs a purely-internal check couldn't."""
    df, returns, dates = _beta_fixture()

    result = compute_cross_sectional_features(df)

    others = {"A": ("B", "C"), "B": ("A", "C"), "C": ("A", "B")}
    for t in ("A", "B", "C"):
        s = pd.Series(returns[t])
        o1, o2 = others[t]
        market = (pd.Series(returns[o1]) + pd.Series(returns[o2])) / 2
        expected = (
            s.rolling(252, min_periods=60).cov(market)
            / market.rolling(252, min_periods=60).var()
        ).to_numpy()

        actual = (
            result[result["ticker"] == t].sort_values("trade_date")["beta_1y"].to_numpy()
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-9, equal_nan=True)


def test_beta_nan_before_min_periods_then_no_lookahead() -> None:
    """Two properties in one: (1) beta_1y is NaN until BETA_MIN_PERIODS rows
    exist for a ticker -- an unstable 2-5 point covariance shouldn't be
    reported as a real beta; (2) once past that point, a row's beta must not
    depend on any later row (truncating the dataframe to an earlier date
    range must not change earlier rows' beta) -- the standard no-lookahead
    guard already applied to price_percentile_1y/volatility_*_percentile,
    now needed here too since this is the first rolling window inside the
    full-universe cross-sectional pass."""
    from src.build_dataset.cross_sectional import BETA_MIN_PERIODS

    df, returns, dates = _beta_fixture(n_days=80)

    full = compute_cross_sectional_features(df.copy())
    a_full = full[full["ticker"] == "A"].sort_values("trade_date").reset_index(drop=True)

    assert a_full.loc[: BETA_MIN_PERIODS - 2, "beta_1y"].isna().all(), (
        "beta must be NaN before BETA_MIN_PERIODS rows are available"
    )
    assert not pd.isna(a_full.loc[BETA_MIN_PERIODS - 1, "beta_1y"]), (
        "beta must be defined from BETA_MIN_PERIODS rows on"
    )

    cutoff = dates[60]  # keep the first 61 dates only
    truncated_df = df[df["trade_date"] <= cutoff].copy()
    truncated = compute_cross_sectional_features(truncated_df)
    a_trunc = truncated[truncated["ticker"] == "A"].sort_values("trade_date").reset_index(drop=True)

    pd.testing.assert_series_equal(
        a_trunc["beta_1y"], a_full.loc[: len(a_trunc) - 1, "beta_1y"],
        check_names=False, obj="beta_1y differs between full and truncated date ranges",
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
