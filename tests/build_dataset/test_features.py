#!/usr/bin/env python3
"""
Per-ticker feature engineering: price technicals, fundamental ratios,
valuation re-anchoring, and the "advanced" contextual features.

Validates that feature formulas (RSI, MAs, volatility, drawdown, returns,
etc.) compute correct values from synthetic price data. Mirrors
src/build_dataset/features.py.

Run from project root: python tests/build_dataset/test_features.py
or: pytest tests/build_dataset/test_features.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_dataset.features import (
    compute_price_features,
    compute_fundamental_features,
    compute_advanced_features,
    recompute_valuation_daily,
)


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    """Approximate equality allowing for floating-point rounding."""
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) < tol


def test_log_return_basic() -> None:
    """Log returns: [100, 102, 101] → [NaN, log(1.02), log(101/102)]."""
    df = pd.DataFrame({
        "ticker": ["A"] * 3,
        "trade_date": pd.date_range("2026-01-01", periods=3),
        "adj_close": [100.0, 102.0, 101.0],
        "adj_high": [100.0, 102.0, 101.0],
        "adj_low": [100.0, 102.0, 101.0],
    })
    result = compute_price_features(df)

    assert pd.isna(result.iloc[0]["log_return"]), "first row: no prior price"
    assert approx(result.iloc[1]["log_return"], np.log(102.0 / 100.0), tol=1e-9)
    assert approx(result.iloc[2]["log_return"], np.log(101.0 / 102.0), tol=1e-9)


def test_moving_averages() -> None:
    """MA20/60: rolling mean of prices. First 19/59 rows should be NaN."""
    prices = [100.0 + i * 0.5 for i in range(100)]  # Linear increase
    df = pd.DataFrame({
        "ticker": ["A"] * 100,
        "trade_date": pd.date_range("2026-01-01", periods=100),
        "adj_close": prices,
        "adj_high": prices,
        "adj_low": prices,
    })
    result = compute_price_features(df)

    # First 19 rows (indices 0-18) should have NaN ma_20
    for i in range(19):
        assert pd.isna(result.iloc[i]["ma_20"]), f"row {i}: ma_20 should be NaN"

    # Row 19 (index 19) is first non-NaN: mean of indices 0-19 (20 values)
    expected_ma20_at_19 = np.mean(prices[0:20])
    assert approx(result.iloc[19]["ma_20"], expected_ma20_at_19, tol=1e-6)

    # Similarly for MA60: first 59 should be NaN
    for i in range(59):
        assert pd.isna(result.iloc[i]["ma_60"]), f"row {i}: ma_60 should be NaN"

    # Row 59 is first non-NaN MA60: mean of indices 0-59 (60 values)
    expected_ma60_at_59 = np.mean(prices[0:60])
    assert approx(result.iloc[59]["ma_60"], expected_ma60_at_59, tol=1e-6)


def test_volatility() -> None:
    """Volatility: std dev of log returns over window. Zero std when prices constant."""
    # Constant prices → zero returns → zero volatility
    df = pd.DataFrame({
        "ticker": ["A"] * 30,
        "trade_date": pd.date_range("2026-01-01", periods=30),
        "adj_close": [100.0] * 30,
        "adj_high": [100.0] * 30,
        "adj_low": [100.0] * 30,
    })
    result = compute_price_features(df)

    # After the first 20 returns (at index 20+), volatility_20d should be ~0
    assert result.iloc[25]["volatility_20d"] < 1e-10, "constant prices should give ~0 volatility"


def test_rsi_calculation() -> None:
    """RSI formula: 100 - (100 / (1 + avg_gain / avg_loss))."""
    # Simple test: alternating up/down → balanced RSI
    prices = [100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0] * 2
    df = pd.DataFrame({
        "ticker": ["A"] * len(prices),
        "trade_date": pd.date_range("2026-01-01", periods=len(prices)),
        "adj_close": prices,
        "adj_high": prices,
        "adj_low": prices,
    })
    result = compute_price_features(df)

    # After 14 periods, RSI with balanced gains/losses should be ~50
    rsi_14 = result.iloc[15]["rsi_14"]
    assert not pd.isna(rsi_14), "RSI should not be NaN at row 15"
    # Balanced up/down → RS = 1 → RSI = 100 - 100/2 = 50
    assert 40 < rsi_14 < 60, f"balanced alternating should give RSI ~50, got {rsi_14:.1f}"


def test_rsi_mixed_trend() -> None:
    """Mixed trend with both gains and losses → RSI should be valid (not NaN)."""
    # Create series with alternating ups and downs: +3%, -1%, +3%, -1%, ...
    prices = []
    price = 100.0
    for i in range(50):
        if i % 2 == 0:
            price *= 1.03  # up 3%
        else:
            price *= 0.99  # down 1%
        prices.append(price)

    df = pd.DataFrame({
        "ticker": ["A"] * len(prices),
        "trade_date": pd.date_range("2026-01-01", periods=len(prices)),
        "adj_close": prices,
        "adj_high": prices,
        "adj_low": prices,
    })
    result = compute_price_features(df)

    # At index 20+, should have valid RSI with both gains and losses
    rsi_14 = result.iloc[25]["rsi_14"]
    assert not pd.isna(rsi_14), f"RSI should be valid (not NaN) at index 25, got {rsi_14}"
    # Bigger ups than downs should give RSI moderately high (>50)
    assert 40 < rsi_14 < 90, f"mixed trend should give RSI 40-90, got {rsi_14:.1f}"


def test_rsi_downtrend() -> None:
    """Pure downtrend → RSI should be very low (near 0)."""
    prices = np.linspace(110.0, 100.0, 30)  # Monotonic decrease
    df = pd.DataFrame({
        "ticker": ["A"] * len(prices),
        "trade_date": pd.date_range("2026-01-01", periods=len(prices)),
        "adj_close": prices,
        "adj_high": prices,
        "adj_low": prices,
    })
    result = compute_price_features(df)

    # After 14 periods, pure downtrend should give low RSI (<30)
    rsi_14 = result.iloc[15]["rsi_14"]
    assert rsi_14 < 30, f"downtrend should give RSI <30, got {rsi_14:.1f}"


def test_rsi_no_down_days() -> None:
    """Pure uptrend (zero down-days in the window) → RSI = 100, not NaN.

    loss=0 makes gain/loss a division by zero; RSI is still well-defined (100),
    it just can't be reached by the plain division formula.
    """
    prices = np.linspace(100.0, 110.0, 30)  # Monotonic increase, no down days
    df = pd.DataFrame({
        "ticker": ["A"] * len(prices),
        "trade_date": pd.date_range("2026-01-01", periods=len(prices)),
        "adj_close": prices,
        "adj_high": prices,
        "adj_low": prices,
    })
    result = compute_price_features(df)

    rsi_14 = result.iloc[15]["rsi_14"]
    assert not pd.isna(rsi_14), "RSI should not be NaN for a pure uptrend"
    assert approx(rsi_14, 100.0), f"pure uptrend should give RSI 100, got {rsi_14}"


def test_rsi_flat_prices() -> None:
    """Perfectly flat prices (zero gain, zero loss) → RSI = 50 (neutral), not NaN."""
    prices = [100.0] * 30
    df = pd.DataFrame({
        "ticker": ["A"] * len(prices),
        "trade_date": pd.date_range("2026-01-01", periods=len(prices)),
        "adj_close": prices,
        "adj_high": prices,
        "adj_low": prices,
    })
    result = compute_price_features(df)

    rsi_14 = result.iloc[15]["rsi_14"]
    assert not pd.isna(rsi_14), "RSI should not be NaN for flat prices"
    assert approx(rsi_14, 50.0), f"flat prices should give neutral RSI 50, got {rsi_14}"


def test_drawdown_calculation() -> None:
    """Drawdown: (price - running_max) / running_max. All-time high → 0, crash → negative."""
    prices = [100.0, 120.0, 110.0, 90.0, 95.0, 105.0]
    df = pd.DataFrame({
        "ticker": ["A"] * len(prices),
        "trade_date": pd.date_range("2026-01-01", periods=len(prices)),
        "adj_close": prices,
        "adj_high": prices,
        "adj_low": prices,
    })
    result = compute_price_features(df)

    # Row 0: no prior max, so (100 - 100) / 100 = 0
    assert approx(result.iloc[0]["drawdown"], 0.0)

    # Row 1: max is 120, price is 120 → (120 - 120) / 120 = 0
    assert approx(result.iloc[1]["drawdown"], 0.0)

    # Row 2: max is 120, price is 110 → (110 - 120) / 120 = -10/120 ≈ -0.0833
    assert approx(result.iloc[2]["drawdown"], -10.0 / 120.0, tol=1e-4)

    # Row 3: max is 120, price is 90 → (90 - 120) / 120 = -30/120 = -0.25
    assert approx(result.iloc[3]["drawdown"], -30.0 / 120.0, tol=1e-4)


def test_hl_ratio() -> None:
    """HL ratio: (high - low) / close."""
    df = pd.DataFrame({
        "ticker": ["A"] * 3,
        "trade_date": pd.date_range("2026-01-01", periods=3),
        "adj_close": [100.0, 102.0, 101.0],
        "adj_high": [105.0, 106.0, 104.0],
        "adj_low": [95.0, 98.0, 99.0],
    })
    result = compute_price_features(df)

    # Row 0: (105 - 95) / 100 = 0.1
    assert approx(result.iloc[0]["hl_ratio"], 0.1)

    # Row 1: (106 - 98) / 102 = 8/102 ≈ 0.0784
    assert approx(result.iloc[1]["hl_ratio"], 8.0 / 102.0, tol=1e-4)


def test_return_windows() -> None:
    """Cumulative log returns over 21/63/126/252 day windows."""
    # Constant daily return of log(1.01)
    daily_ret = np.log(1.01)
    prices = [100.0 * (1.01 ** i) for i in range(300)]
    df = pd.DataFrame({
        "ticker": ["A"] * len(prices),
        "trade_date": pd.date_range("2026-01-01", periods=len(prices)),
        "adj_close": prices,
        "adj_high": prices,
        "adj_low": prices,
    })
    result = compute_price_features(df)

    # After 21 days (row 21), return_1m should be ~21 * daily_ret
    ret_1m_at_21 = result.iloc[21]["return_1m"]
    expected = 21 * daily_ret
    assert approx(ret_1m_at_21, expected, tol=1e-4), f"1m return at row 21: expected {expected:.6f}, got {ret_1m_at_21:.6f}"

    # After 63 days (row 63), return_3m should be ~63 * daily_ret
    ret_3m_at_63 = result.iloc[63]["return_3m"]
    expected = 63 * daily_ret
    assert approx(ret_3m_at_63, expected, tol=1e-4)


def test_ticker_grouping_isolation() -> None:
    """Price features computed separately per ticker (no cross-contamination)."""
    df = pd.DataFrame({
        "ticker": ["A", "A", "B", "B"],
        "trade_date": pd.date_range("2026-01-01", periods=4),
        "adj_close": [100.0, 102.0, 50.0, 52.0],
        "adj_high": [100.0, 102.0, 50.0, 52.0],
        "adj_low": [100.0, 102.0, 50.0, 52.0],
    })
    result = compute_price_features(df)

    # Ticker A: [100, 102] → log_return [NaN, log(1.02)]
    assert pd.isna(result.iloc[0]["log_return"])
    assert approx(result.iloc[1]["log_return"], np.log(102.0 / 100.0))

    # Ticker B: [50, 52] → log_return [NaN, log(1.04)], NOT log(52/102)
    assert pd.isna(result.iloc[2]["log_return"])
    assert approx(result.iloc[3]["log_return"], np.log(52.0 / 50.0))


def test_non_positive_prices_masked() -> None:
    """Non-positive prices → NaN in log_return (no div-by-zero crashes)."""
    df = pd.DataFrame({
        "ticker": ["A", "A", "A", "A"],
        "trade_date": pd.date_range("2026-01-01", periods=4),
        "adj_close": [100.0, 0.0, -50.0, 101.0],
        "adj_high": [100.0, 1.0, 1.0, 101.0],
        "adj_low": [100.0, 0.0, 0.0, 101.0],
    })
    result = compute_price_features(df)

    # Row 1: price=0 → NaN after masking
    assert pd.isna(result.iloc[1]["log_return"])

    # Row 2: price=-50 → NaN after masking
    assert pd.isna(result.iloc[2]["log_return"])


def test_fundamental_features_ratios() -> None:
    """Fundamental derived ratios: book_to_market, earnings_yield, etc."""
    df = pd.DataFrame({
        "ticker": ["A"],
        "reference_date": pd.to_datetime(["2026-01-01"]),
        "market_cap": [1000.0],
        "equity": [500.0],
        "net_income": [100.0],
        "net_revenue": [1000.0],
        "cash": [200.0],
        "current_assets": [1200.0],
        "current_liabilities": [400.0],
        "total_assets": [2000.0],
        "total_debt": [600.0],
        "net_debt": [300.0],
        "gross_margin": [0.5],
        "net_margin": [0.1],
        "roe": [0.2],
        "roa": [0.05],
        "current_ratio": [1.5],
        "debt_equity": [0.6],
        "ebitda": [150.0],
    })
    result = compute_fundamental_features(df)

    # book_to_market = equity / market_cap = 500 / 1000 = 0.5
    assert approx(result.iloc[0]["book_to_market"], 0.5)

    # earnings_yield = net_income / market_cap = 100 / 1000 = 0.1
    assert approx(result.iloc[0]["earnings_yield"], 0.1)

    # cash_ratio = cash / current_liabilities = 200 / 400 = 0.5
    assert approx(result.iloc[0]["cash_ratio"], 0.5)

    # net_debt_to_assets = net_debt / total_assets = 300 / 2000 = 0.15
    assert approx(result.iloc[0]["net_debt_to_assets"], 0.15)

    # working_capital_ratio = (current_assets - current_liabilities) / total_assets = (1200 - 400) / 2000 = 0.4
    assert approx(result.iloc[0]["working_capital_ratio"], 0.4)


def test_recompute_valuation_daily_rescales_by_price_factor() -> None:
    """pl/pvp/market_cap scale by close/close_price; book_to_market by its inverse."""
    df = pd.DataFrame({
        "ticker": ["A"],
        "trade_date": pd.to_datetime(["2026-01-01"]),
        "reference_date": pd.to_datetime(["2026-01-01"]),
        "fundamentals_available_date": pd.to_datetime(["2026-01-01"]),
        "close": [110.0],
        "close_price": [100.0],
        "pl": [10.0],
        "pvp": [2.0],
        "market_cap": [1000.0],
        "net_debt": [200.0],
        "ev_ebit": [12.0],
        "book_to_market": [0.55],
    })
    result = recompute_valuation_daily(df)

    factor = 110.0 / 100.0
    assert approx(result.iloc[0]["pl"], 10.0 * factor)
    assert approx(result.iloc[0]["pvp"], 2.0 * factor)
    assert approx(result.iloc[0]["market_cap"], 1000.0 * factor)
    assert approx(result.iloc[0]["book_to_market"], 0.55 / factor)

    # ev_ebit rebuilt algebraically from the API's own EV, not scaled directly
    ev_api = 1000.0 + 200.0
    denom = ev_api / 12.0
    expected_ev_ebit = (1000.0 * factor + 200.0) / denom
    assert approx(result.iloc[0]["ev_ebit"], expected_ev_ebit)

    assert result.iloc[0]["has_fundamentals"] == 1.0
    assert "close_price" not in result.columns


def test_recompute_valuation_daily_no_fundamentals_flag() -> None:
    """Rows with no filing (reference_date NaT) get has_fundamentals=0."""
    df = pd.DataFrame({
        "ticker": ["A"],
        "trade_date": pd.to_datetime(["2026-01-01"]),
        "reference_date": pd.to_datetime([None]),
        "fundamentals_available_date": pd.to_datetime([None]),
        "close": [110.0],
        "close_price": [np.nan],
        "pl": [np.nan],
        "market_cap": [np.nan],
        "net_debt": [np.nan],
    })
    result = recompute_valuation_daily(df)

    assert result.iloc[0]["has_fundamentals"] == 0.0


def _advanced_features_fixture(n_rows: int) -> pd.DataFrame:
    """Minimal single-ticker frame with every column compute_advanced_features touches."""
    dates = pd.date_range("2026-01-01", periods=n_rows, freq="D")
    volatility = [0.1, 0.2, 0.05, 0.3, 0.15, 0.25][:n_rows]
    return pd.DataFrame({
        "ticker": ["A"] * n_rows,
        "sector": ["Tech"] * n_rows,
        "trade_date": dates,
        "reference_date": dates,
        "div_value_recent": [0.5] * n_rows,
        "lpa": [1.0] * n_rows,
        "ebitda": [100.0] * n_rows,
        "shares_outstanding": [1000.0] * n_rows,
        "net_revenue": [500.0] * n_rows,
        "net_income": [50.0] * n_rows,
        "revenue_growth_yoy": [0.05] * n_rows,
        "earnings_growth_yoy": [0.03] * n_rows,
        "volatility_20d": volatility,
        "volatility_60d": volatility,
        "adj_close": [100.0 + i for i in range(n_rows)],
        "pl": [10.0] * n_rows,
        "drawdown": [0.0] * n_rows,
        "pvp": [2.0] * n_rows,
        "roe": [0.15] * n_rows,
        "debt_equity": [0.5] * n_rows,
        "div_yield_12m": [0.03] * n_rows,
        "return_1m": [0.01] * n_rows,
        "return_3m": [0.02] * n_rows,
        "return_12m": [0.05] * n_rows,
        "net_margin": [0.1] * n_rows,
        "roa": [0.05] * n_rows,
        "selic": [0.1] * n_rows,
        "cagr_earnings_5y_final": [5.0] * n_rows,
        "cagr_revenue_5y_final": [3.0] * n_rows,
    })


def test_volatility_percentile_no_lookahead() -> None:
    """volatility_20d_percentile at row i must not depend on rows after i (T1 regression guard:
    a plain .rank(pct=True) here would rank each row against the ticker's future volatility too)."""
    df = _advanced_features_fixture(6)

    full = compute_advanced_features(df.copy())
    truncated = compute_advanced_features(df.iloc[:3].copy())

    for i in range(3):
        assert approx(
            full.iloc[i]["volatility_20d_percentile"],
            truncated.iloc[i]["volatility_20d_percentile"],
        )
        assert approx(
            full.iloc[i]["volatility_60d_percentile"],
            truncated.iloc[i]["volatility_60d_percentile"],
        )


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


def test_trend_4q_uses_real_quarters_not_daily_rows() -> None:
    """roe_trend_4q (and friends) must diff over 4 fiscal quarters, not 4 rows
    of the daily panel. Fundamentals are forward-filled for ~60 trading days
    between filings, so diffing raw daily rows is 0 almost every day with a
    spurious blip for the few rows right after a new filing lands."""
    quarter_ends = pd.to_datetime(
        ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"]
    )
    roe_by_quarter = [8.0, 9.0, 9.5, 10.0, 12.0, 15.0]

    rows = []
    for qe, roe in zip(quarter_ends, roe_by_quarter):
        for d in pd.date_range(qe, periods=60):  # forward-filled daily rows within the quarter
            rows.append({"trade_date": d, "reference_date": qe, "roe": roe})
    df = pd.DataFrame(rows)
    df["ticker"] = "A"
    df["sector"] = "Tech"
    df = _fill_advanced_feature_columns(df)

    result = compute_advanced_features(df)

    # First 4 quarters have no filing 4 quarters back yet -> NaN, not 0
    for qe in quarter_ends[:4]:
        block = result.loc[result["reference_date"] == qe, "roe_trend_4q"]
        assert block.isna().all(), f"{qe}: expected NaN (no quarter 4 back yet)"

    # 5th quarter (2024-03-31): 12.0 - 8.0 = 4.0, constant across every daily
    # row of the quarter -- not just a 4-row blip right after the filing
    block5 = result.loc[result["reference_date"] == quarter_ends[4], "roe_trend_4q"]
    assert (block5 == 4.0).all()

    # 6th quarter (2024-06-30): 15.0 - 9.0 = 6.0
    block6 = result.loc[result["reference_date"] == quarter_ends[5], "roe_trend_4q"]
    assert (block6 == 6.0).all()

    # Regression guard against the old daily-row bug: every row within a
    # quarter must share the exact same trend value (dropna=False so an
    # all-NaN quarter also counts as "consistent").
    per_quarter_values = result.groupby("reference_date")["roe_trend_4q"].nunique(dropna=False)
    assert (per_quarter_values <= 1).all(), "trend value must be constant within a quarter"


def test_n_quarters_available_counts_real_filings() -> None:
    """n_quarters_available: cumulative count of distinct reference_date (quarterly filings)
    per ticker, expanding/non-decreasing, 0 before first filing."""
    quarter_ends = pd.to_datetime(
        ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"]
    )
    rows = []
    for qe in quarter_ends:
        for d in pd.date_range(qe, periods=60):
            rows.append({
                "ticker": "A", "trade_date": d, "reference_date": qe,
                "div_value_recent": 0.5, "lpa": 1.0, "ebitda": 100.0,
                "shares_outstanding": 1000.0, "net_revenue": 500.0, "net_income": 50.0,
                "revenue_growth_yoy": 0.05, "earnings_growth_yoy": 0.03,
                "volatility_20d": 0.1, "volatility_60d": 0.1, "adj_close": 100.0,
                "pl": 10.0, "drawdown": 0.0, "pvp": 2.0, "roe": 0.15, "debt_equity": 0.5,
                "div_yield_12m": 0.03, "return_1m": 0.01, "return_3m": 0.02,
                "return_12m": 0.05, "net_margin": 0.1, "roa": 0.05, "selic": 0.1,
                "cagr_earnings_5y_final": 5.0, "cagr_revenue_5y_final": 3.0,
            })
    df = pd.DataFrame(rows)

    result = compute_advanced_features(df)

    # Within each quarter block (reference_date), n_quarters_available must be constant
    for i, qe in enumerate(quarter_ends):
        block = result.loc[result["reference_date"] == qe, "n_quarters_available"]
        assert (block == i + 1).all(), f"quarter {i} should have count {i+1}"

    # Non-decreasing within ticker
    assert (result.sort_values("trade_date").groupby("ticker")["n_quarters_available"].diff().fillna(0) >= 0).all()

    # No NaN
    assert result["n_quarters_available"].notna().all()


def test_n_quarters_available_separate_tickers() -> None:
    """n_quarters_available counts per ticker independently (no bleed)."""
    rows = []
    for ticker in ("A", "B"):
        for qe in pd.to_datetime(["2023-03-31", "2023-06-30"]):
            for d in pd.date_range(qe, periods=20):
                rows.append({
                    "ticker": ticker, "trade_date": d, "reference_date": qe,
                    "div_value_recent": 0.5, "lpa": 1.0, "ebitda": 100.0,
                    "shares_outstanding": 1000.0, "net_revenue": 500.0, "net_income": 50.0,
                    "revenue_growth_yoy": 0.05, "earnings_growth_yoy": 0.03,
                    "volatility_20d": 0.1, "volatility_60d": 0.1, "adj_close": 100.0,
                    "pl": 10.0, "drawdown": 0.0, "pvp": 2.0, "roe": 0.15, "debt_equity": 0.5,
                    "div_yield_12m": 0.03, "return_1m": 0.01, "return_3m": 0.02,
                    "return_12m": 0.05, "net_margin": 0.1, "roa": 0.05, "selic": 0.1,
                    "cagr_earnings_5y_final": 5.0, "cagr_revenue_5y_final": 3.0,
                })
    df = pd.DataFrame(rows)

    result = compute_advanced_features(df)

    # Both tickers should have count 1 in their first quarter, count 2 in second
    a1 = result[(result["ticker"] == "A") & (result["reference_date"] == pd.Timestamp("2023-03-31"))]["n_quarters_available"]
    b1 = result[(result["ticker"] == "B") & (result["reference_date"] == pd.Timestamp("2023-03-31"))]["n_quarters_available"]
    assert (a1 == 1).all()
    assert (b1 == 1).all()

    a2 = result[(result["ticker"] == "A") & (result["reference_date"] == pd.Timestamp("2023-06-30"))]["n_quarters_available"]
    b2 = result[(result["ticker"] == "B") & (result["reference_date"] == pd.Timestamp("2023-06-30"))]["n_quarters_available"]
    assert (a2 == 2).all()
    assert (b2 == 2).all()


def test_cagr_defined_flags() -> None:
    """cagr_earnings_defined and cagr_revenue_defined equal notna() of their *_final columns."""
    df = _advanced_features_fixture(6)
    df["cagr_earnings_5y_final"] = [np.nan, 5.0, np.nan, 3.0, np.nan, 2.0]
    df["cagr_revenue_5y_final"] = [1.0, np.nan, 3.0, np.nan, 5.0, np.nan]

    result = compute_advanced_features(df)

    assert (result["cagr_earnings_defined"] == df["cagr_earnings_5y_final"].notna().astype(float)).all()
    assert (result["cagr_revenue_defined"] == df["cagr_revenue_5y_final"].notna().astype(float)).all()
    assert result["cagr_earnings_defined"].isin([0, 1]).all()
    assert result["cagr_revenue_defined"].isin([0, 1]).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
