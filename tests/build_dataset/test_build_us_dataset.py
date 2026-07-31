#!/usr/bin/env python3
"""
build_us_dataset.py — SIC->sector mapping, company_info cik-collision guard,
US macro merge (no-lookahead + rate conversions), daily valuation ratios
computed fresh (not re-anchored), and the universe gate.

Run from project root: python tests/build_dataset/test_build_us_dataset.py
or: pytest tests/build_dataset/test_build_us_dataset.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_dataset import build_us_dataset as us  # noqa: E402
from src.build_dataset import cagr_handler as ch  # noqa: E402


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) < tol


# =============================================================================
# SIC -> SECTOR
# =============================================================================

def test_sic_to_sector_maps_known_divisions() -> None:
    sic = pd.Series([2086, 3571, 6021, 9999, np.nan, 99999])
    sector = us.sic_to_sector(sic)
    assert sector.iloc[0] == "Manufacturing"       # 2086: bottled/canned soft drinks (KO)
    assert sector.iloc[1] == "Manufacturing"       # 3571: electronic computers (AAPL)
    assert sector.iloc[2] == "Finance, Insurance, Real Estate"  # 6021: national commercial banks
    assert sector.iloc[3] == "Public Administration"
    assert pd.isna(sector.iloc[4])   # NaN SIC -> NaN sector
    assert pd.isna(sector.iloc[5])   # out-of-range SIC -> NaN, not a crash


# =============================================================================
# COMPANY INFO MERGE
# =============================================================================

def test_merge_company_info_us_does_not_collide_cik_columns() -> None:
    """dataset already carries `cik` from the per-filing fundamentals row
    (loaders.load_fundamentals's US branch); company_info.parquet has its own
    `cik` too. A naive df.merge(..., on='ticker') would suffix both as
    cik_x/cik_y -- regression guard for that."""
    dataset = pd.DataFrame({
        "ticker": ["AAPL", "AAPL"],
        "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "cik": ["0000320193", "0000320193"],
    })
    company_info = pd.DataFrame({
        "ticker": ["AAPL"],
        "cik": ["0000320193"],
        "sic": [3571],
        "sic_description": ["Electronic Computers"],
    })

    result = us.merge_company_info_us(dataset, company_info)

    assert "cik_x" not in result.columns
    assert "cik_y" not in result.columns
    assert (result["cik"] == "0000320193").all()
    assert (result["sector"] == "Manufacturing").all()


def test_merge_company_info_us_null_sic_gives_null_sector() -> None:
    dataset = pd.DataFrame({"ticker": ["ZZZZ"], "trade_date": pd.to_datetime(["2024-01-02"])})
    company_info = pd.DataFrame({
        "ticker": ["ZZZZ"], "cik": ["1"], "sic": [None], "sic_description": [None],
    })
    result = us.merge_company_info_us(dataset, company_info)
    assert result["sector"].isna().all()


# =============================================================================
# MACRO MERGE (NO LOOKAHEAD + RATE CONVERSION)
# =============================================================================

def test_merge_macro_us_cpi_availability_is_shifted_forward(tmp_path, monkeypatch) -> None:
    """CPI for calendar month M must not be visible before its real ~mid-M+1
    publication date -- an exact-month asof merge would leak up to ~45 days
    of future inflation into every day of month M (same class of bug BR's
    ipca shift guards against)."""
    monkeypatch.setattr(us, "US_MACRO_DIR", tmp_path)

    rf_dates = pd.bdate_range("2023-12-01", periods=80)
    pd.DataFrame({
        "reference_date": rf_dates, "risk_free_3m": [5.0] * 80,
    }).to_parquet(tmp_path / "risk_free_3m.parquet")

    # Oct/Nov/Dec/Jan cpi levels -- Nov's and Dec's MoM readings become
    # available on 2023-12-16 and 2024-01-16 respectively (reference month
    # start + 1 month + 15 days).
    pd.DataFrame({
        "reference_date": pd.to_datetime(["2023-10-01", "2023-11-01", "2023-12-01", "2024-01-01"]),
        "cpi_sa": [298.0, 299.0, 300.0, 301.0],
    }).to_parquet(tmp_path / "cpi_sa.parquet")

    dataset = pd.DataFrame({
        "ticker": ["AAPL"] * 3,
        "trade_date": pd.to_datetime(["2024-01-10", "2024-01-17", "2024-02-01"]),
    })

    result = us.merge_macro_us(dataset).set_index("trade_date")

    nov_reading = (299.0 - 298.0) / 298.0 * 100   # available 2023-12-16
    dec_reading = (300.0 - 299.0) / 299.0 * 100   # available 2024-01-16

    # Before Dec's reading goes public (2024-01-16), only Nov's is visible.
    assert approx(result.loc[pd.Timestamp("2024-01-10"), "ipca"], nov_reading)
    # From 2024-01-16 onward, Dec's reading is visible -- Jan's own reading
    # doesn't arrive until 2024-02-16, so both later dates still see Dec's.
    assert approx(result.loc[pd.Timestamp("2024-01-17"), "ipca"], dec_reading)
    assert approx(result.loc[pd.Timestamp("2024-02-01"), "ipca"], dec_reading)


def test_merge_macro_us_selic_is_daily_equivalent_of_annualized_rate(tmp_path, monkeypatch) -> None:
    """DTB3 is quoted as an annualized %; `selic` must come out as a genuine
    daily-compounding rate (same convention as BR's selic), not a raw /252 or
    the annualized figure passed through unconverted."""
    monkeypatch.setattr(us, "US_MACRO_DIR", tmp_path)

    dates = pd.bdate_range("2024-01-01", periods=5)
    pd.DataFrame({
        "reference_date": dates, "risk_free_3m": [5.0] * 5,
    }).to_parquet(tmp_path / "risk_free_3m.parquet")
    pd.DataFrame({
        "reference_date": dates[:1], "cpi_sa": [300.0],
    }).to_parquet(tmp_path / "cpi_sa.parquet")

    dataset = pd.DataFrame({"ticker": ["AAPL"], "trade_date": [dates[0]]})
    result = us.merge_macro_us(dataset)

    expected_daily = ((1 + 5.0 / 100) ** (1 / 252) - 1) * 100
    assert approx(result["selic"].iloc[0], expected_daily)
    # Annualizing it back via the same compounding convention recovers ~5%,
    # not ~5/252 (an under-conversion) or 5.0 unconverted (no conversion at all).
    reannualized = (1 + expected_daily / 100) ** 252 - 1
    assert approx(reannualized, 0.05, tol=1e-4)


def test_merge_macro_us_selic_trend_20d_no_leak_across_disjoint_tickers(tmp_path, monkeypatch) -> None:
    """Same leak class BR's merge_macro guards against: selic_trend_20d must
    come off the raw daily risk-free series (real trading-day grid), never
    leak one ticker's future into another's rows sharing no real calendar
    overlap."""
    monkeypatch.setattr(us, "US_MACRO_DIR", tmp_path)

    dates = pd.bdate_range("2024-01-01", periods=40)
    rf_vals = [5.0 + 0.01 * i for i in range(40)]
    pd.DataFrame({
        "reference_date": dates, "risk_free_3m": rf_vals,
    }).to_parquet(tmp_path / "risk_free_3m.parquet")
    pd.DataFrame({
        "reference_date": dates[:1], "cpi_sa": [300.0],
    }).to_parquet(tmp_path / "cpi_sa.parquet")

    dataset = pd.DataFrame({
        "ticker": ["A"] * 20 + ["B"] * 20,
        "trade_date": list(dates[:20]) + list(dates[20:]),
    })
    result = us.merge_macro_us(dataset).set_index(["ticker", "trade_date"])

    assert result.loc["A", "selic_trend_20d"].isna().all()
    assert pd.notna(result.loc[("B", dates[20]), "selic_trend_20d"])


# =============================================================================
# DAILY VALUATION RATIOS
# =============================================================================

def test_compute_valuation_daily_us_matches_hand_computed_ratios() -> None:
    df = pd.DataFrame({
        "close": [200.0],
        "shares_outstanding": [1_000_000.0],
        "net_income": [10_000_000.0],
        "equity": [50_000_000.0],
        "net_revenue": [40_000_000.0],
        "total_assets": [300_000_000.0],
        "ebit": [20_000_000.0],
        "net_debt": [5_000_000.0],
        "reference_date": pd.to_datetime(["2024-01-01"]),
    })

    result = us.compute_valuation_daily_us(df)

    expected_mcap = 200.0 * 1_000_000.0
    assert approx(result["market_cap"].iloc[0], expected_mcap)
    assert approx(result["pl"].iloc[0], expected_mcap / 10_000_000.0)
    assert approx(result["pvp"].iloc[0], expected_mcap / 50_000_000.0)
    assert approx(result["p_sr"].iloc[0], expected_mcap / 40_000_000.0)
    assert approx(result["p_assets"].iloc[0], expected_mcap / 300_000_000.0)
    assert approx(result["p_ebit"].iloc[0], expected_mcap / 20_000_000.0)
    assert approx(result["book_to_market"].iloc[0], 50_000_000.0 / expected_mcap)
    assert approx(result["ev_ebit"].iloc[0], (expected_mcap + 5_000_000.0) / 20_000_000.0)
    assert result["has_fundamentals"].iloc[0] == 1.0


def test_compute_valuation_daily_us_zero_denominator_is_nan_not_inf() -> None:
    df = pd.DataFrame({
        "close": [200.0], "shares_outstanding": [1_000_000.0],
        "net_income": [0.0], "equity": [0.0], "net_revenue": [0.0],
        "total_assets": [0.0], "ebit": [0.0], "net_debt": [0.0],
        "reference_date": pd.to_datetime(["2024-01-01"]),
    })
    result = us.compute_valuation_daily_us(df)
    for col in ("pl", "pvp", "p_sr", "p_assets", "p_ebit", "ev_ebit"):
        assert pd.isna(result[col].iloc[0])
    assert not np.isinf(result[["pl", "pvp", "p_sr", "p_assets", "p_ebit", "ev_ebit"]].to_numpy()).any()


# =============================================================================
# UNIVERSE GATE
# =============================================================================

def test_build_universe_gate_thresholds() -> None:
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.concat([
        # Qualifies: long history, liquid, real price.
        pd.DataFrame({"ticker": "GOOD", "trade_date": dates, "close": 50.0, "volume": 1_000_000}),
        # Fails dollar-volume floor: liquid enough in row count but a penny stock.
        pd.DataFrame({"ticker": "PENNY", "trade_date": dates, "close": 0.05, "volume": 1_000_000}),
        # Fails min-rows: too little history.
        pd.DataFrame({"ticker": "SHORT", "trade_date": dates[:10], "close": 50.0, "volume": 1_000_000}),
    ], ignore_index=True)

    gate = us.build_universe_gate(prices, min_rows=250, min_median_close=1.0,
                                   min_median_dollar_volume=1_000_000)

    assert gate == {"GOOD"}


def test_build_universe_gate_from_files_matches_in_memory_gate(tmp_path) -> None:
    """The per-file-scan gate (build_us_dataset.py's fix for §8.0 Failure 1 --
    load_prices() OOMs on the full 9,593-ticker universe before the in-memory
    gate ever runs) must decide exactly the same qualifying set as
    build_universe_gate on the same data, just without ever holding it all in
    memory at once. Same 3 fixtures as test_build_universe_gate_thresholds,
    written to real per-ticker parquet files this time."""
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    fixtures = {
        "GOOD": pd.DataFrame({"trade_date": dates, "close": 50.0, "volume": 1_000_000}),
        "PENNY": pd.DataFrame({"trade_date": dates, "close": 0.05, "volume": 1_000_000}),
        "SHORT": pd.DataFrame({"trade_date": dates[:10], "close": 50.0, "volume": 1_000_000}),
    }
    for ticker, df in fixtures.items():
        df.to_parquet(tmp_path / f"{ticker}.parquet")

    gate = us.build_universe_gate_from_files(
        tmp_path, min_rows=250, min_median_close=1.0, min_median_dollar_volume=1_000_000
    )

    assert gate == {"GOOD"}


# =============================================================================
# CAGR ANCHOR MONTH (US fiscal-year-end fix, docs/US_DATASET_BUILD_PLAN.md §8.1)
# =============================================================================

def _quarterly_income(start, periods=28, growth_per_year=0.05):
    """Steady growth_per_year%-a-year net_income/net_revenue on a quarterly
    grid starting at `start` -- CAGR over any 20-quarter (5y) window should
    read back as growth_per_year*100 wherever it's defined."""
    dates = pd.date_range(start=start, periods=periods, freq="3ME")
    quarterly_growth = (1 + growth_per_year) ** (1 / 4)
    values = 100.0 * quarterly_growth ** np.arange(periods)
    return pd.DataFrame({"reference_date": dates, "net_income": values, "net_revenue": values * 3})


def test_fill_cagr_columns_default_anchor_still_december():
    """Regression guard: fill_cagr_columns's default (anchor_month=12) must
    stay byte-identical to pre-fix BR behavior -- December-anchored, annual
    update, held constant through the following Q1-Q3."""
    df = _quarterly_income("2015-03-31")  # Mar/Jun/Sep/Dec cycle -> hits December
    result = ch.fill_cagr_columns(df)
    # December rows are at index 3, 7, 11, ...; the first with a full 20-quarter
    # lookback is index 23 (row 3 + 20) -- rows 20-22 (Mar/Jun/Sep) precede it.
    assert result["cagr_earnings_5y_final"].iloc[23:].notna().all()
    assert np.allclose(result["cagr_earnings_5y_final"].iloc[23:], 5.0, atol=0.5)
    # broadcast: the 3 non-December quarters after an anchor share its value
    assert result["cagr_earnings_5y_final"].iloc[23:27].nunique() == 1


def test_fill_cagr_columns_anchor_month_none_fixes_non_december_fye():
    """The bug: a company whose fiscal year never lands on December (e.g.
    Agilent, FYE Oct 31 -- confirmed 8.6% of real US tickers) got 100% NaN
    CAGR under the December-only anchor, since raw Bolsai-style cagr_*_5y is
    unpopulated for the whole US universe and the calculated fallback was
    the only source. anchor_month=None (what build_us_dataset.py now passes
    via fill_missing_cagr) fixes it."""
    df = _quarterly_income("2015-01-31")  # Jan/Apr/Jul/Oct cycle -> never December

    broken = ch.fill_cagr_columns(df)  # old default (anchor_month=12)
    assert broken["cagr_earnings_5y_final"].isna().all(), (
        "fixture should reproduce the bug under the December-only default"
    )

    fixed = ch.fill_cagr_columns(df, anchor_month=None)
    assert fixed["cagr_earnings_5y_final"].iloc[20:].notna().all()
    assert np.allclose(fixed["cagr_earnings_5y_final"].iloc[20:], 5.0, atol=0.5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
