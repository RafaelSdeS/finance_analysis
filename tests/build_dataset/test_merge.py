#!/usr/bin/env python3
"""
Merge operations: prices+fundamentals (filing-lag no-lookahead), company
info, and dividends. Mirrors src/build_dataset/merge.py.

Run from project root: python tests/build_dataset/test_merge.py
or: pytest tests/build_dataset/test_merge.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_dataset.merge import (
    merge_prices_and_fundamentals,
    merge_company_info,
    merge_dividends,
    STATUS_INFERENCE_WINDOW_DAYS,
)
from src.build_dataset.quality_filters import FILING_LAG_DAYS_QUARTERLY


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    """Approximate equality allowing for floating-point rounding."""
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) < tol


def test_merge_company_info_infers_status_from_price_recency() -> None:
    """Tickers absent from company_info (and with no sibling match) get status
    inferred from price recency: still trading near the dataset's last date is
    ATIVO, a ticker whose last trade is older than the window is CANCELADA."""
    max_date = pd.Timestamp("2026-07-10")
    old_last_trade = max_date - pd.Timedelta(days=STATUS_INFERENCE_WINDOW_DAYS + 1)

    df = pd.DataFrame({
        "ticker": ["RECENT3", "RECENT3", "OLD3", "OLD3"],
        "trade_date": [
            max_date - pd.Timedelta(days=30), max_date,
            old_last_trade - pd.Timedelta(days=10), old_last_trade,
        ],
    })
    company_info = pd.DataFrame({
        "ticker": ["OTHER3"],
        "cvm_code": ["1"],
        "status": ["ATIVO"],
    })

    result = merge_company_info(df, company_info)

    assert (result.loc[result["ticker"] == "RECENT3", "status"] == "ATIVO").all()
    assert (result.loc[result["ticker"] == "OLD3", "status"] == "CANCELADA").all()


def test_merge_dividends_flags_missing_data() -> None:
    """A ticker with no dividend rows at all gets has_dividends=0, distinct
    from a ticker with real (even old, out-of-window) dividend history — so
    downstream div_yield_12m==0 isn't ambiguous with 'never collected'."""
    dataset = pd.DataFrame({
        "ticker": ["HASDIV", "HASDIV", "NODATA", "NODATA"],
        "trade_date": pd.to_datetime(
            ["2026-01-01", "2026-06-01", "2026-01-01", "2026-06-01"]
        ),
    })
    dividends = pd.DataFrame({
        "ticker": ["HASDIV"],
        "ex_date": pd.to_datetime(["2020-01-01"]),
        "value_per_share": [1.0],
    })

    result = merge_dividends(dataset, dividends)

    assert (result.loc[result["ticker"] == "HASDIV", "has_dividends"] == 1).all()
    assert (result.loc[result["ticker"] == "NODATA", "has_dividends"] == 0).all()


def test_merge_applies_filing_lag() -> None:
    """merge_asof only picks up a fundamental once its statutory filing lag has elapsed
    (T31: reference_date is the fiscal quarter-end, not the real filing date)."""
    ref_date = pd.Timestamp("2026-03-31")
    available = ref_date + pd.Timedelta(days=FILING_LAG_DAYS_QUARTERLY)

    fundamentals = pd.DataFrame({
        "ticker": ["A"],
        "reference_date": [ref_date],
        "pl": [15.0],
    })
    prices = pd.DataFrame({
        "ticker": ["A", "A", "A"],
        "trade_date": [ref_date, available - pd.Timedelta(days=1), available],
        "close": [100.0, 100.0, 100.0],
    })

    result = merge_prices_and_fundamentals(prices, fundamentals)

    assert pd.isna(result.iloc[0]["pl"])          # trade_date == reference_date: not yet filed
    assert pd.isna(result.iloc[1]["pl"])           # one day before the lag elapses: still not filed
    assert approx(result.iloc[2]["pl"], 15.0)      # lag has elapsed: fundamental now visible


def test_merge_honors_actual_filing_date() -> None:
    """When fundamentals carry a real fundamentals_available_date (CVM DT_RECEB),
    merge_asof uses it instead of the statutory fallback — both directions:
    an early filer is visible before the statutory deadline, a late filer after."""
    ref_date = pd.Timestamp("2026-03-31")
    early, late = ref_date + pd.Timedelta(days=20), ref_date + pd.Timedelta(days=200)

    fundamentals = pd.DataFrame({
        "ticker": ["EARLY", "LATE"],
        "reference_date": [ref_date, ref_date],
        "fundamentals_available_date": [early, late],
        "pl": [10.0, 20.0],
    })
    prices = pd.DataFrame({
        "ticker": ["EARLY", "EARLY", "LATE", "LATE"],
        "trade_date": [early - pd.Timedelta(days=1), early,
                       ref_date + pd.Timedelta(days=45), late],
        "close": [100.0, 100.0, 100.0, 100.0],
    })

    result = merge_prices_and_fundamentals(prices, fundamentals).set_index(
        ["ticker", "trade_date"])

    assert pd.isna(result.loc[("EARLY", early - pd.Timedelta(days=1)), "pl"])
    assert approx(result.loc[("EARLY", early), "pl"], 10.0)   # visible at day 20 < statutory 45
    assert pd.isna(result.loc[("LATE", ref_date + pd.Timedelta(days=45)), "pl"])  # statutory day, not yet filed
    assert approx(result.loc[("LATE", late), "pl"], 20.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
