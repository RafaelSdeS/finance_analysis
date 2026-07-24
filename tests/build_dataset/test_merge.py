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

from src.build_dataset import merge
from src.build_dataset.merge import (
    merge_prices_and_fundamentals,
    merge_company_info,
    merge_macro,
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
        "ticker": ["A", "A", "A", "A"],
        "trade_date": [ref_date, available - pd.Timedelta(days=1), available,
                       available + pd.Timedelta(days=1)],
        "close": [100.0, 100.0, 100.0, 100.0],
    })

    result = merge_prices_and_fundamentals(prices, fundamentals)

    assert pd.isna(result.iloc[0]["pl"])          # trade_date == reference_date: not yet filed
    assert pd.isna(result.iloc[1]["pl"])           # one day before the lag elapses: still not filed
    # exact filing date itself: not yet visible (allow_exact_matches=False --
    # a same-day filing can't have been seen before that day's close, 2026-07-23 audit)
    assert pd.isna(result.iloc[2]["pl"])
    assert approx(result.iloc[3]["pl"], 15.0)      # the day AFTER filing: now visible


def test_merge_honors_actual_filing_date() -> None:
    """When fundamentals carry a real fundamentals_available_date (CVM DT_RECEB),
    merge_asof uses it instead of the statutory fallback — both directions:
    an early filer is visible before the statutory deadline, a late filer after.
    Visibility starts the day AFTER the filing date, not the filing date itself
    (allow_exact_matches=False, 2026-07-23 audit: a same-day filing can't have
    been seen before that day's close)."""
    ref_date = pd.Timestamp("2026-03-31")
    early, late = ref_date + pd.Timedelta(days=20), ref_date + pd.Timedelta(days=200)

    fundamentals = pd.DataFrame({
        "ticker": ["EARLY", "LATE"],
        "reference_date": [ref_date, ref_date],
        "fundamentals_available_date": [early, late],
        "pl": [10.0, 20.0],
    })
    prices = pd.DataFrame({
        "ticker": ["EARLY", "EARLY", "EARLY", "LATE", "LATE", "LATE"],
        "trade_date": [early - pd.Timedelta(days=1), early, early + pd.Timedelta(days=1),
                       ref_date + pd.Timedelta(days=45), late, late + pd.Timedelta(days=1)],
        "close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    })

    result = merge_prices_and_fundamentals(prices, fundamentals).set_index(
        ["ticker", "trade_date"])

    assert pd.isna(result.loc[("EARLY", early - pd.Timedelta(days=1)), "pl"])
    assert pd.isna(result.loc[("EARLY", early), "pl"])                          # exact filing date: not yet
    assert approx(result.loc[("EARLY", early + pd.Timedelta(days=1)), "pl"], 10.0)  # day after: visible, day 21 < statutory 45
    assert pd.isna(result.loc[("LATE", ref_date + pd.Timedelta(days=45)), "pl"])  # statutory day, not yet filed
    assert pd.isna(result.loc[("LATE", late), "pl"])                            # exact late filing date: not yet
    assert approx(result.loc[("LATE", late + pd.Timedelta(days=1)), "pl"], 20.0)  # day after: visible


def test_merge_macro_aligns_by_date_no_lookahead(tmp_path, monkeypatch) -> None:
    """merge_macro is ticker-independent (one calendar-date series applies to
    every ticker) and must never look ahead: a trade_date before the series'
    first published date gets NaN, and a trade_date between two publications
    holds the most recent one, never the next (no lookahead) and never stale
    beyond the next update (forward-fill applied, not left at the old value
    forever)."""
    monkeypatch.setattr(merge, "MACRO_DIR", tmp_path)

    pd.DataFrame({"reference_date": pd.to_datetime(["2026-01-01", "2026-01-08"]),
                  "selic": [0.10, 0.11]}).to_parquet(tmp_path / "selic.parquet")
    pd.DataFrame({"reference_date": pd.to_datetime(["2026-01-02"]),
                  "cdi": [0.09]}).to_parquet(tmp_path / "cdi.parquet")
    pd.DataFrame({"reference_date": pd.to_datetime(["2026-01-01"]),
                  "ipca": [0.005]}).to_parquet(tmp_path / "ipca.parquet")

    dataset = pd.DataFrame({
        "ticker": ["A", "A", "A", "A", "B", "B"],
        "trade_date": pd.to_datetime(
            ["2025-12-31", "2026-01-01", "2026-01-05", "2026-01-08",
             "2025-12-31", "2026-01-05"]
        ),
    })

    result = merge_macro(dataset).set_index(["ticker", "trade_date"])

    # Before the series' first publication: no lookahead into the future value
    assert pd.isna(result.loc[("A", pd.Timestamp("2025-12-31")), "selic"])
    assert pd.isna(result.loc[("B", pd.Timestamp("2025-12-31")), "selic"])
    # Exact publication date
    assert approx(result.loc[("A", pd.Timestamp("2026-01-01")), "selic"], 0.10)
    # Between two publications: holds the earlier one, not the later
    assert approx(result.loc[("A", pd.Timestamp("2026-01-05")), "selic"], 0.10)
    # On/after the second publication: updated
    assert approx(result.loc[("A", pd.Timestamp("2026-01-08")), "selic"], 0.11)
    # Same date, independent ticker "B": identical macro value (ticker-independent)
    assert approx(result.loc[("B", pd.Timestamp("2026-01-05")), "selic"], 0.10)

    # cdi merged independently, correct column name, no cross-contamination
    assert pd.isna(result.loc[("A", pd.Timestamp("2025-12-31")), "cdi"])
    assert approx(result.loc[("A", pd.Timestamp("2026-01-05")), "cdi"], 0.09)

    # ipca must NOT be visible on its own reference_date (2026-01-01) or even
    # by 2026-01-08 -- IBGE doesn't actually publish that reading until
    # ~mid-February. Every trade_date in this fixture predates the real
    # release, so ipca must read NaN throughout (would have read 0.005 with
    # the pre-fix reference_date-direct merge -- that was the lookahead bug).
    assert pd.isna(result.loc[("A", pd.Timestamp("2026-01-01")), "ipca"])
    assert pd.isna(result.loc[("A", pd.Timestamp("2026-01-08")), "ipca"])


def test_merge_macro_ipca_visible_only_after_publication_lag(tmp_path, monkeypatch) -> None:
    """ipca's reference_date (month start, per SGS convention) must not be used
    directly as its availability date -- IBGE publishes month M's reading
    around day 8-11 of month M+1, so using reference_date leaks ~40 days of
    future inflation into every day of month M (2026-07-23 audit finding).
    merge_macro instead shifts to reference_date + 1 month + IPCA_PUBLICATION_LAG_DAYS."""
    monkeypatch.setattr(merge, "MACRO_DIR", tmp_path)

    pd.DataFrame({"reference_date": pd.to_datetime(["2026-01-01"]),
                  "selic": [0.10]}).to_parquet(tmp_path / "selic.parquet")
    pd.DataFrame({"reference_date": pd.to_datetime(["2026-01-01"]),
                  "cdi": [0.09]}).to_parquet(tmp_path / "cdi.parquet")
    pd.DataFrame({"reference_date": pd.to_datetime(["2026-01-01"]),
                  "ipca": [0.005]}).to_parquet(tmp_path / "ipca.parquet")

    available_date = (
        pd.Timestamp("2026-01-01") + pd.DateOffset(months=1)
        + pd.Timedelta(days=merge.IPCA_PUBLICATION_LAG_DAYS)
    )
    dataset = pd.DataFrame({
        "ticker": ["A"] * 3,
        "trade_date": [
            available_date - pd.Timedelta(days=1),
            available_date,
            available_date + pd.Timedelta(days=1),
        ],
    })

    result = merge_macro(dataset).set_index(["ticker", "trade_date"])

    assert pd.isna(result.loc[("A", available_date - pd.Timedelta(days=1)), "ipca"])
    assert approx(result.loc[("A", available_date), "ipca"], 0.005)
    assert approx(result.loc[("A", available_date + pd.Timedelta(days=1)), "ipca"], 0.005)


def test_merge_macro_ipca_daily_equiv_same_footing_as_selic_cdi(tmp_path, monkeypatch) -> None:
    """ipca_daily_equiv converts ipca's native MONTHLY rate to a daily rate on
    the same percent-per-trading-day footing as selic/cdi -- so a future
    consumer has an obviously-safe column to read instead of repeating the
    exact unit-mismatch bug that caused a Critical audit finding (raw ipca
    silently treated as if it were daily). Geometric decompounding: 21
    trading days of ipca_daily_equiv must compound back to the original
    monthly factor, and the log1p of it must exactly match
    compute_macro_features' real_return formula (log1p(ipca/100)/21) --
    same math, just factored differently, so the two never drift apart."""
    monkeypatch.setattr(merge, "MACRO_DIR", tmp_path)

    pd.DataFrame({"reference_date": pd.date_range("2026-01-01", periods=60, freq="B"),
                  "selic": [0.05] * 60}).to_parquet(tmp_path / "selic.parquet")
    pd.DataFrame({"reference_date": pd.to_datetime(["2026-01-01"]),
                  "cdi": [0.04]}).to_parquet(tmp_path / "cdi.parquet")
    pd.DataFrame({"reference_date": pd.to_datetime(["2026-01-01"]),
                  "ipca": [0.62]}).to_parquet(tmp_path / "ipca.parquet")

    dataset = pd.DataFrame({
        "ticker": ["A"] * 60,
        "trade_date": pd.date_range("2026-01-01", periods=60, freq="B"),
    })
    result = merge_macro(dataset)

    row = result.dropna(subset=["ipca_daily_equiv"]).iloc[0]
    assert approx(row["ipca"], 0.62)

    # 21 compounded days of the daily-equivalent rate reconstructs the
    # original monthly factor
    reconstructed_monthly_factor = (1 + row["ipca_daily_equiv"] / 100) ** merge.TRADING_DAYS_PER_MONTH
    assert approx(reconstructed_monthly_factor, 1 + row["ipca"] / 100, tol=1e-9)

    # exact mathematical equivalence to real_return's existing formula
    import numpy as np
    assert approx(
        np.log1p(row["ipca_daily_equiv"] / 100),
        np.log1p(row["ipca"] / 100) / 21,
        tol=1e-12,
    )

    # NaN wherever raw ipca is NaN (before the publication-lag availability
    # date) -- never a value computed from a not-yet-visible reading
    still_unavailable = result[result["ipca"].isna()]
    assert still_unavailable["ipca_daily_equiv"].isna().all()


def test_merge_macro_selic_trend_no_ticker_boundary_leak(tmp_path, monkeypatch) -> None:
    """selic_trend_20d must be computed on the raw daily selic series (one row
    per real trading day, ticker-independent), not per ticker downstream --
    a per-ticker-batch computation was found to leak trailing selic values
    across ticker/batch boundaries however it was windowed (2026-07-23 audit
    + regression check: two tickers with entirely non-overlapping historical
    date ranges must each get their OWN correct trend, not one bleeding into
    the other's)."""
    monkeypatch.setattr(merge, "MACRO_DIR", tmp_path)

    dates = pd.bdate_range("2026-01-01", periods=40)
    selic_vals = [0.05 + 0.001 * i for i in range(40)]
    pd.DataFrame({"reference_date": dates, "selic": selic_vals}).to_parquet(tmp_path / "selic.parquet")
    pd.DataFrame({"reference_date": dates[:1], "cdi": [0.04]}).to_parquet(tmp_path / "cdi.parquet")
    pd.DataFrame({"reference_date": dates[:1], "ipca": [0.4]}).to_parquet(tmp_path / "ipca.parquet")

    # Ticker A only trades the first 20 days, ticker B only the last 20 --
    # entirely disjoint, the worst case for any per-batch/per-ticker leak.
    dataset = pd.DataFrame({
        "ticker": ["A"] * 20 + ["B"] * 20,
        "trade_date": list(dates[:20]) + list(dates[20:]),
    })

    result = merge_macro(dataset).set_index(["ticker", "trade_date"])

    # A's own first 20 days: never 20 real trading days of history behind
    # anything in this fixture -> NaN throughout, never B's future values.
    assert result.loc["A", "selic_trend_20d"].isna().all()

    # B's first row (dates[20]) is the dataset's 21st real trading day --
    # exactly one full 20-day window of history exists on the TRUE calendar,
    # regardless of which ticker owns which rows.
    expected = selic_vals[20] - selic_vals[0]
    assert approx(result.loc[("B", dates[20]), "selic_trend_20d"], expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
