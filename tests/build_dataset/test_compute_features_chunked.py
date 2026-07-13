#!/usr/bin/env python3
"""
Integration test for compute_features_chunked's 3-pass batching (build_ml_dataset.py):
verifies the chunked output matches an unchunked, one-shot run of the same
pipeline stages (features.py + cross_sectional.py + clean.py).

Run from project root: python tests/build_dataset/test_compute_features_chunked.py
or: pytest tests/build_dataset/test_compute_features_chunked.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_dataset.build_ml_dataset import compute_features_chunked
from src.build_dataset.clean import clean_dataset
from src.build_dataset.cross_sectional import compute_cross_sectional_features
from src.build_dataset.features import (
    compute_dividend_features,
    compute_macro_features,
    compute_price_features,
    compute_advanced_features,
    recompute_valuation_daily,
)


def _chunked_pipeline_fixture(n_days: int = 260) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multi-ticker, multi-sector frame with every raw column the full feature
    pipeline (compute_price_features through compute_cross_sectional_features)
    needs. 6 tickers in ticker-appearance order T1..T6, sectors assigned
    [A,A,B,A,B,B] so that at chunk_size=2 SectorA's members (T1,T2,T4) land in
    two different batches ([T1,T2] and [T3,T4]) -- the exact boundary a broken
    per-batch cross-sectional computation would get wrong.
    n_days=260 so return_1m/3m/12m (rolling 21/63/252-day sums) are all
    non-NaN on the last date -- otherwise momentum_vs_market/sector would
    trivially match as NaN==NaN without exercising the actual logic.
    """
    tickers = ["T1", "T2", "T3", "T4", "T5", "T6"]
    sectors = {"T1": "SectorA", "T2": "SectorA", "T3": "SectorB",
               "T4": "SectorA", "T5": "SectorB", "T6": "SectorB"}
    drift = {"T1": 0.010, "T2": 0.006, "T3": -0.004, "T4": 0.014, "T5": 0.002, "T6": 0.008}
    pl = {"T1": 10.0, "T2": 12.0, "T3": 8.0, "T4": 15.0, "T5": 9.0, "T6": 11.0}
    pvp = {"T1": 1.5, "T2": 1.8, "T3": 1.2, "T4": 2.1, "T5": 1.3, "T6": 1.6}
    roe = {"T1": 0.10, "T2": 0.12, "T3": 0.08, "T4": 0.15, "T5": 0.09, "T6": 0.11}
    debt_equity = {"T1": 0.5, "T2": 0.6, "T3": 0.4, "T4": 0.7, "T5": 0.45, "T6": 0.55}

    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    rows = []
    for t in tickers:
        price = 100.0
        for d in dates:
            price *= (1 + drift[t])
            rows.append({
                "ticker": t, "sector": sectors[t],
                "trade_date": d, "reference_date": dates[0], "fundamentals_available_date": dates[0],
                "adj_close": price, "adj_high": price * 1.01, "adj_low": price * 0.99,
                "close": 100.0, "close_price": 100.0,
                "market_cap": 1000.0, "net_debt": 100.0,
                "pl": pl[t], "pvp": pvp[t], "roe": roe[t], "debt_equity": debt_equity[t],
                "net_margin": 0.1, "roa": 0.05,
                # normally set by merge_dividends() (called once, before chunking,
                # in main()) -- our fixture skips that step and feeds
                # compute_features_chunked directly, so supply it here instead.
                "div_value_recent": 0.5,
                "lpa": 1.0, "ebitda": 100.0, "shares_outstanding": 1000.0,
                "net_revenue": 500.0, "net_income": 50.0,
                "revenue_growth_yoy": 0.05, "earnings_growth_yoy": 0.03,
                "selic": 0.1, "ipca": 0.04,
            })
    dataset = pd.DataFrame(rows)
    dividends = pd.DataFrame({
        "ticker": pd.Series(dtype=str),
        "ex_date": pd.Series(dtype="datetime64[ns]"),
        "value_per_share": pd.Series(dtype=float),
    })
    return dataset, dividends


def test_chunked_matches_unchunked_cross_sectional(tmp_path) -> None:
    """Regression guard for the batching bug: compute_features_chunked splits
    the WITHIN-ticker feature functions into ticker batches, but sector/market
    -relative features (compute_cross_sectional_features) must run once on the
    full universe, not per batch. chunk_size=2 here deliberately splits
    SectorA's three tickers (T1, T2, T4) across two different batches -- if
    cross-sectional features were ever computed per-batch again, this would
    catch it: their sector stats would silently diverge from the unchunked
    reference computed directly on the whole dataset in one shot."""
    dataset, dividends = _chunked_pipeline_fixture()

    out_path = tmp_path / "chunked.parquet"
    compute_features_chunked(dataset.copy(), dividends, out_path, chunk_size=2)
    chunked = pd.read_parquet(out_path)

    reference = compute_price_features(dataset.copy())
    reference = compute_dividend_features(reference, dividends)
    reference = compute_macro_features(reference)
    reference = recompute_valuation_daily(reference)
    reference = compute_advanced_features(reference)
    reference = compute_cross_sectional_features(reference)
    reference = clean_dataset(reference)

    chunked = chunked.set_index(["ticker", "trade_date"]).sort_index()
    reference = reference.set_index(["ticker", "trade_date"]).sort_index()
    assert len(chunked) == len(reference)

    cross_cols = [
        "pl_zscore_sector", "pvp_zscore_sector", "roe_zscore_sector", "debt_equity_zscore_sector",
        "div_yield_sector_percentile",
        "momentum_vs_market_1m", "momentum_vs_market_3m", "momentum_vs_market_12m",
        "momentum_vs_sector_1m", "momentum_vs_sector_3m", "momentum_vs_sector_12m",
    ]
    # last date is where return_1m/3m/12m are all non-NaN (rolling windows filled)
    last_date = dataset["trade_date"].max()
    for col in cross_cols:
        pd.testing.assert_series_equal(
            chunked.xs(last_date, level="trade_date")[col],
            reference.xs(last_date, level="trade_date")[col],
            check_names=False,
            obj=f"{col} differs between chunked and unchunked pipelines",
        )
        # sanity: the column isn't trivially all-NaN on both sides (which would
        # pass the equality check above without actually exercising the logic)
        assert chunked.xs(last_date, level="trade_date")[col].notna().any(), (
            f"{col} is all-NaN on the check date — fixture isn't exercising this column"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
