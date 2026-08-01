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
    compute_history_relative_features,
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
                "adj_open": price, "volume": 1_000_000.0, "traded_amount": 100_000_000.0,
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
                "cagr_earnings_5y_final": 5.0, "cagr_revenue_5y_final": 3.0,
            })
    dataset = pd.DataFrame(rows)
    dividends = pd.DataFrame({
        "ticker": pd.Series(dtype=str),
        "ex_date": pd.Series(dtype="datetime64[ns]"),
        "value_per_share": pd.Series(dtype=float),
    })
    return dataset, dividends


def _synthetic_benchmark(dates):
    """BOVA11 stand-in: compute_cross_sectional_features now reads the market
    series from this fixed external benchmark instead of the ticker panel
    itself (2026-07-24 audit, Issue 2) -- both the chunked and unchunked
    reference pipelines below must be fed the SAME one for the consistency
    check to mean anything."""
    dates = pd.DatetimeIndex(dates).unique()
    return pd.DataFrame({
        "trade_date": dates, "log_return": 0.0005,
        "return_1m": 0.01, "return_3m": 0.03, "return_12m": 0.06,
    })


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
    benchmark = _synthetic_benchmark(dataset["trade_date"].unique())

    out_path = tmp_path / "chunked.parquet"
    compute_features_chunked(dataset.copy(), dividends, benchmark, out_path, chunk_size=2)
    chunked = pd.read_parquet(out_path)

    reference = compute_price_features(dataset.copy())
    reference = compute_dividend_features(reference, dividends)
    reference = compute_macro_features(reference)
    reference = recompute_valuation_daily(reference)
    reference = compute_advanced_features(reference)
    reference = compute_history_relative_features(reference)
    reference = compute_cross_sectional_features(reference, benchmark)
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
    # beta_1y deliberately excluded from this equality check: this fixture's
    # prices are constant daily drift (no noise), so every ticker's log_return
    # is time-constant and the market series has exactly zero variance ->
    # beta_1y is 0/0 = NaN everywhere here, which would trip the "isn't
    # trivially all-NaN" sanity assertion below for a reason unrelated to
    # chunking correctness. beta_1y's own chunking-safety and correctness are
    # covered directly in test_cross_sectional.py instead.
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


def test_batch_fn_path_matches_dataset_slicing_path(tmp_path) -> None:
    """Regression guard for the `tickers`/`batch_fn` params added to fix the
    real US-scale OOM (docs/US_DATASET_BUILD_PLAN.md §8.0.1 follow-up):
    build_us_dataset.py now merges per-batch via `batch_fn` instead of
    slicing a pre-merged `dataset`. The two code paths must produce
    byte-identical output on the same data -- `batch_fn` here is a trivial
    dataset-slicer (same logic Pass 1 used before this change), so any
    divergence would mean the new plumbing itself (not the merge it wraps)
    broke something."""
    dataset, dividends = _chunked_pipeline_fixture()
    benchmark = _synthetic_benchmark(dataset["trade_date"].unique())

    old_path = tmp_path / "old.parquet"
    compute_features_chunked(dataset.copy(), dividends, benchmark, old_path, chunk_size=2)

    new_path = tmp_path / "new.parquet"
    tickers = list(dataset["ticker"].unique())
    compute_features_chunked(
        None, dividends, benchmark, new_path, chunk_size=2,
        tickers=tickers, batch_fn=lambda bt: dataset[dataset["ticker"].isin(bt)].copy(),
    )

    old = pd.read_parquet(old_path).set_index(["ticker", "trade_date"]).sort_index()
    new = pd.read_parquet(new_path).set_index(["ticker", "trade_date"]).sort_index()
    pd.testing.assert_frame_equal(old, new)


def test_batch_fn_release_actually_frees_captured_state(tmp_path) -> None:
    """Regression guard for a subtle reference-lifetime bug found building
    the fix for a real 3rd US-scale OOM (docs/US_DATASET_BUILD_PLAN.md
    §8.0.2 follow-up): a plain `del batch_fn` inside compute_features_chunked
    does NOT free anything batch_fn captured, because whatever CALLED this
    function (e.g. build_us_dataset.main()) keeps its OWN reference to the
    same batch_fn object bound in its frame for this entire synchronous
    call -- confirmed via a minimal repro before landing the fix. Only a
    batch_fn that MUTATES its own state via release() (build_us_dataset's
    _MergeBatcher) actually gets freed, regardless of how many outer scopes
    still hold a reference to the batch_fn object itself. This test
    deliberately keeps its own `releasable` reference alive through the call
    (mirroring what main() does) -- if compute_features_chunked stopped
    calling release(), or someone "simplified" it back to `del batch_fn`,
    this would catch it."""
    import weakref

    dataset, dividends = _chunked_pipeline_fixture()
    benchmark = _synthetic_benchmark(dataset["trade_date"].unique())
    out_path = tmp_path / "out.parquet"

    class Releasable:
        def __init__(self, dataset, captured):
            self._dataset = dataset
            self._captured = captured  # stands in for prices/fundamentals/company_info

        def __call__(self, batch_tickers):
            return self._dataset[self._dataset["ticker"].isin(batch_tickers)].copy()

        def release(self):
            self._dataset = None
            self._captured = None

    captured = pd.DataFrame({"x": range(10)})  # a unique object, referenced nowhere else
    ref = weakref.ref(captured)
    releasable = Releasable(dataset, captured)
    del captured  # this test's own direct reference gone; releasable._captured still holds it

    tickers = list(dataset["ticker"].unique())
    # `releasable` stays bound HERE for the whole call below, exactly like
    # build_us_dataset.main()'s own batch_fn variable -- that's the entire
    # point of the test.
    compute_features_chunked(
        None, dividends, benchmark, out_path, chunk_size=2,
        tickers=tickers, batch_fn=releasable,
    )

    assert ref() is None, (
        "batch_fn.release() should have freed its captured state before/during "
        "Pass 2 -- compute_features_chunked may have stopped calling release(), "
        "or reverted to a plain `del batch_fn` (which a live caller-side "
        "reference always defeats)"
    )


def test_chunked_pipeline_stays_within_coarse_memory_and_time_ceiling(tmp_path) -> None:
    """Tripwire, not a precise benchmark. Memory is an explicit, recurring
    design concern throughout this pipeline (chunk_size exists specifically
    to avoid OOM; several helpers elsewhere are written the way they are
    only to avoid materializing full-width copies -- see numeric_columns()
    in test_utils.py) -- yet nothing asserted a ceiling. A future change that
    silently reintroduces an O(n^2) memory/time pattern should fail this
    test long before it OOMs a real build."""
    import resource
    import time

    dataset, dividends = _chunked_pipeline_fixture(n_days=500)
    benchmark = _synthetic_benchmark(dataset["trade_date"].unique())
    out_path = tmp_path / "chunked.parquet"

    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
    start = time.monotonic()
    compute_features_chunked(dataset.copy(), dividends, benchmark, out_path, chunk_size=2)
    elapsed = time.monotonic() - start
    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    assert elapsed < 30.0, (
        f"chunked pipeline took {elapsed:.1f}s on a tiny 6-ticker/500-day fixture "
        f"-- investigate a runtime regression"
    )
    assert (mem_after - mem_before) < 500_000, (  # 500 MB, deliberately generous
        f"chunked pipeline grew peak RSS by {(mem_after - mem_before) / 1024:.0f} MB on a "
        f"tiny synthetic fixture -- investigate a memory regression (full-frame copy, "
        f"chunking bypassed, etc.)"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
