"""Memory budgeting (src/build_dataset/memory.py) and the per-batch disk load
that the budget is only meaningful because of.

The point of these: a full-scale US build was OOM-killed four separate times
(docs/US_DATASET_BUILD_PLAN.md §8.0.1/§8.0.3, plus Pass 2 on 2026-08-16 and
2026-08-23), twice taking unrelated desktop applications with it. The fixes
are only load-bearing if the budget actually responds to the machine's state
and the per-batch load actually produces the same numbers as loading
everything at once.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_dataset import memory  # noqa: E402
from src.build_dataset.features import compute_fundamental_features, fill_missing_cagr  # noqa: E402
from src.build_dataset.loaders import load_fundamentals, load_prices  # noqa: E402


# =============================================================================
# BUDGET
# =============================================================================

def test_budget_leaves_a_reserve_for_the_rest_of_the_machine() -> None:
    """The whole requirement is "don't assume the build owns the machine" —
    the budget must be strictly less than what's free, by the reserve."""
    os.environ.pop(memory.ENV_BUDGET, None)
    os.environ[memory.ENV_RESERVE] = "3"
    try:
        assert memory.budget_gb() <= memory.available_gb() - 3 or memory.budget_gb() == 1.5
    finally:
        del os.environ[memory.ENV_RESERVE]


def test_budget_and_reserve_are_overridable() -> None:
    os.environ[memory.ENV_BUDGET] = "6.5"
    try:
        assert memory.budget_gb() == 6.5
    finally:
        del os.environ[memory.ENV_BUDGET]


def test_chunk_size_scales_with_budget_and_clamps() -> None:
    """A busier machine must get smaller batches, but never so small that the
    parquet row groups stop compressing (MIN_CHUNK)."""
    big = memory.chunk_size_for(memory.US_BYTES_PER_TICKER, budget=8.0)
    small = memory.chunk_size_for(memory.US_BYTES_PER_TICKER, budget=2.0)
    assert big > small
    assert memory.chunk_size_for(memory.US_BYTES_PER_TICKER, budget=0.01) == memory.MIN_CHUNK
    assert memory.chunk_size_for(memory.US_BYTES_PER_TICKER, budget=10_000) == memory.MAX_CHUNK


def test_available_gb_is_plausible() -> None:
    """MemAvailable, not MemFree — MemFree reads near-zero on any machine
    that's been up a while and would starve the build for no reason."""
    assert 0.05 < memory.available_gb() < 4096


# =============================================================================
# PER-BATCH LOADING == WHOLE-UNIVERSE LOADING
# =============================================================================

def _write_fundamentals(dir, tickers, dates):
    """One parquet per ticker, mirroring the real US layout: `end` rather than
    `reference_date` (loaders renames it), no `ticker` column (the filename is
    the ticker), and the line items compute_fundamental_features reads."""
    n = len(dates)
    # Raw line items plus the ratios src/data_collection/ratios.compute_ratios
    # already derives at collection time — compute_fundamental_features reads
    # both kinds, so a fixture with only raw items KeyErrors on net_debt/roe/…
    derived = ["net_debt", "debt_equity", "roe", "roa", "net_margin", "gross_margin",
               "current_ratio", "roic", "asset_turnover", "ebit_margin",
               "ebit_over_assets", "net_debt_equity", "lpa", "vpa"]
    for i, t in enumerate(tickers):
        cols = {
            "end": dates,
            "fundamentals_available_date": dates,
            "net_income": [10.0 + i + j for j in range(n)],
            "equity": [100.0 + i + 5 * j for j in range(n)],
            "net_revenue": [50.0 + i + 2 * j for j in range(n)],
            "total_assets": [200.0 + i + 8 * j for j in range(n)],
            "current_assets": [80.0 + i + j for j in range(n)],
            "current_liabilities": [40.0 + i + j for j in range(n)],
            "cash": [20.0 + i + j for j in range(n)],
            "total_debt": [60.0 + i + j for j in range(n)],
            "ebit": [15.0 + i + j for j in range(n)],
            "cashflow_ops": [18.0 + i + j for j in range(n)],
            "capex": [-5.0 - i for _ in range(n)],
            "shares_outstanding": [1000.0 + i for _ in range(n)],
            "cost_of_revenue": [30.0 + i + j for j in range(n)],
        }
        cols.update({c: [1.0 + i + 0.1 * j for j in range(n)] for c in derived})
        pd.DataFrame(cols).to_parquet(dir / f"{t}.parquet")


def test_per_batch_fundamentals_stages_match_whole_universe(tmp_path) -> None:
    """Pass 1 now reads and derives fundamentals one ticker-batch at a time
    instead of loading the universe up front (that up-front load was the
    build's 5.5GB memory FLOOR — every other fix bought headroom on top of a
    floor that never moved).

    That is only safe because compute_fundamental_features and
    fill_missing_cagr are strictly per-ticker groupby passes. This asserts the
    property directly rather than trusting it: batch-then-derive must equal
    derive-then-slice, exactly. If someone later adds a cross-ticker statistic
    to either function, this fails — which is the point, because the failure
    would otherwise be silent and would corrupt every affected column.
    """
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    dates = pd.to_datetime(["2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31"])
    _write_fundamentals(tmp_path, tickers, dates)

    def derive(f):
        return fill_missing_cagr(compute_fundamental_features(f, margin_col="net_margin"),
                                  anchor_month=None)

    whole = derive(load_fundamentals(dir=tmp_path, optimize_dtypes=True, quiet=True))
    batched = pd.concat(
        [derive(load_fundamentals(dir=tmp_path, tickers=set(b), optimize_dtypes=True, quiet=True))
         for b in (tickers[:2], tickers[2:])],
        ignore_index=True,
    )

    key = ["ticker", "reference_date"]
    cols = sorted(whole.columns.difference(key))
    pd.testing.assert_frame_equal(
        whole.set_index(key).sort_index()[cols],
        batched.set_index(key).sort_index()[cols],
    )


def test_load_fundamentals_tickers_filter_reads_only_that_batch(tmp_path) -> None:
    _write_fundamentals(tmp_path, ["AAA", "BBB", "CCC"], pd.to_datetime(
        ["2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31"]))
    f = load_fundamentals(dir=tmp_path, tickers={"AAA", "CCC"}, quiet=True)
    assert sorted(f["ticker"].unique()) == ["AAA", "CCC"]


def test_load_prices_columns_projection_matches_dense_read(tmp_path) -> None:
    """build_us_dataset resolves its final ticker list from a 2-column read
    (~0.36GB at US scale) instead of the dense ~5GB panel. The projection has
    to agree with the dense read on exactly what the coverage filter reads
    from it: the ticker set, per-ticker row counts, and last trade dates."""
    dates = pd.bdate_range("2022-01-03", periods=30)
    for i, t in enumerate(["AAA", "BBB"]):
        pd.DataFrame({
            "ticker": t, "trade_date": dates,
            "open": 1.0 + i, "high": 2.0 + i, "low": 0.5 + i,
            "close": 1.5 + i, "volume": 1000 + i,
        }).to_parquet(tmp_path / f"{t}.parquet")

    dense = load_prices(dir=tmp_path, quiet=True)
    thin = load_prices(dir=tmp_path, columns=["ticker", "trade_date"], quiet=True)

    assert list(thin.columns) == ["ticker", "trade_date"]
    assert sorted(thin["ticker"].unique()) == sorted(dense["ticker"].unique())
    pd.testing.assert_series_equal(
        thin.groupby("ticker")["trade_date"].max(),
        dense.groupby("ticker")["trade_date"].max(),
    )
    pd.testing.assert_series_equal(thin.groupby("ticker").size(), dense.groupby("ticker").size())


def test_fundamentals_ticker_index_skips_empty_files(tmp_path) -> None:
    """The coverage filter's second argument is now a filename index rather
    than the real ~0.5GB table. A file that exists but holds no rows must
    still read as uncovered, exactly as it would have under a real load."""
    from src.build_dataset.build_us_dataset import fundamentals_ticker_index

    dates = pd.to_datetime(["2022-03-31"])
    pd.DataFrame({"end": dates, "net_income": [1.0]}).to_parquet(tmp_path / "AAA.parquet")
    pd.DataFrame({"end": pd.to_datetime([]), "net_income": []}).to_parquet(
        tmp_path / "EMPTY.parquet")

    assert fundamentals_ticker_index(tmp_path)["ticker"].tolist() == ["AAA"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
