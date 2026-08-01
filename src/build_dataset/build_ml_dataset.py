"""
build_ml_dataset.py
===================

Constrói um dataset final para Machine Learning unindo:

1. Prices (daily)
2. Fundamentals (quarterly)
3. Company info (static)

Resultado:
    Uma linha por:
        (ticker, trade_date)

Com:
    - preços diários
    - fundamentos mais recentes disponíveis
    - informações da empresa

Saída:
    data/processed/ml_dataset.parquet

Uso:
    python -m src.build_dataset.build_ml_dataset

Pipeline stages live in sibling modules (loaders, repair, continuity,
quality_filters, merge, features, cross_sectional, clean, manifest); this
file only orchestrates the call order and the memory-bounded feature pass.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .clean import clean_dataset
from .continuity import apply_ticker_continuity
from .cross_sectional import (
    BENCHMARK_COLS,
    CROSS_SECTIONAL_INPUT_COLS,
    CROSS_SECTIONAL_OUTPUT_COLS,
    compute_cross_sectional_features,
)
from .features import (
    compute_advanced_features,
    compute_dividend_features,
    compute_fundamental_features,
    compute_history_relative_features,
    compute_macro_features,
    compute_price_features,
    fill_missing_cagr,
    recompute_valuation_daily,
)
from .loaders import load_company_info, load_dividends, load_fundamentals, load_prices
from .manifest import sync_dataset_version, write_manifest, write_split_config
from .merge import merge_company_info, merge_dividends, merge_macro, merge_prices_and_fundamentals
from .paths import OUTPUT_PATH
from .quality_filters import (
    attach_filing_dates,
    drop_orphan_prefix_rows,
    filter_excessive_filing_lag,
    filter_tickers_with_no_fundamentals,
)
from .repair import repair_unadjusted_splits

# IBOV-proxy ETF, used as the true market benchmark for beta_1y/
# momentum_vs_market_* (cross_sectional.py) -- collected the same as every
# other ticker (data_collection.config.BENCHMARK_TICKERS) but excluded from
# the dataset's own rows since it has no fundamentals (an ETF, not an
# operating company).
BENCHMARK_TICKER = "BOVA11"


# =============================================================================
# FEATURE COMPUTATION
# =============================================================================

def compute_features_chunked(dataset, dividends, benchmark, output_path, chunk_size=150,
                              valuation_fn=recompute_valuation_daily, tickers=None, batch_fn=None):
    """Three-pass, memory-bounded feature computation.

    `valuation_fn`: the daily valuation-ratio step run in Pass 1, defaulting
    to BR's recompute_valuation_daily (rescales a filing-date vendor ratio by
    close/close_price). Injected rather than hardcoded so build_us_dataset.py
    can pass compute_valuation_daily_us instead -- US fundamentals carry no
    price-anchored ratio to rescale in the first place (measured 0% raw
    coverage on market_cap/pl/pvp/etc., docs/US_DATASET_BUILD_PLAN.md §4.4),
    so that function computes them fresh from the daily close instead.

    `tickers`/`batch_fn`: when `batch_fn` is given, Pass 1 calls
    `batch_fn(batch_tickers)` to produce each batch instead of slicing a
    pre-merged `dataset` (which can then be None). This exists because
    merge_prices_and_fundamentals's own output -- the daily panel with every
    fundamentals column forward-filled onto it -- is the actual OOM point at
    US scale (measured ~5.9GB RSS for 800/3,134 tickers, extrapolating past
    available RAM at full scale; the per-ticker-batch feature loop below was
    already memory-bounded, the merge that fed it wasn't). Doing the merge
    itself per-batch means the wide frame is never built for more than one
    chunk_size-ticker batch at a time. `tickers` supplies the batch list
    directly in this mode (no merged `dataset` to derive it from). Default
    (`batch_fn=None`) is byte-identical to the original in-memory path --
    BR's build_ml_dataset.py call is unaffected.

    A fully unchunked pass OOM'd in practice — the dataset's dense-numeric
    size looks like ~1.3-2GB, but clean_dataset's inf->NaN replace() makes a
    full transient copy of all ~123 numeric columns, and main() keeps
    prices/fundamentals/company_info resident throughout, so real peak usage
    is well above the naive estimate. Ticker-batching alone isn't a safe fix
    either: several features (see compute_cross_sectional_features) compare
    each stock to the full market on the same date, so computing them on a
    25-ticker batch silently compares against the wrong universe.

    Pass 1: within-ticker feature functions run per ticker-batch (bounded
      memory — never holds more than one batch of the wide frame) and stream
      to a temp parquet. Also accumulates a SLIM projection (11 narrow
      columns instead of ~130) which is cheap to hold in full.
    Pass 2: cross-sectional features computed once on the slim full-universe
      projection.
    Pass 3: stream the temp file back out batch by batch (one row group at a
      time), merge in the small cross-sectional result, clean, write final —
      keeping clean_dataset's memory bounded to one batch too.

    chunk_size also sets the row-group size of both parquet files (one batch
    = one row group): too small hurts compression badly (dictionary/RLE
    encoding resets every row group — 25 tickers/batch measured at ~4% size
    reduction vs. ~75% for a single row group), too large risks OOM again.
    150 tickers/batch gives ~4-5 row groups for the full universe.

    `benchmark`: BOVA11's own price-feature series (trade_date + log_return/
    return_1m/3m/12m), passed through unchanged into Pass 2's
    compute_cross_sectional_features -- see BENCHMARK_TICKER / that
    function's docstring.
    """
    tmp_path = output_path.with_suffix(".tmp.parquet")

    if tickers is None:
        tickers = dataset["ticker"].unique()
    batches = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

    print()
    print("=" * 80)
    print(f"PASS 1/3: PER-TICKER FEATURES IN {len(batches)} BATCHES (chunk_size={chunk_size})")
    print("=" * 80)

    slim_parts = []
    writer = None
    try:
        for batch_idx, batch_tickers in enumerate(batches, 1):
            if batch_fn is not None:
                batch = batch_fn(batch_tickers)
            else:
                batch = dataset[dataset["ticker"].isin(batch_tickers)].copy()
            print(f"Batch {batch_idx}/{len(batches)}: {len(batch_tickers)} tickers, {len(batch)} rows")

            batch = compute_price_features(batch)
            batch = compute_dividend_features(batch, dividends)
            batch = compute_macro_features(batch)
            batch = valuation_fn(batch)
            batch = compute_advanced_features(batch)
            batch = compute_history_relative_features(batch)

            slim_cols = [c for c in CROSS_SECTIONAL_INPUT_COLS if c in batch.columns]
            slim_parts.append(batch[slim_cols].copy())

            table = pa.Table.from_pandas(batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            else:
                # later batches can promote e.g. an all-NaN int column to float —
                # cast to the schema locked in by batch 1 so row groups stay uniform
                table = table.cast(writer.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    # Release batch_fn's captured state now, not just at function exit --
    # Pass 2/3 never call it again. NOTE: a plain `del batch_fn` here does
    # NOT free anything it closed over -- whatever called this function
    # (build_us_dataset.main()) keeps its OWN reference to the same batch_fn
    # object bound for this entire synchronous nested call (a caller's frame
    # persists for the full duration of a call it's blocked on), so deleting
    # only OUR copy of that reference never brings its refcount to 0
    # (confirmed via a minimal repro before landing this). release() instead
    # MUTATES the object's own attributes -- visible through every reference
    # to it, caller included -- which is what actually drops the raw
    # prices/fundamentals/company_info tables before Pass 2's full-universe
    # frame is built. Without this they stayed resident through Pass 2/3 too
    # (which don't need them at all) -- a real 3rd OOM in the US build
    # (docs/US_DATASET_BUILD_PLAN.md §8.0.2 follow-up) on top of the one §8.3
    # already fixed in Pass 2 itself. A plain callable (no `release`, e.g. a
    # test's bare lambda) is a no-op here.
    if batch_fn is not None:
        getattr(batch_fn, "release", lambda: None)()

    print()
    print("=" * 80)
    print("PASS 2/3: CROSS-SECTIONAL (MARKET/SECTOR) FEATURES")
    print("=" * 80)

    slim = pd.concat(slim_parts, ignore_index=True)
    del slim_parts
    slim = compute_cross_sectional_features(slim, benchmark)
    slim = slim[["ticker", "trade_date"] + CROSS_SECTIONAL_OUTPUT_COLS].set_index(
        ["ticker", "trade_date"]
    )

    print()
    print("=" * 80)
    print("PASS 3/3: MERGING CROSS-SECTIONAL FEATURES + CLEANING")
    print("=" * 80)

    pf = pq.ParquetFile(tmp_path)
    total_rows = 0
    writer = None
    try:
        for rg in range(pf.num_row_groups):
            batch = pf.read_row_group(rg).to_pandas()
            batch = batch.join(slim, on=["ticker", "trade_date"])
            batch = clean_dataset(batch)

            table = pa.Table.from_pandas(batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)

            total_rows += len(batch)
            print(f"Row group {rg + 1}/{pf.num_row_groups}: {len(batch)} rows (total {total_rows})")
    finally:
        if writer is not None:
            writer.close()

    tmp_path.unlink()
    print(f"Feature computation complete: {total_rows} rows")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    prices       = load_prices()
    fundamentals = load_fundamentals()
    # drop orphan-prefix rows BEFORE anything else: garbage from an unrelated
    # earlier holder of a recycled ticker code must never reach any logic downstream.
    prices       = drop_orphan_prefix_rows(prices)
    # repair BEFORE continuity: events are keyed under each entity's original
    # ticker name at the time of the split; repair them in place under those names
    # so the factor math runs on the pre-splice data. Continuity then renames both
    # the repaired old leg and its associated rows onto the new ticker.
    prices       = repair_unadjusted_splits(prices)
    # splice AFTER split repair: each leg is now internally continuous; splicing
    # them together preserves that invariant.
    prices, fundamentals = apply_ticker_continuity(prices, fundamentals)

    # Capture BOVA11 (market benchmark) BEFORE the fundamentals-coverage
    # filter drops it below -- it has none by design (an ETF, not an
    # operating company) but has already had the same split-repair/continuity
    # treatment as every other ticker above, so its own compute_price_features
    # run here is methodologically identical to the rest of the universe's.
    # Never becomes a row in the output dataset; threaded through purely as
    # an external reference series for cross_sectional.py's beta_1y/
    # momentum_vs_market_* (2026-07-24 audit, Issue 2).
    benchmark_prices = prices[prices["ticker"] == BENCHMARK_TICKER].copy()
    if benchmark_prices.empty:
        raise ValueError(
            f"{BENCHMARK_TICKER} not found in prices -- required as the market "
            f"benchmark for beta_1y/momentum_vs_market_* (cross_sectional.py)"
        )
    benchmark = compute_price_features(benchmark_prices)[BENCHMARK_COLS]

    prices, dropped_no_fundamentals = filter_tickers_with_no_fundamentals(prices, fundamentals)
    fundamentals = compute_fundamental_features(fundamentals)
    fundamentals = fill_missing_cagr(fundamentals)
    company_info = load_company_info()
    dividends    = load_dividends()

    fundamentals = attach_filing_dates(fundamentals, company_info)
    fundamentals = filter_excessive_filing_lag(fundamentals)
    dataset = merge_prices_and_fundamentals(prices, fundamentals)
    del prices, fundamentals  # dead from here on; keeping them resident during
    # the macro/dividends merges below was inflating peak memory for nothing
    dataset = merge_company_info(dataset, company_info)
    del company_info
    dataset = merge_macro(dataset)
    dataset = merge_dividends(dataset, dividends)
    compute_features_chunked(dataset, dividends, benchmark, OUTPUT_PATH, chunk_size=150)
    del dataset

    print()
    print("=" * 80)
    print("WRITING MANIFEST & CONFIG")
    print("=" * 80)

    # single read-back for manifest/split_config (unavoidable — both need the
    # full date range / column distributions), now with nothing else resident
    dataset = pd.read_parquet(OUTPUT_PATH)
    manifest = write_manifest(dataset, dropped_no_fundamentals=dropped_no_fundamentals)
    write_split_config(dataset)
    sync_dataset_version(manifest)

    print(f"Saved to: {OUTPUT_PATH}")

    print()
    print("=" * 80)
    print("FINAL DATASET SUMMARY")
    print("=" * 80)
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")
    print()
    print("Columns:")
    for col in dataset.columns:
        print(f"  {col}")
    print()
    print(dataset.head())


if __name__ == "__main__":
    main()
