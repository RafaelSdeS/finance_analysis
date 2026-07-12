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
    CROSS_SECTIONAL_INPUT_COLS,
    CROSS_SECTIONAL_OUTPUT_COLS,
    compute_cross_sectional_features,
)
from .features import (
    compute_advanced_features,
    compute_dividend_features,
    compute_fundamental_features,
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
    filter_excessive_filing_lag,
    filter_tickers_with_no_fundamentals,
)
from .repair import repair_unadjusted_splits


# =============================================================================
# FEATURE COMPUTATION
# =============================================================================

def compute_features_chunked(dataset, dividends, output_path, chunk_size=150):
    """Three-pass, memory-bounded feature computation.

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
    """
    tmp_path = output_path.with_suffix(".tmp.parquet")

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
            batch = dataset[dataset["ticker"].isin(batch_tickers)].copy()
            print(f"Batch {batch_idx}/{len(batches)}: {len(batch_tickers)} tickers, {len(batch)} rows")

            batch = compute_price_features(batch)
            batch = compute_dividend_features(batch, dividends)
            batch = compute_macro_features(batch)
            batch = recompute_valuation_daily(batch)
            batch = compute_advanced_features(batch)

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

    print()
    print("=" * 80)
    print("PASS 2/3: CROSS-SECTIONAL (MARKET/SECTOR) FEATURES")
    print("=" * 80)

    slim = pd.concat(slim_parts, ignore_index=True)
    del slim_parts
    slim = compute_cross_sectional_features(slim)
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
    # splice BEFORE split repair: corporate_events.parquet records splits
    # under each entity's current canonical ticker (e.g. BHIA3), even for
    # splits that happened while it traded as VVAR3/VIIA3 — repair can only
    # find those rows once continuity has renamed them onto the new ticker.
    prices, fundamentals = apply_ticker_continuity(prices, fundamentals)
    prices       = repair_unadjusted_splits(prices)
    prices       = filter_tickers_with_no_fundamentals(prices, fundamentals)
    fundamentals = compute_fundamental_features(fundamentals)
    fundamentals = fill_missing_cagr(fundamentals)
    company_info = load_company_info()
    dividends    = load_dividends()

    fundamentals = attach_filing_dates(fundamentals, company_info)
    fundamentals = filter_excessive_filing_lag(fundamentals)
    dataset = merge_prices_and_fundamentals(prices, fundamentals)
    dataset = merge_company_info(dataset, company_info)
    dataset = merge_macro(dataset)
    dataset = merge_dividends(dataset, dividends)
    # free the pre-merge tables before the heavy feature/clean passes — they're
    # no longer needed but stay resident (still-named locals) otherwise
    del prices, fundamentals, company_info
    compute_features_chunked(dataset, dividends, OUTPUT_PATH, chunk_size=150)
    del dataset

    print()
    print("=" * 80)
    print("WRITING MANIFEST & CONFIG")
    print("=" * 80)

    # single read-back for manifest/split_config (unavoidable — both need the
    # full date range / column distributions), now with nothing else resident
    dataset = pd.read_parquet(OUTPUT_PATH)
    manifest = write_manifest(dataset)
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
