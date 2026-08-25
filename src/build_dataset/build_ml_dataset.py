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

import ctypes
import gc
from functools import partial

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import memory
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
from .merge import MergeBatcher, merge_company_info
from .paths import OUTPUT_PATH
from .quality_filters import (
    attach_filing_dates,
    drop_orphan_prefix_rows,
    filter_excessive_filing_lag,
    filter_tickers_with_no_fundamentals,
)
from .repair import repair_isolated_adj_close_glitches, repair_unadjusted_splits

# IBOV-proxy ETF, used as the true market benchmark for beta_1y/
# momentum_vs_market_* (cross_sectional.py) -- collected the same as every
# other ticker (data_collection.config.BENCHMARK_TICKERS) but excluded from
# the dataset's own rows since it has no fundamentals (an ETF, not an
# operating company).
BENCHMARK_TICKER = "BOVA11"


def _reclaim_memory():
    """gc.collect() + malloc_trim(0): CPython's refcounting frees objects
    immediately, but glibc malloc doesn't hand freed heap arenas back to the
    OS just because nothing references them anymore -- RSS reflects the
    high-water mark, not live data. Call after any loop iteration that
    allocates/frees a large (multi-hundred-MB+) frame, or retained-but-unused
    arenas ratchet RSS up iteration over iteration until the OS kills the
    process (confirmed real at US full-build scale, both in Pass 1's
    per-ticker-batch loop and Pass 3's per-row-group loop below). Cheap
    (sub-second) and a no-op on non-glibc platforms (guarded).
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    # Print what the rlimit is actually measuring. malloc_trim moves RSS but
    # barely moves VmData (measured 2.42 -> 2.39 GiB), so this is the only
    # number that says whether a pass left its memory behind -- and the builds
    # have now died three times over guessing at it.
    print(f"[mem] VmData {memory.vmdata_gb():.2f} GiB")


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
    Both main()s derive it from the machine's actual free memory at startup
    (memory.report / memory.chunk_size_for) rather than hardcoding it — the
    default here is only for tests and direct callers. That's what makes the
    same build fit an idle machine and a busy one; the memory.MIN_CHUNK clamp
    is where "too small to compress" is enforced.

    `benchmark`: BOVA11's own price-feature series (trade_date + log_return/
    return_1m/3m/12m), passed through unchanged into Pass 2's
    compute_cross_sectional_features -- see BENCHMARK_TICKER / that
    function's docstring.
    """
    tmp_path = output_path.with_suffix(".tmp.parquet")
    slim_path = output_path.with_suffix(".slim.parquet")

    if tickers is None:
        tickers = dataset["ticker"].unique()
    batches = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

    print()
    print("=" * 80)
    print(f"PASS 1/3: PER-TICKER FEATURES IN {len(batches)} BATCHES (chunk_size={chunk_size})")
    print("=" * 80)

    writer = None
    slim_writer = None
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

            # Stream the slim projection to its own parquet rather than
            # accumulating every batch's slice in a list for Pass 2 to concat.
            # The list held the entire universe's slim frame resident for all
            # of Pass 1 (~3.0GB at full US scale, growing batch over batch),
            # and `pd.concat(slim_parts)` then held the parts AND the
            # concatenated result at once -- a ~6GB spike at the Pass 1->2
            # boundary, on a machine measured with ~8.5GB available. Writing
            # it out costs one extra file and makes Pass 1's memory genuinely
            # bounded by one batch.
            slim_cols = [c for c in CROSS_SECTIONAL_INPUT_COLS if c in batch.columns]
            slim_table = pa.Table.from_pandas(batch[slim_cols], preserve_index=False)
            if slim_writer is None:
                slim_writer = pq.ParquetWriter(slim_path, slim_table.schema)
            else:
                slim_table = slim_table.cast(slim_writer.schema)
            slim_writer.write_table(slim_table)
            del slim_table

            table = pa.Table.from_pandas(batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            else:
                # later batches can promote e.g. an all-NaN int column to float —
                # cast to the schema locked in by batch 1 so row groups stay uniform
                table = table.cast(writer.schema)
            writer.write_table(table)
            del batch, table
            _reclaim_memory()
    finally:
        if writer is not None:
            writer.close()
        if slim_writer is not None:
            slim_writer.close()

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

    # Confirmed real: a full US-scale run OOM-killed inside Pass 2 twice on
    # 2026-08-16 (journalctl: anon-rss ~8.5-9.2GB, this machine's 15GB minus
    # ~4GB in other resident apps) right at this Pass 1->2 boundary.
    _reclaim_memory()

    print()
    print("=" * 80)
    print("PASS 2/3: CROSS-SECTIONAL (MARKET/SECTOR) FEATURES")
    print("=" * 80)

    # Read the slim projection back dictionary-encoded: ticker/sector are the
    # two object columns and at full US scale they hold ~1.9GB of raw Python
    # strings between them (vs ~0.12GB for a float64 column), so handing
    # pandas a dictionary-typed arrow column -- which lands as `category`
    # dtype -- avoids ever materializing them. compute_cross_sectional_features
    # wants exactly that dtype anyway; see its own memory-shape comment.
    slim_tbl = pq.read_table(slim_path)
    for name in ("ticker", "sector"):
        if name in slim_tbl.column_names:
            i = slim_tbl.schema.get_field_index(name)
            slim_tbl = slim_tbl.set_column(i, name, slim_tbl.column(name).dictionary_encode())
    slim = slim_tbl.to_pandas()
    del slim_tbl

    # Returns exactly ["ticker", "trade_date"] + CROSS_SECTIONAL_OUTPUT_COLS,
    # already narrow and already in that order -- re-selecting those columns
    # here is what used to hold the wide frame and its narrow copy at the same
    # time (~6.9GB peak, the measured OOM point), so don't. Assert instead:
    # dropping the re-selection means column order is now the producer's
    # responsibility, and a silent reorder would change the output parquet's
    # schema. Cheap (a list compare), and it fails at the Pass 2 boundary
    # rather than at the end of an hour-long build.
    slim = compute_cross_sectional_features(slim, benchmark)
    expected_cols = ["ticker", "trade_date"] + CROSS_SECTIONAL_OUTPUT_COLS
    assert list(slim.columns) == expected_cols, (
        f"cross-sectional output columns drifted from CROSS_SECTIONAL_OUTPUT_COLS:\n"
        f"  got:      {list(slim.columns)}\n  expected: {expected_cols}"
    )
    slim = slim.set_index(["ticker", "trade_date"])

    # Same rationale as the Pass1->2 boundary above -- Pass 2's own merge/
    # groupby machinery churns through several large temporary frames
    # (confirmed real: a full-universe run OOM-killed INSIDE
    # compute_cross_sectional_features itself, not just at a pass boundary,
    # 2026-08-23), so reclaim before Pass 3 starts its own per-row-group loop.
    _reclaim_memory()

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

            # ponytail: reindex the 12 narrow cross-sectional columns onto this
            # row group's keys and assign them in, rather than DataFrame.join.
            # join routes through concat([left, right], axis=1), which copies
            # the WHOLE 159-column left frame to add 12 columns -- a 306MB
            # allocation per row group at BR scale, and the exact one that
            # raised MemoryError here on 2026-08-23. This copies 12 columns.
            aligned = slim.reindex(
                pd.MultiIndex.from_arrays([batch["ticker"], batch["trade_date"]])
            )
            # A dtype mismatch on the join keys (slim's ticker level is
            # `category`, batch's column is object) would produce an all-NaN
            # block rather than an error, silently shipping a dataset with no
            # cross-sectional features at all -- so check the match landed.
            assert aligned.notna().to_numpy().any(), (
                "cross-sectional features matched no rows in this row group"
            )
            for col in aligned.columns:
                batch[col] = aligned[col].to_numpy()
            del aligned
            batch = clean_dataset(batch)

            table = pa.Table.from_pandas(batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)

            total_rows += len(batch)
            print(f"Row group {rg + 1}/{pf.num_row_groups}: {len(batch)} rows (total {total_rows})")
            del batch, table
            _reclaim_memory()
    finally:
        if writer is not None:
            writer.close()

    tmp_path.unlink()
    slim_path.unlink(missing_ok=True)
    print(f"Feature computation complete: {total_rows} rows")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Size the batches to the RAM actually free right now (minus a reserve for
    # the rest of the machine) instead of a fixed chunk_size=150, and cap this
    # process so an overrun raises MemoryError here rather than handing the
    # kernel's OOM killer a choice of victims. See memory.py.
    chunk_size = memory.report("BR build", memory.BR_BYTES_PER_TICKER,
                               memory.BR_BASELINE_BYTES)

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
    # isolated single-day adj_close glitches (vendor batch defect, unrelated
    # to any split) -- same "repair before continuity" ordering rationale.
    prices       = repair_isolated_adj_close_glitches(prices)
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

    # Merge per ticker-batch rather than whole-universe. The 4 merges widen the
    # panel from 14 price columns to ~60, and Pass 1 takes it to ~130; at BR's
    # 1.76M rows the whole-universe version of that is ~1.7GB before
    # merge_asof's own concat and merge_macro's sort each take a full copy of
    # it. That is where this build actually died (2026-08-23: RLIMIT_DATA hit
    # on an 891MB allocation for a single 67-column block inside
    # merge_prices_and_fundamentals). Batching them means the wide frame never
    # exists for more than chunk_size tickers at a time -- the same fix the US
    # build already runs (see merge.MergeBatcher).
    #
    # Unlike US, the batches are sliced from memory rather than reloaded from
    # disk: BR's split-repair/continuity/BOVA11 steps above are whole-universe
    # by nature (apply_ticker_continuity splices one ticker's history onto
    # another's, so both legs must be present at once), and the narrow
    # pre-merge frames they produce are only ~200MB anyway. It's the merge
    # OUTPUT that's big here, not its inputs.
    tickers = prices["ticker"].unique()
    batch_fn = MergeBatcher(
        # p=/f= as DEFAULT ARGS, not free variables: a closure captures the
        # variable's cell, so the `del prices, fundamentals` below would empty
        # that cell and make this raise NameError on the first batch. Defaults
        # bind the value here and now, which is exactly what lets main() drop
        # its own references while batch_fn keeps the frames alive.
        load_batch=lambda bt, p=prices, f=fundamentals: (
            p[p["ticker"].isin(bt)], f[f["ticker"].isin(bt)]
        ),
        company_info=company_info,
        dividends=dividends,
        # Bind the WHOLE universe's last date: merge_company_info's two status
        # rules ask "did this ticker trade recently?", and a batch of names
        # delisted years ago would otherwise take its own last date as "now"
        # and mark every one of them ATIVO.
        company_info_fn=partial(merge_company_info, dataset_end=prices["trade_date"].max()),
    )
    # main()'s own references, not the closure's -- batch_fn keeps prices/
    # fundamentals alive until its release() drops them after Pass 1.
    del prices, fundamentals, company_info

    compute_features_chunked(None, dividends, benchmark, OUTPUT_PATH, chunk_size=chunk_size,
                              tickers=tickers, batch_fn=batch_fn)
    del batch_fn

    print()
    print("=" * 80)
    print("WRITING MANIFEST & CONFIG")
    print("=" * 80)

    # Stream from parquet instead of loading the full dataset into memory --
    # write_manifest reads one column at a time via parquet_path;
    # write_split_config only ever needs trade_date, so a single-column
    # read stands in for the full dataset (same pattern as build_us_dataset.py)
    manifest = write_manifest(parquet_path=OUTPUT_PATH, dropped_no_fundamentals=dropped_no_fundamentals)
    write_split_config(pd.read_parquet(OUTPUT_PATH, columns=["trade_date"]))
    sync_dataset_version(manifest)

    print(f"Saved to: {OUTPUT_PATH}")

    print()
    print("=" * 80)
    print("FINAL DATASET SUMMARY")
    print("=" * 80)
    pf = pq.ParquetFile(OUTPUT_PATH)
    print(f"Rows: {pf.metadata.num_rows}")
    print(f"Columns: {len(pf.schema_arrow.names)}")
    print()
    print("Columns:")
    for col in pf.schema_arrow.names:
        print(f"  {col}")


if __name__ == "__main__":
    main()
