"""cross_sectional.py — sector/market-relative features (Pass 2 of
compute_features_chunked). Unlike features.py, these need the full universe
on the same date at once, so they can't run on a ticker-batch in isolation.
"""

import numpy as np
import pandas as pd

# Every column this module emits is a derived statistic — a z-score, a
# percentile rank, a return difference, a rolling beta — carrying two or three
# significant digits of real signal on top of inputs that are themselves
# noisy estimates. float32's ~7 digits is far more than any of them mean, and
# this is the one stage that holds the whole universe at once: at US scale the
# 12 output columns are 1.47GB in float64 and 0.74GB here, and Pass 3 then
# keeps that same frame resident while it cleans every row group. Halving it
# is the cheapest real headroom available in either pass. (US fundamentals are
# already float32 throughout via load_fundamentals(optimize_dtypes=True), so
# this also stops the outputs being wider than the inputs they derive from.)
OUT_DTYPE = "float32"

# Inputs compute_cross_sectional_features() needs, and the columns it adds —
# used to slim the frame down before holding the full universe in memory.
CROSS_SECTIONAL_INPUT_COLS = [
    "ticker", "trade_date", "sector", "pl", "pvp", "roe", "debt_equity",
    "div_yield_12m", "return_1m", "return_3m", "return_12m", "log_return",
]
CROSS_SECTIONAL_OUTPUT_COLS = [
    "pl_zscore_sector", "pvp_zscore_sector", "roe_zscore_sector", "debt_equity_zscore_sector",
    "div_yield_sector_percentile",
    "momentum_vs_market_1m", "momentum_vs_market_3m", "momentum_vs_market_12m",
    "momentum_vs_sector_1m", "momentum_vs_sector_3m", "momentum_vs_sector_12m",
    "beta_1y",
]

# Columns compute_price_features leaves on the benchmark (BOVA11) series that
# this module actually needs -- build_ml_dataset.main() computes these the
# same way as every other ticker (same function, same methodology) before
# passing the result in as `benchmark`.
BENCHMARK_COLS = ["trade_date", "log_return", "return_1m", "return_3m", "return_12m"]

# Rolling window for beta vs. market, in trading days (~1 calendar year,
# matching return_12m/price_percentile_1y's convention elsewhere). min_periods
# is deliberately less than the full window (unlike a fixed-length sum like
# return_12m) so beta isn't NaN for a ticker's entire first year -- but not so
# low that a 5-10 day window produces a wildly unstable covariance estimate.
BETA_WINDOW = 252
BETA_MIN_PERIODS = 60


def compute_cross_sectional_features(df, benchmark):
    """Sector/market-relative features: how does this stock compare to its
    sector peers and to the true market index on the same date. Must run on
    the full dataset in one shot — computing this per ticker-batch (as an
    earlier version did) silently compares each stock against whichever
    handful of tickers landed in its batch instead of the true sector,
    corrupting every sector-relative column.

    `benchmark`: BOVA11's (IBOV-proxy ETF) own price-feature series --
    trade_date + log_return/return_1m/return_3m/return_12m, computed by the
    SAME compute_price_features() used for every other ticker, so it's
    methodologically identical (same split-repair/continuity treatment, same
    return-window conventions). Used as the market series for beta_1y and
    momentum_vs_market_* (2026-07-24 audit, Issue 2 -- previously an
    equal-weighted mean of whatever tickers happened to be in the collected
    panel on that date, which silently redefines "the market" as "the
    companies that survived to dataset-end," a second, benchmark-level
    survivorship bias distinct from the universe-selection-level one
    documented elsewhere). BOVA11 itself is never a row in the output
    dataset (quality_filters.filter_tickers_with_no_fundamentals still drops
    it, having no fundamentals, and rightly so -- it's an ETF, not an
    operating company) -- it's threaded through purely as an external
    reference series, so this change doesn't touch row/ticker counts,
    manifest fingerprinting, or dataset_v{N} shape, only the DEFINITION of
    beta_1y/momentum_vs_market_* (Issue 12's original "changes dataset shape"
    objection assumed BOVA11 would need to become a ticker row itself, which
    this design avoids).

    Sector-relative features (momentum_vs_sector_*, *_zscore_sector,
    div_yield_sector_percentile) are unaffected by this and still compare
    against sector peers within the panel -- there's no equivalent
    "benchmark" for a sector, and peer comparison is exactly the intended
    semantics there.
    """

    print()
    print("=" * 80)
    print("COMPUTING CROSS-SECTIONAL (MARKET/SECTOR) FEATURES")
    print("=" * 80)

    # --- MEMORY SHAPE (why this function looks the way it does) ---
    #
    # This is the one stage that must hold the whole universe at once, so it
    # is also the one stage where frame width is a hard memory constraint.
    # Measured at full US scale (15.35M rows, 2026-08-23): the two OBJECT
    # columns dominate everything else combined -- `sector` 1.07GB and
    # `ticker` 0.80GB of raw Python strings, vs ~0.12GB for a float64 column.
    #
    # Two consequences, both load-bearing:
    #
    # 1. ticker/sector are cast to `category` (dictionary + int codes) for the
    #    groupby work: 1.87GB -> ~0.05GB, and the groupbys hash ints instead
    #    of strings. `sector` is an INPUT only (never an output column), so it
    #    is simply never carried into the result.
    #
    # 2. Results are accumulated into a separate NARROW frame (`out`) rather
    #    than assigned back onto `df`. Widening `df` meant the caller then had
    #    to slice the 12 output columns back out of a ~24-column frame, which
    #    holds the wide frame and the narrow copy simultaneously -- measured
    #    ~6.9GB peak against ~8.5GB available, and the actual point a real
    #    full-universe run was OOM-killed (2026-08-23, died between this
    #    function's final print and the caller's re-selection). `out` is built
    #    directly in `["ticker", "trade_date"] + CROSS_SECTIONAL_OUTPUT_COLS`
    #    order so the caller needs no re-selection at all -- keep the
    #    assignment order below in sync with that list.
    df["ticker"] = df["ticker"].astype("category")
    df["sector"] = df["sector"].astype("category")

    out = pd.DataFrame(index=df.index)

    # ponytail: vectorized z-score via cython groupby transforms (no Python per-group calls)
    # NaN-sector rows are dropped by groupby and stay NaN, matching the old loop's skip.
    # observed=True: explicit regardless of pandas-version default, now that sector is
    # categorical -- without it, a categorical grouper can produce one group per
    # (trade_date, sector) COMBINATION including sector levels never actually observed
    # on that date, wasting work computing empty groups.
    sector_grp = df.groupby(["trade_date", "sector"], sort=False, observed=True)
    for col in ["pl", "pvp", "roe", "debt_equity"]:
        if col in df.columns:
            mean = sector_grp[col].transform("mean")
            std = sector_grp[col].transform("std")
            # std <= 0 or NaN (single-stock sectors) → NaN, same as the old guard
            out[f"{col}_zscore_sector"] = ((df[col] - mean) / std.where(std > 0)).astype(OUT_DTYPE)
            del mean, std

    # Sector-of-one guard: with a single member, a stock's "vs sector" metric
    # trivially collapses to itself (mean = own value, rank = 100th pct) —
    # NaN it out rather than silently reporting "in line with sector".
    sector_size = sector_grp["ticker"].transform("size")

    # Dividend yield percentile: percentile rank within sector per date
    out["div_yield_sector_percentile"] = sector_grp["div_yield_12m"].rank(
        pct=True
    ).where(sector_size > 1).astype(OUT_DTYPE)

    # --- MOMENTUM DECOMPOSITION (stock vs sector vs market) ---

    # Market momentum: subtract BOVA11's OWN return on the same date -- a
    # single shared benchmark series, not a per-date panel mean, so no
    # self-exclusion/NaN-dilution logic is needed here (BOVA11 is never a row
    # in `df` to begin with). Exact trade_date match (not asof): both series
    # are same-exchange (B3) daily data sharing the same trading calendar, so
    # a date BOVA11 didn't trade is correctly NaN here too, not silently
    # papered over with a stale prior value.
    #
    # .map() against a trade_date-indexed lookup, not .merge(): merge() builds
    # an entirely new frame holding every existing column plus the benchmark
    # ones -- a full second copy of the whole-universe frame just to attach 4
    # narrow columns. Each mapped series is also a local, freed as soon as
    # it's consumed, rather than a column parked on `df` until a later drop.
    # drop_duplicates: .map needs a unique index, and a duplicated benchmark
    # date would silently raise here rather than quietly broadcasting as the
    # old merge did.
    bench = benchmark[BENCHMARK_COLS].drop_duplicates("trade_date").set_index("trade_date")

    for horizon in ("1m", "3m", "12m"):
        mkt = df["trade_date"].map(bench[f"return_{horizon}"])
        out[f"momentum_vs_market_{horizon}"] = (df[f"return_{horizon}"] - mkt).astype(OUT_DTYPE)
        del mkt

    # Sector momentum: subtract sector mean (per date, sector) from each
    # return. Reuses sector_grp (built once above) instead of rebuilding
    # groupby(["trade_date", "sector"]) 3 more times -- at full US scale
    # (~16M rows) that was 4 total group-key factorizations over the whole
    # universe instead of 1, real transient-memory/CPU waste that was part
    # of what pushed a real run into an OOM kill (confirmed via journalctl,
    # 2026-08-16).
    for horizon in ("1m", "3m", "12m"):
        out[f"momentum_vs_sector_{horizon}"] = (
            df[f"return_{horizon}"] - sector_grp[f"return_{horizon}"].transform("mean")
        ).where(sector_size > 1).astype(OUT_DTYPE)

    del sector_grp, sector_size

    # --- ROLLING BETA VS MARKET ---

    # Same BOVA11 series as above, now rolled per-ticker over TIME (not a
    # same-date snapshot) -- needs one groupby("ticker") pass.
    #
    # Grouped over a 3-column projection, not the full frame: each yielded `g`
    # is a copy of every column it carries, so grouping the wide frame churns
    # ~20 dead columns per ticker for the 2 the rolling actually reads.
    #
    # Accumulate only the narrow (cov/var-derived) per-ticker beta Series, not
    # a full-width copy of `g` -- this loop runs on the WHOLE universe at once
    # (unlike compute_price_features's identically-shaped per-ticker loop,
    # which only ever sees one ~150-ticker Pass-1 batch). Appending full `g`
    # slices into a list and pd.concat-ing them at the end held a full SECOND
    # copy of the entire frame -- the same "accumulate everything then concat"
    # shape that OOM-killed merge_prices_and_fundamentals at US scale
    # (docs/US_DATASET_BUILD_PLAN.md §8.0.1); this was the untouched sibling.
    # The concatenated single-column Series aligns back by each group's
    # preserved, unique row index -- verified byte-identical to the old
    # full-frame-accumulation output.
    # Results land in ONE preallocated array, written by position, instead of
    # accumulating a per-ticker list of Series for a final pd.concat. The list
    # form cost roughly three copies of the beta column at once at US scale --
    # each of the ~2,900 parts carries its own index alongside its values
    # (doubling it), and concat then holds every part AND the joined result
    # live simultaneously. Preallocating makes it exactly one array, and the
    # loop's only per-iteration allocations are the group's own two rolling
    # temporaries, which are freed each pass.
    beta_input = df[["ticker", "trade_date", "log_return"]].copy()
    beta_input["_mkt_log_return"] = df["trade_date"].map(bench["log_return"])
    beta = np.full(len(df), np.nan, dtype=OUT_DTYPE)
    for _, g in beta_input.groupby("ticker", sort=False, observed=True):
        g = g.sort_values("trade_date")
        cov = g["log_return"].rolling(BETA_WINDOW, min_periods=BETA_MIN_PERIODS).cov(g["_mkt_log_return"])
        var = g["_mkt_log_return"].rolling(BETA_WINDOW, min_periods=BETA_MIN_PERIODS).var()
        # get_indexer maps this group's (preserved, unique) row labels back to
        # positions in `df` -- the same alignment pd.concat used to do by index,
        # just without materializing the intermediate Series.
        beta[df.index.get_indexer(g.index)] = (cov / var).to_numpy()
    del beta_input
    out["beta_1y"] = beta
    del beta

    # Key columns last, inserted at the front: `out` is now exactly
    # ["ticker", "trade_date"] + CROSS_SECTIONAL_OUTPUT_COLS, in that order.
    # ticker goes back to object -- it's the join key Pass 3 matches against
    # its own object ticker column, and a categorical key on one side of a
    # join is exactly the kind of silent all-NaN mismatch that no unit test
    # at fixture scale would catch.
    out.insert(0, "trade_date", df["trade_date"])
    out.insert(0, "ticker", df["ticker"].astype(object))

    print(f"Cross-sectional features computed for {len(out)} rows")
    return out
