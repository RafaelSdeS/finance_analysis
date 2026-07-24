"""cross_sectional.py — sector/market-relative features (Pass 2 of
compute_features_chunked). Unlike features.py, these need the full universe
on the same date at once, so they can't run on a ticker-batch in isolation.
"""

import pandas as pd

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

    # ponytail: vectorized z-score via cython groupby transforms (no Python per-group calls)
    # NaN-sector rows are dropped by groupby and stay NaN, matching the old loop's skip.
    sector_grp = df.groupby(["trade_date", "sector"], sort=False)
    for col in ["pl", "pvp", "roe", "debt_equity"]:
        if col in df.columns:
            mean = sector_grp[col].transform("mean")
            std = sector_grp[col].transform("std")
            # std <= 0 or NaN (single-stock sectors) → NaN, same as the old guard
            df[f"{col}_zscore_sector"] = (df[col] - mean) / std.where(std > 0)

    # Sector-of-one guard: with a single member, a stock's "vs sector" metric
    # trivially collapses to itself (mean = own value, rank = 100th pct) —
    # NaN it out rather than silently reporting "in line with sector".
    sector_size = sector_grp["ticker"].transform("size")

    # Dividend yield percentile: percentile rank within sector per date
    df["div_yield_sector_percentile"] = sector_grp["div_yield_12m"].rank(
        pct=True
    ).where(sector_size > 1)

    # --- MOMENTUM DECOMPOSITION (stock vs sector vs market) ---

    # Market momentum: subtract BOVA11's OWN return on the same date -- a
    # single shared benchmark series, not a per-date panel mean, so no
    # self-exclusion/NaN-dilution logic is needed here (BOVA11 is never a row
    # in `df` to begin with). Exact trade_date match (not asof): both series
    # are same-exchange (B3) daily data sharing the same trading calendar, so
    # a date BOVA11 didn't trade is correctly NaN here too, not silently
    # papered over with a stale prior value.
    bench = benchmark[BENCHMARK_COLS].rename(columns={
        "log_return": "_mkt_log_return", "return_1m": "_mkt_return_1m",
        "return_3m": "_mkt_return_3m", "return_12m": "_mkt_return_12m",
    })
    df = df.merge(bench, on="trade_date", how="left")

    df["momentum_vs_market_1m"] = df["return_1m"] - df["_mkt_return_1m"]
    df["momentum_vs_market_3m"] = df["return_3m"] - df["_mkt_return_3m"]
    df["momentum_vs_market_12m"] = df["return_12m"] - df["_mkt_return_12m"]

    # Sector momentum: subtract sector mean (per date, sector) from each return
    df["momentum_vs_sector_1m"] = (
        df["return_1m"]
        - df.groupby(["trade_date", "sector"])["return_1m"].transform("mean")
    ).where(sector_size > 1)
    df["momentum_vs_sector_3m"] = (
        df["return_3m"]
        - df.groupby(["trade_date", "sector"])["return_3m"].transform("mean")
    ).where(sector_size > 1)
    df["momentum_vs_sector_12m"] = (
        df["return_12m"]
        - df.groupby(["trade_date", "sector"])["return_12m"].transform("mean")
    ).where(sector_size > 1)

    # --- ROLLING BETA VS MARKET ---

    # Same BOVA11 series as above, now rolled per-ticker over TIME (not a
    # same-date snapshot) -- needs one groupby("ticker") pass, same shape as
    # compute_price_features.
    result = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("trade_date")
        cov = g["log_return"].rolling(BETA_WINDOW, min_periods=BETA_MIN_PERIODS).cov(g["_mkt_log_return"])
        var = g["_mkt_log_return"].rolling(BETA_WINDOW, min_periods=BETA_MIN_PERIODS).var()
        g["beta_1y"] = cov / var
        result.append(g)
    df = pd.concat(result, ignore_index=True)
    df = df.drop(columns=["_mkt_log_return", "_mkt_return_1m", "_mkt_return_3m", "_mkt_return_12m"])

    print(f"Cross-sectional features computed for {len(df)} rows")
    return df
