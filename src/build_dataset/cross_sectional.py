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

# Rolling window for beta vs. market, in trading days (~1 calendar year,
# matching return_12m/price_percentile_1y's convention elsewhere). min_periods
# is deliberately less than the full window (unlike a fixed-length sum like
# return_12m) so beta isn't NaN for a ticker's entire first year -- but not so
# low that a 5-10 day window produces a wildly unstable covariance estimate.
BETA_WINDOW = 252
BETA_MIN_PERIODS = 60


def _exclude_self_mean(df, group_col, value_col):
    """Equal-weighted mean of value_col within each group_col group, EXCLUDING
    the row's own value -- (group_sum - x) / (n - 1). A plain groupby(...).mean()
    (the previous convention here) is self-inclusive: a ticker's own return
    pulls its own "market"/"sector" reference toward itself, artificially
    shrinking every momentum/beta figure -- materially so on thin dates with
    few tickers (2026-07-23 audit). NaN when there's no other member to
    compare against (n <= 1), not a division by zero.

    n/total are both derived from value_col's own non-NaN count (groupby
    "count"/"sum" both skip NaN) -- NOT a blanket ticker-count like
    groupby(...).transform("size"), which counts every row in the group
    including ones where value_col is NaN. Using "size" as the denominator
    while "sum" already dropped the NaN rows silently dilutes the mean
    toward zero on any date where some tickers have an undefined value
    (e.g. still in return_12m's warm-up year) -- worse the thinner/younger
    the universe on that date, exactly where this feature matters most
    (2026-07-24 audit)."""
    grp = df.groupby(group_col)[value_col]
    total = grp.transform("sum")
    n = grp.transform("count")
    self_val = df[value_col]
    self_defined = self_val.notna()
    other_total = total - self_val.where(self_defined, 0.0)
    other_n = n - self_defined.astype(int)
    return (other_total / other_n).where(other_n > 0)


def compute_cross_sectional_features(df):
    """Sector/market-relative features: how does this stock compare to every
    OTHER stock trading on the same date. Must run on the full dataset in one
    shot — computing this per ticker-batch (as an earlier version did) silently
    compares each stock against whichever handful of tickers landed in its
    batch instead of the true market/sector, corrupting every one of these
    columns.

    "Market" here means the equal-weighted mean return of every OTHER ticker
    in the dataset's current universe on that date -- not a true cap-weighted
    index. BOVA11 (the collected IBOV-proxy benchmark) is excluded from the
    dataset entirely upstream (quality_filters.filter_tickers_with_no_fundamentals
    drops it, having no fundamentals) and is not wired in here; routing it
    through as an actual index benchmark was considered (2026-07-23 audit,
    Issue 12) and deliberately deferred -- it changes the dataset's row/ticker
    shape (manifest fingerprinting, dataset_v{N} versioning, any downstream
    code iterating "all tickers" expecting only operating companies) and
    redefines beta_1y/momentum_vs_market_* semantically, which is a bigger
    design decision than a bug fix. What WAS fixed here is the self-inclusion
    bug below -- a real correctness issue independent of that larger question.
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

    # ponytail: use groupby.transform() for vectorized momentum (1000x faster than loops)
    # Market momentum: subtract the mean return of every OTHER ticker (per
    # date) from each return -- self-EXCLUDED, see _exclude_self_mean.
    df["momentum_vs_market_1m"] = (
        df["return_1m"] - _exclude_self_mean(df, "trade_date", "return_1m")
    )
    df["momentum_vs_market_3m"] = (
        df["return_3m"] - _exclude_self_mean(df, "trade_date", "return_3m")
    )
    df["momentum_vs_market_12m"] = (
        df["return_12m"] - _exclude_self_mean(df, "trade_date", "return_12m")
    )

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

    # Market return: equal-weighted mean log_return of every OTHER ticker in
    # the full universe, per date -- self-EXCLUDED (see _exclude_self_mean;
    # a ticker's own return previously pulled its own benchmark toward
    # itself, artificially shrinking its measured beta). Reuses the same
    # full-universe requirement as the momentum/zscore features above, but
    # beta then needs a per-ticker rolling window over TIME (not a same-date
    # snapshot), so unlike everything above it can't stay a single
    # groupby(date).transform() -- needs one groupby("ticker") pass, same
    # shape as compute_price_features.
    market_log_return = _exclude_self_mean(df, "trade_date", "log_return")

    result = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("trade_date")
        mkt = market_log_return.loc[g.index]
        cov = g["log_return"].rolling(BETA_WINDOW, min_periods=BETA_MIN_PERIODS).cov(mkt)
        var = mkt.rolling(BETA_WINDOW, min_periods=BETA_MIN_PERIODS).var()
        g["beta_1y"] = cov / var
        result.append(g)
    df = pd.concat(result, ignore_index=True)

    print(f"Cross-sectional features computed for {len(df)} rows")
    return df
