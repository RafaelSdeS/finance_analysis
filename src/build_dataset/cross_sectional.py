"""cross_sectional.py — sector/market-relative features (Pass 2 of
compute_features_chunked). Unlike features.py, these need the full universe
on the same date at once, so they can't run on a ticker-batch in isolation.
"""

# Inputs compute_cross_sectional_features() needs, and the columns it adds —
# used to slim the frame down before holding the full universe in memory.
CROSS_SECTIONAL_INPUT_COLS = [
    "ticker", "trade_date", "sector", "pl", "pvp", "roe", "debt_equity",
    "div_yield_12m", "return_1m", "return_3m", "return_12m",
]
CROSS_SECTIONAL_OUTPUT_COLS = [
    "pl_zscore_sector", "pvp_zscore_sector", "roe_zscore_sector", "debt_equity_zscore_sector",
    "div_yield_sector_percentile",
    "momentum_vs_market_1m", "momentum_vs_market_3m", "momentum_vs_market_12m",
    "momentum_vs_sector_1m", "momentum_vs_sector_3m", "momentum_vs_sector_12m",
]


def compute_cross_sectional_features(df):
    """Sector/market-relative features: how does this stock compare to every
    OTHER stock trading on the same date. Must run on the full dataset in one
    shot — computing this per ticker-batch (as an earlier version did) silently
    compares each stock against whichever handful of tickers landed in its
    batch instead of the true market/sector, corrupting every one of these
    columns.
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
    # Market momentum: subtract market mean (per date) from each return
    df["momentum_vs_market_1m"] = (
        df["return_1m"] - df.groupby("trade_date")["return_1m"].transform("mean")
    )
    df["momentum_vs_market_3m"] = (
        df["return_3m"] - df.groupby("trade_date")["return_3m"].transform("mean")
    )
    df["momentum_vs_market_12m"] = (
        df["return_12m"] - df.groupby("trade_date")["return_12m"].transform("mean")
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

    print(f"Cross-sectional features computed for {len(df)} rows")
    return df
