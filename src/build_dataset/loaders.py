"""loaders.py — read the four raw sources (prices, fundamentals, company info,
dividends) off disk into concatenated DataFrames."""

import pandas as pd

from .paths import COMPANY_INFO_PATH, DIVIDENDS_DIR, FUNDAMENTALS_DIR, PRICES_DIR

# Columns the fundamentals API doesn't actually populate
FUNDAMENTALS_NULL_COLS = [
    "sector",
    "subsector",
    "segment",
    "listing_segment",
    "stock_type",
]

# Collection-time provenance: per-line-item filing dates (`*_filed`, one per
# SEC concept) and XBRL/item6/EX-27 source-document metadata. Real value
# columns (`fds_multiplier` -- actually applied to rescale figures) are NOT
# in this set. Present only in US fundamentals files (SEC source); BR
# (BolsAI) never has them, confirmed by column-name inspection, so dropping
# these unconditionally is a no-op for BR and trims ~19 columns' width off
# the US fundamentals frame before it gets forward-filled onto the (far
# larger) daily panel -- docs/US_DATASET_BUILD_PLAN.md §8.0.
FUNDAMENTALS_PROVENANCE_COLS = {
    "item6_filename", "item6_form", "fds_filename", "fds_form",
    "fds_article", "fds_multiplier_explicit", "tenq_filename", "tenq_form",
}


# =============================================================================
# LOAD ALL PRICE FILES
# =============================================================================

def load_prices(dir=None, tickers=None):
    # `dir=None` (not `dir=PRICES_DIR`) deliberately -- a bound default is
    # captured at import time, so monkeypatching module.PRICES_DIR in a test
    # would silently be ignored. Re-read the module global at call time instead.
    if dir is None:
        dir = PRICES_DIR

    dfs = []
    files = sorted(dir.glob("*.parquet"))
    if tickers is not None:
        # US-scale callers pre-gate the universe from a cheap per-file scan
        # (build_us_dataset.build_universe_gate_from_files) before this ever
        # runs -- loading only the qualifying files avoids ever holding the
        # full 9,593-ticker/34M-row US universe resident just to filter it
        # down afterward (docs/US_DATASET_BUILD_PLAN.md §8.0). None (default)
        # preserves the "load everything" behavior every existing caller
        # (BR, and any test not passing tickers=) relies on.
        files = [f for f in files if f.stem in tickers]

    print()
    print("=" * 80)
    print("LOADING PRICES")
    print("=" * 80)

    for file in files:
        print(f"Loading: {file.name}")
        df = pd.read_parquet(file)
        df = df.dropna(axis=1, how="all")  # Drop all-NA columns per-file
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        dfs.append(df)

    prices = pd.concat(dfs, ignore_index=True, sort=False)
    prices = prices.sort_values(["ticker", "trade_date"])

    # Phantom non-trading day: every OHLC field NaN (confirmed CAMB3
    # 2019-08-15, volume 0) -- a raw-source artifact, not a real trading row.
    ohlc_cols = [c for c in ("open", "high", "low", "close") if c in prices.columns]
    if ohlc_cols:
        all_nan = prices[ohlc_cols].isna().all(axis=1)
        if all_nan.any():
            print(f"Dropping {int(all_nan.sum())} phantom all-NaN OHLC row(s)")
            prices = prices[~all_nan]

    print(f"Total price rows: {len(prices)}")

    return prices


# =============================================================================
# LOAD ALL FUNDAMENTALS
# =============================================================================

def load_fundamentals(dir=None, optimize_dtypes=False):
    """optimize_dtypes: downcast numeric columns to float32 and
    `fundamentals_tier` to category (`cik` excluded -- stays an identifier).
    Default False preserves BR's exact float64 precision (existing tests
    assert fill_cagr_columns output to 1e-6 tolerance against real BR
    figures); build_us_dataset.py passes True, since at US scale this
    fundamentals frame gets forward-filled onto a 15.4M-row daily panel where
    the width otherwise risks OOM (docs/US_DATASET_BUILD_PLAN.md §8.0)."""
    if dir is None:  # see load_prices's comment on why not `dir=FUNDAMENTALS_DIR`
        dir = FUNDAMENTALS_DIR

    dfs = []
    files = sorted(dir.glob("*.parquet"))

    print()
    print("=" * 80)
    print("LOADING FUNDAMENTALS")
    print("=" * 80)

    for file in files:
        print(f"Loading: {file.name}")
        df = pd.read_parquet(file)
        df = df.dropna(axis=1, how="all")  # Drop all-NA columns per-file
        # US fundamentals files (data/raw/us/fundamentals/) carry no `ticker`
        # column at all (only `cik`, which is many-to-one with ticker) and
        # date the period as `end`, not `reference_date` -- filename IS the
        # ticker (one file per ticker, same convention as prices/dividends).
        if "ticker" not in df.columns:
            df["ticker"] = file.stem
        if "reference_date" not in df.columns and "end" in df.columns:
            df = df.rename(columns={"end": "reference_date"})
        df["reference_date"] = pd.to_datetime(df["reference_date"])
        dfs.append(df)

    fundamentals = pd.concat(dfs, ignore_index=True, sort=False)

    # Drop columns that are always null (API doesn't return them)
    cols_to_drop = [
        c for c in FUNDAMENTALS_NULL_COLS
        if c in fundamentals.columns
    ]
    if cols_to_drop:
        fundamentals = fundamentals.drop(columns=cols_to_drop)
        print(f"Dropped always-null columns: {cols_to_drop}")

    # Drop redundant corporate_name — company_info has it with more detail
    if "corporate_name" in fundamentals.columns:
        fundamentals = fundamentals.drop(columns=["corporate_name"])
        print("Dropped redundant 'corporate_name' from fundamentals")

    provenance_cols = [
        c for c in fundamentals.columns
        if c.endswith("_filed") or c in FUNDAMENTALS_PROVENANCE_COLS
    ]
    if provenance_cols:
        fundamentals = fundamentals.drop(columns=provenance_cols)
        print(f"Dropped collection-time provenance columns: {provenance_cols}")

    if optimize_dtypes:
        numeric_cols = fundamentals.select_dtypes(include="number").columns.drop(
            "cik", errors="ignore"
        )
        fundamentals[numeric_cols] = fundamentals[numeric_cols].astype("float32")
        if "fundamentals_tier" in fundamentals.columns:
            fundamentals["fundamentals_tier"] = fundamentals["fundamentals_tier"].astype("category")
        print(f"Downcast {len(numeric_cols)} numeric columns to float32 "
              f"(+ fundamentals_tier to category)")

    fundamentals = fundamentals.sort_values(["ticker", "reference_date"])

    print(f"Total fundamentals rows: {len(fundamentals)}")

    return fundamentals


# =============================================================================
# LOAD COMPANY INFO
# =============================================================================

def load_company_info():

    print()
    print("=" * 80)
    print("LOADING COMPANY INFO")
    print("=" * 80)

    df = pd.read_parquet(COMPANY_INFO_PATH)

    print(f"Company rows: {len(df)}")

    return df


def company_siblings(company_info):
    """cvm_code -> sorted tickers of the same company (PETR3/PETR4-style classes).

    Fundamentals are per-company, tickers are per-share-class; anything that
    counts "companies" (diversification, IC universes, merger-leg resolution)
    should group by this instead of treating each ticker as a separate firm.
    """
    ok = company_info.dropna(subset=["cvm_code", "ticker"])
    ok = ok[ok["cvm_code"].astype(str).str.strip() != ""]
    return {code: sorted(g["ticker"].dropna().unique().tolist())
            for code, g in ok.groupby("cvm_code")}


# =============================================================================
# LOAD DIVIDENDS
# =============================================================================

def load_dividends(dir=None):
    if dir is None:  # see load_prices's comment on why not `dir=DIVIDENDS_DIR`
        dir = DIVIDENDS_DIR

    dfs = []
    files = sorted(dir.glob("*.parquet"))

    print()
    print("=" * 80)
    print("LOADING DIVIDENDS")
    print("=" * 80)

    for file in files:
        print(f"Loading: {file.name}")
        df = pd.read_parquet(file)
        df["ex_date"] = pd.to_datetime(df["ex_date"])
        dfs.append(df)

    dividends = pd.concat(dfs, ignore_index=True)
    dividends = dividends.sort_values(["ticker", "ex_date"])

    # Sanity ceiling: a real BRL per-share dividend is at most low tens even
    # for extreme cases. PDGR3's raw file has all 5 of its events in the
    # hundreds of millions (vendor unit/labeling error, confirmed isolated to
    # this ticker across all 523 raw dividend files, 2026-07-16) -- left in,
    # this inflates div_yield_12m up to 154,600%
    # (docs/TOP50_UNIVERSE_ML_READINESS_AUDIT.md §1.4). Threshold, not a
    # hardcoded ticker name, so it also catches a future recurrence of this
    # same vendor failure mode on a different ticker.
    implausible = dividends["value_per_share"].abs() > 1000
    if implausible.any():
        print(f"Dropping {implausible.sum()} dividend rows with implausible "
              f"value_per_share (>1000): {sorted(dividends.loc[implausible, 'ticker'].unique())}")
        dividends = dividends[~implausible]

    print(f"Total dividend rows: {len(dividends)}")

    return dividends
