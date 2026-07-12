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


# =============================================================================
# LOAD ALL PRICE FILES
# =============================================================================

def load_prices():

    dfs = []
    files = sorted(PRICES_DIR.glob("*.parquet"))

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

    print(f"Total price rows: {len(prices)}")

    return prices


# =============================================================================
# LOAD ALL FUNDAMENTALS
# =============================================================================

def load_fundamentals():

    dfs = []
    files = sorted(FUNDAMENTALS_DIR.glob("*.parquet"))

    print()
    print("=" * 80)
    print("LOADING FUNDAMENTALS")
    print("=" * 80)

    for file in files:
        print(f"Loading: {file.name}")
        df = pd.read_parquet(file)
        df = df.dropna(axis=1, how="all")  # Drop all-NA columns per-file
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

def load_dividends():

    dfs = []
    files = sorted(DIVIDENDS_DIR.glob("*.parquet"))

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

    print(f"Total dividend rows: {len(dividends)}")

    return dividends
