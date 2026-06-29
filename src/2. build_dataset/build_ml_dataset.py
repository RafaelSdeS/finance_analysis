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
    python build_ml_dataset.py
"""

import sys
from pathlib import Path
import pandas as pd

# cagr_handler.py lives in src/ — make it importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cagr_handler import fill_cagr_columns


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]
PRICES_DIR = ROOT / "data/raw/prices"
FUNDAMENTALS_DIR = ROOT / "data/raw/fundamentals"
COMPANY_INFO_PATH = ROOT / "data/raw/company_info/company_info.parquet"
MACRO_DIR = ROOT / "data/raw/macro"
OUTPUT_PATH = ROOT / "data/processed/ml_dataset.parquet"

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
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        dfs.append(df)

    prices = pd.concat(dfs, ignore_index=True)
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
        df["reference_date"] = pd.to_datetime(df["reference_date"])
        dfs.append(df)

    fundamentals = pd.concat(dfs, ignore_index=True)

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
# FILL MISSING CAGR VALUES
# =============================================================================

def fill_missing_cagr(fundamentals):

    print()
    print("=" * 80)
    print("FILLING MISSING CAGR VALUES")
    print("=" * 80)

    # Group by ticker and apply CAGR filling
    dfs = []
    for ticker in sorted(fundamentals["ticker"].unique()):
        ticker_df = fundamentals[fundamentals["ticker"] == ticker].copy()
        
        # Track coverage before
        earnings_before = ticker_df["cagr_earnings_5y"].isna().sum() if "cagr_earnings_5y" in ticker_df.columns else 0
        revenue_before = ticker_df["cagr_revenue_5y"].isna().sum() if "cagr_revenue_5y" in ticker_df.columns else 0
        
        # Fill CAGR
        ticker_df = fill_cagr_columns(ticker_df)
        
        # Track coverage after
        earnings_after = ticker_df["cagr_earnings_5y_final"].isna().sum()
        revenue_after = ticker_df["cagr_revenue_5y_final"].isna().sum()
        
        dfs.append(ticker_df)
        
        print(f"{ticker}: earnings nulls {earnings_before} → {earnings_after}, revenue nulls {revenue_before} → {revenue_after}")

    fundamentals = pd.concat(dfs, ignore_index=True)
    fundamentals = fundamentals.sort_values(["ticker", "reference_date"])

    print(f"CAGR filling complete: {len(fundamentals)} total rows")

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


# =============================================================================
# MERGE DAILY PRICES + QUARTERLY FUNDAMENTALS
# =============================================================================

def merge_prices_and_fundamentals(prices, fundamentals):

    print()
    print("=" * 80)
    print("MERGING PRICES + FUNDAMENTALS")
    print("=" * 80)

    merged_dfs = []

    for ticker in sorted(prices["ticker"].unique()):

        print(f"Merging {ticker}")

        p = (
            prices[prices["ticker"] == ticker]
            .copy()
            .sort_values("trade_date")
        )

        f = (
            fundamentals[fundamentals["ticker"] == ticker]
            .copy()
            .sort_values("reference_date")
        )

        # merge_asof: uses the most recent fundamental
        # available up to each trade_date (no lookahead bias)
        merged = pd.merge_asof(
            p,
            f,
            left_on="trade_date",
            right_on="reference_date",
            by="ticker",
            direction="backward",
        )

        merged_dfs.append(merged)

    final_df = pd.concat(merged_dfs, ignore_index=True)

    print(f"Merged rows: {len(final_df)}")

    return final_df


# =============================================================================
# ADD STATIC COMPANY INFO
# =============================================================================

def merge_company_info(df, company_info):

    print()
    print("=" * 80)
    print("ADDING COMPANY INFO")
    print("=" * 80)

    # ticker_primary duplicates ticker — drop before merging
    company_info = company_info.drop(
        columns=[c for c in ["ticker_primary"] if c in company_info.columns]
    )

    merged = df.merge(
        company_info,
        on="ticker",
        how="left",
    )

    return merged


# =============================================================================
# ADD MACRO SERIES (SELIC, CDI, IPCA)
# =============================================================================

def merge_macro(dataset):

    print()
    print("=" * 80)
    print("ADDING MACRO SERIES")
    print("=" * 80)

    # Macro is ticker-independent: one value per calendar date applies to all
    # tickers, so no `by=` and no per-ticker loop. Rename each macro date key
    # to avoid colliding with the fundamentals `reference_date` already present.
    for name in ("selic", "cdi", "ipca"):
        print(f"Merging {name}")
        m = pd.read_parquet(MACRO_DIR / f"{name}.parquet")[["reference_date", name]]
        m = m.rename(columns={"reference_date": f"{name}_date"}).sort_values(f"{name}_date")
        dataset = pd.merge_asof(
            dataset.sort_values("trade_date"),
            m,
            left_on="trade_date",
            right_on=f"{name}_date",
            direction="backward",   # latest macro value <= trade_date (no lookahead)
        ).drop(columns=f"{name}_date")

    return dataset.sort_values(["ticker", "trade_date"]).reset_index(drop=True)


# =============================================================================
# CLEAN DATA
# =============================================================================

def clean_dataset(df):

    print()
    print("=" * 80)
    print("CLEANING DATASET")
    print("=" * 80)

    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed duplicates: {before - len(df)}")

    df = df.sort_values(["ticker", "trade_date"]).reset_index(drop=True)

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    prices       = load_prices()
    fundamentals = load_fundamentals()
    fundamentals = fill_missing_cagr(fundamentals)
    company_info = load_company_info()

    dataset = merge_prices_and_fundamentals(prices, fundamentals)
    dataset = merge_company_info(dataset, company_info)
    dataset = merge_macro(dataset)
    dataset = clean_dataset(dataset)

    print()
    print("=" * 80)
    print("SAVING DATASET")
    print("=" * 80)

    dataset.to_parquet(OUTPUT_PATH, index=False)

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