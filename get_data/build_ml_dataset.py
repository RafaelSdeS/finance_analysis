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

from pathlib import Path
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PRICES_DIR = Path("data/raw/prices")
FUNDAMENTALS_DIR = Path("data/raw/fundamentals")

COMPANY_INFO_PATH = Path(
    "data/raw/company_info/company_info.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/ml_dataset.parquet"
)


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

        df["trade_date"] = pd.to_datetime(
            df["trade_date"]
        )

        dfs.append(df)

    prices = pd.concat(
        dfs,
        ignore_index=True
    )

    prices = prices.sort_values(
        ["ticker", "trade_date"]
    )

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

        df["reference_date"] = pd.to_datetime(
            df["reference_date"]
        )

        dfs.append(df)

    fundamentals = pd.concat(
        dfs,
        ignore_index=True
    )

    fundamentals = fundamentals.sort_values(
        ["ticker", "reference_date"]
    )

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


# =============================================================================
# MERGE DAILY PRICES + QUARTERLY FUNDAMENTALS
# =============================================================================

def merge_prices_and_fundamentals(
    prices,
    fundamentals
):

    print()
    print("=" * 80)
    print("MERGING PRICES + FUNDAMENTALS")
    print("=" * 80)

    merged_dfs = []

    tickers = sorted(
        prices["ticker"].unique()
    )

    for ticker in tickers:

        print(f"Merging {ticker}")

        p = (
            prices[
                prices["ticker"] == ticker
            ]
            .copy()
            .sort_values("trade_date")
        )

        f = (
            fundamentals[
                fundamentals["ticker"] == ticker
            ]
            .copy()
            .sort_values("reference_date")
        )

        # merge_asof:
        # pega o fundamento mais recente
        # disponível até aquela data

        merged = pd.merge_asof(
            p,
            f,
            left_on="trade_date",
            right_on="reference_date",
            by="ticker",
            direction="backward"
        )

        merged_dfs.append(merged)

    final_df = pd.concat(
        merged_dfs,
        ignore_index=True
    )

    print(f"Merged rows: {len(final_df)}")

    return final_df


# =============================================================================
# ADD STATIC COMPANY INFO
# =============================================================================

def merge_company_info(
    df,
    company_info
):

    print()
    print("=" * 80)
    print("ADDING COMPANY INFO")
    print("=" * 80)

    merged = df.merge(
        company_info,
        on="ticker",
        how="left",
        suffixes=("", "_company")
    )

    return merged


# =============================================================================
# CLEAN DATA
# =============================================================================

def clean_dataset(df):

    print()
    print("=" * 80)
    print("CLEANING DATASET")
    print("=" * 80)

    # remove duplicados
    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(f"Removed duplicates: {before - after}")

    # ordena
    df = df.sort_values(
        ["ticker", "trade_date"]
    )

    df = df.reset_index(drop=True)

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # =========================================================
    # Load datasets
    # =========================================================

    prices = load_prices()

    fundamentals = load_fundamentals()

    company_info = load_company_info()

    # =========================================================
    # Merge
    # =========================================================

    dataset = merge_prices_and_fundamentals(
        prices,
        fundamentals
    )

    dataset = merge_company_info(
        dataset,
        company_info
    )

    # =========================================================
    # Clean
    # =========================================================

    dataset = clean_dataset(dataset)

    # =========================================================
    # Save
    # =========================================================

    print()
    print("=" * 80)
    print("SAVING DATASET")
    print("=" * 80)

    dataset.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print(f"Saved to: {OUTPUT_PATH}")

    # =========================================================
    # Final summary
    # =========================================================

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


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()