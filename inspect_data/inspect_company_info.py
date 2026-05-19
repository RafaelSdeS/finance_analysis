"""
inspect_company_info.py
=======================

Inspeciona o parquet de company info
e mostra:
    - estrutura
    - colunas
    - tipos
    - nulls
    - amostras
    - estatísticas básicas

Uso:
    python inspect_company_info.py
"""

from pathlib import Path
import pandas as pd


PARQUET_PATH = "../data/raw/company_info/company_info.parquet"


def print_header(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_section(title: str):
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def main():

    path = Path(PARQUET_PATH)

    if not path.exists():
        print(f"File not found: {path}")
        return

    df = pd.read_parquet(path)

    print_header("COMPANY INFO INSPECTION")

    print(f"File     : {path}")
    print(f"Rows     : {len(df)}")
    print(f"Columns  : {len(df.columns)}")
    print(f"Memory   : {df.memory_usage(deep=True).sum() / 1024**2:.4f} MB")

    # ==========================================================
    # Columns
    # ==========================================================

    print_section("COLUMNS")

    for col in df.columns:

        dtype = str(df[col].dtype)

        null_pct = (
            df[col].isnull().mean() * 100
        )

        unique_count = df[col].nunique(dropna=True)

        print(
            f"{col:<20} "
            f"{dtype:<15} "
            f"nulls={null_pct:>6.2f}% "
            f"unique={unique_count}"
        )

    # ==========================================================
    # Sample rows
    # ==========================================================

    print_section("SAMPLE ROWS")

    print(df.head(10).to_string(index=False))

    # ==========================================================
    # Null analysis
    # ==========================================================

    print_section("NULL ANALYSIS")

    nulls = df.isnull().sum()

    for col, count in nulls.items():

        pct = count / len(df) * 100

        print(
            f"{col:<20} "
            f"{count:>5} "
            f"({pct:>6.2f}%)"
        )

    # ==========================================================
    # Categorical distributions
    # ==========================================================

    categorical_cols = [
        col for col in df.columns
        if df[col].dtype == "object"
    ]

    for col in categorical_cols:

        print_section(f"VALUE COUNTS: {col}")

        print(
            df[col]
            .value_counts(dropna=False)
            .head(20)
            .to_string()
        )

    # ==========================================================
    # Duplicate check
    # ==========================================================

    print_section("DUPLICATE CHECK")

    duplicates = df.duplicated().sum()

    print(f"Duplicate rows: {duplicates}")

    if "ticker" in df.columns:

        duplicated_tickers = (
            df["ticker"]
            .duplicated()
            .sum()
        )

        print(f"Duplicated tickers: {duplicated_tickers}")

    # ==========================================================
    # Data preview
    # ==========================================================

    print_section("DATAFRAME INFO")

    print(df.info())

    print()
    print("=" * 80)
    print("INSPECTION FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()