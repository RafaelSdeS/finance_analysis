# inspect_final_dataset.py
#
# Visualiza e inspeciona o dataset final consolidado.
#
# Uso:
#   python inspect_final_dataset.py
#
# Ou:
#   python inspect_final_dataset.py --file data/processed/ml_dataset.parquet

import argparse
from pathlib import Path

import pandas as pd


def print_separator():
    print("-" * 80)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        type=str,
        default="../data/processed/ml_dataset.parquet",
        help="Path do parquet final"
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Número de linhas de exemplo"
    )

    args = parser.parse_args()

    file_path = Path(args.file)

    print("=" * 80)
    print("FINAL DATASET INSPECTION")
    print("=" * 80)

    if not file_path.exists():
        print(f"\nERROR: arquivo não encontrado:\n{file_path}")
        return

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    df = pd.read_parquet(file_path)

    print(f"File     : {file_path}")
    print(f"Rows     : {len(df):,}")
    print(f"Columns  : {len(df.columns)}")
    print(
        f"Memory   : "
        f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )

    # -------------------------------------------------------------------------
    # COLUMN LIST
    # -------------------------------------------------------------------------

    print("\n")
    print_separator()
    print("COLUMNS")
    print_separator()

    for col in df.columns:

        dtype = str(df[col].dtype)

        null_pct = (df[col].isna().mean()) * 100

        unique = df[col].nunique(dropna=True)

        print(
            f"{col:<35} "
            f"{dtype:<15} "
            f"nulls={null_pct:6.2f}% "
            f"unique={unique}"
        )

    # -------------------------------------------------------------------------
    # SAMPLE ROWS
    # -------------------------------------------------------------------------

    print("\n")
    print_separator()
    print("SAMPLE ROWS")
    print_separator()

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.max_colwidth",
        40
    ):
        print(df.head(args.rows))

    # -------------------------------------------------------------------------
    # BASIC STATISTICS
    # -------------------------------------------------------------------------

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if numeric_cols:

        print("\n")
        print_separator()
        print("NUMERIC STATISTICS")
        print_separator()

        stats = df[numeric_cols].describe().T

        with pd.option_context(
            "display.max_rows",
            200,
            "display.max_columns",
            None,
            "display.width",
            200
        ):
            print(stats)

    # -------------------------------------------------------------------------
    # NULL ANALYSIS
    # -------------------------------------------------------------------------

    print("\n")
    print_separator()
    print("NULL ANALYSIS")
    print_separator()

    nulls = df.isnull().sum()

    nulls = nulls[nulls > 0].sort_values(ascending=False)

    if len(nulls) == 0:
        print("No null values.")
    else:

        for col, count in nulls.items():

            pct = 100 * count / len(df)

            print(f"{col:<35} {count:>10} ({pct:6.2f}%)")

    # -------------------------------------------------------------------------
    # DUPLICATE CHECK
    # -------------------------------------------------------------------------

    print("\n")
    print_separator()
    print("DUPLICATE CHECK")
    print_separator()

    total_duplicates = df.duplicated().sum()

    print(f"Duplicate rows: {total_duplicates}")

    if "ticker" in df.columns:
        print(f"Duplicated tickers: {df['ticker'].duplicated().sum()}")

    # -------------------------------------------------------------------------
    # DATE RANGE
    # -------------------------------------------------------------------------

    possible_date_cols = [
        "date",
        "reference_date",
        "trading_date",
        "datetime"
    ]

    for col in possible_date_cols:

        if col in df.columns:

            print("\n")
            print_separator()
            print(f"DATE RANGE ({col})")
            print_separator()

            print(f"Min date: {df[col].min()}")
            print(f"Max date: {df[col].max()}")

            break

    # -------------------------------------------------------------------------
    # TICKERS
    # -------------------------------------------------------------------------

    if "ticker" in df.columns:

        print("\n")
        print_separator()
        print("TICKERS")
        print_separator()

        print(df["ticker"].value_counts())

    # -------------------------------------------------------------------------
    # SECTORS
    # -------------------------------------------------------------------------

    if "sector" in df.columns:

        print("\n")
        print_separator()
        print("SECTORS")
        print_separator()

        print(df["sector"].value_counts())

    # -------------------------------------------------------------------------
    # DATAFRAME INFO
    # -------------------------------------------------------------------------

    print("\n")
    print_separator()
    print("DATAFRAME INFO")
    print_separator()

    print(df.info())

    print("\n")
    print("=" * 80)
    print("INSPECTION FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()