"""
inspect_all_data.py
===================

Scans all folders inside:

data/raw/

and displays:
- available parquet files
- shape
- columns
- date ranges
- sample rows

Folders scanned:
----------------
data/raw/prices/
data/raw/fundamentals/
data/raw/financials/
data/raw/macro/

Usage
-----
python inspect_all_data.py

Optional
--------
python inspect_all_data.py --rows 3
python inspect_all_data.py --stats
"""

from pathlib import Path
import argparse

import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

RAW_DIR = Path("../data/raw")

FOLDERS = [
    "prices",
    "fundamentals",
    "financials",
    "macro",
]


# =============================================================================
# HELPERS
# =============================================================================

def print_title(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_subtitle(title):

    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def detect_date_column(df):

    possible = [
        "trade_date",
        "reference_date",
        "date",
    ]

    for col in possible:
        if col in df.columns:
            return col

    return None


def inspect_file(path: Path, rows: int, show_stats: bool):

    print_subtitle(path.name)

    try:

        df = pd.read_parquet(path)

    except Exception as e:

        print(f"ERROR reading parquet: {e}")

        return

    # -------------------------------------------------------------------------
    # BASIC INFO
    # -------------------------------------------------------------------------

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {df.shape[1]}")

    memory_mb = (
        df.memory_usage(deep=True).sum()
        / (1024 * 1024)
    )

    print(f"Memory  : {memory_mb:.2f} MB")

    # -------------------------------------------------------------------------
    # DATE RANGE
    # -------------------------------------------------------------------------

    date_col = detect_date_column(df)

    if date_col:

        try:

            dates = pd.to_datetime(df[date_col])

            print(
                f"Date range: "
                f"{dates.min().date()} → {dates.max().date()}"
            )

        except:
            pass

    # -------------------------------------------------------------------------
    # COLUMNS
    # -------------------------------------------------------------------------

    print("\nColumns:")

    for col in df.columns:

        dtype = str(df[col].dtype)

        null_pct = df[col].isnull().mean() * 100

        print(
            f"  {col:<35} "
            f"{dtype:<15} "
            f"nulls={null_pct:>6.2f}%"
        )

    # -------------------------------------------------------------------------
    # SAMPLE
    # -------------------------------------------------------------------------

    print("\nSample rows:")

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
    ):

        print(df.head(rows).to_string())

    # -------------------------------------------------------------------------
    # STATS
    # -------------------------------------------------------------------------

    if show_stats:

        numeric = df.select_dtypes(include="number")

        if not numeric.empty:

            print("\nNumeric statistics:")

            stats = numeric.describe().T

            print(stats.to_string())


# =============================================================================
# MAIN
# =============================================================================

def inspect_all(rows: int, show_stats: bool):

    print_title("BOLSAI RAW DATA INSPECTION")

    if not RAW_DIR.exists():

        print(f"Folder not found: {RAW_DIR}")

        return

    # =========================================================================
    # LOOP THROUGH FOLDERS
    # =========================================================================

    for folder_name in FOLDERS:

        folder = RAW_DIR / folder_name

        print_title(f"FOLDER: {folder}")

        if not folder.exists():

            print("Folder does not exist.")

            continue

        parquet_files = sorted(folder.glob("*.parquet"))

        if not parquet_files:

            print("No parquet files found.")

            continue

        print(f"Files found: {len(parquet_files)}")

        # ---------------------------------------------------------------------
        # EACH FILE
        # ---------------------------------------------------------------------

        for file_path in parquet_files:

            inspect_file(
                path=file_path,
                rows=rows,
                show_stats=show_stats,
            )


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rows",
        type=int,
        default=3,
        help="Rows to display"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show numeric statistics"
    )

    args = parser.parse_args()

    inspect_all(
        rows=args.rows,
        show_stats=args.stats,
    )