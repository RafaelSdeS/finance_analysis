"""
Feature Engineering for ML Agent Training

Handles missing feature computation and data preparation.
"""

import pandas as pd
import numpy as np


MAX_ABS_LOG_RETURN = 1.0  # |log r| > 1.0 (±172%/day) on B3 = data error or untradeable event


def compute_returns(df: pd.DataFrame, price_col: str = "adj_close", ticker_col: str = "ticker") -> pd.DataFrame:
    """
    Compute log returns from split-adjusted prices (grouped by ticker).

    Args:
        df: DataFrame with prices
        price_col: Column for prices — must be split-adjusted (default: "adj_close").
                   Raw 'close' produces massive fake returns on split days.
        ticker_col: Column name for ticker symbols (default: "ticker")

    Returns:
        DataFrame with 'returns' column added (log returns per ticker)

    Data-cleaning rules (corrupt observations → NaN, never imputed):
        - Non-positive prices (adj_close <= 0 exists in a few tickers) → NaN price
        - |log return| > MAX_ABS_LOG_RETURN → NaN (split residue / feed glitches;
          ~0.04% of rows). Downstream (env) treats NaN returns as 0.
        - First row per ticker is NaN (no previous price) — expected.
    """
    df = df.copy()

    print(f"Computing log returns from '{price_col}' (grouped by '{ticker_col}')...")

    prices = df[price_col].where(df[price_col] > 0)  # non-positive → NaN
    df["returns"] = np.log(prices / prices.groupby(df[ticker_col]).shift(1))

    corrupt = df["returns"].abs() > MAX_ABS_LOG_RETURN
    df.loc[corrupt, "returns"] = np.nan

    nan_count = df["returns"].isnull().sum()
    print(f"✓ Computed returns:")
    print(f"  Valid: {df['returns'].notna().sum():,} rows")
    print(f"  Corrupt (|log r| > {MAX_ABS_LOG_RETURN}) → NaN: {corrupt.sum():,} rows")
    print(f"  Total NaN: {nan_count:,} (~{nan_count/len(df)*100:.2f}%)")
    print(f"  Mean: {df['returns'].mean():.6f}")
    print(f"  Std:  {df['returns'].std():.6f}")
    print(f"  Range: [{df['returns'].min():.4f}, {df['returns'].max():.4f}]")

    return df


def prepare_training_dataset(
    dataset_path: str = "data/processed/ml_dataset.parquet",
    output_path: str = "data/processed/ml_dataset_training.parquet",
    force_recompute: bool = False
) -> pd.DataFrame:
    """
    Load and prepare dataset for training (ensure returns are computed).

    Args:
        dataset_path: Path to raw ml_dataset.parquet
        output_path: Path to save prepared dataset
        force_recompute: If True, recompute returns even if they exist

    Returns:
        Prepared DataFrame ready for training
    """
    print("=" * 70)
    print("PREPARING DATASET FOR TRAINING")
    print("=" * 70)

    print(f"\nLoading {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Always recompute: returns definition may change (e.g., close → adj_close fix)
    df = compute_returns(df)

    # Save prepared dataset
    print(f"\nSaving to {output_path}...")
    df.to_parquet(output_path, index=False, compression="snappy")
    print(f"✓ Saved prepared dataset: {len(df):,} rows, {len(df.columns)} columns")

    return df


if __name__ == "__main__":
    # One-time preparation
    df = prepare_training_dataset()
