"""
Feature Engineering for ML Agent Training

Handles missing feature computation and data preparation.
"""

import pandas as pd
import numpy as np

MAX_ABS_LOG_RETURN = 1.0  # |log r| > 1.0 (±172%/day) on B3 = data error or untradeable event
CASH_ANNUALIZED_RANGE = (0.01, 0.30)  # sane historical SELIC bounds (~2%-26% seen 2000-2026)
FUNDAMENTAL_FEATURES = [
    "pl",
    "pvp",
    "roe",
    "debt_equity",
    "roic",
    "roa",
    "net_margin",
    "gross_margin",
    "ebitda_margin",
    "current_ratio",
    "cash_ratio",
    "earnings_growth_yoy",
    "revenue_growth_yoy",
    "ebitda_growth_yoy",
    "has_fundamentals",
]


def synthesize_cash_asset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create synthetic CASH rows (one per trading date) representing a risk-free
    SELIC-earning position. CASH is always active, giving the agent an option
    to go to cash on every step without forcing full equity investment.

    The synthetic price index compounds at the daily SELIC rate (selic as %/day).
    """
    # Extract one macro row per date (SELIC/CDI/IPCA are date-level constants)
    macro = (
        df[["trade_date", "selic", "cdi", "ipca"]]
        .drop_duplicates("trade_date")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    # Synthetic price index: starts at 100, compounds daily at SELIC rate
    # selic column is daily %; convert to fraction for compounding
    price = 100.0 * (1.0 + macro["selic"] / 100.0).cumprod()

    # Build CASH rows: one per date, all 23 state features set
    cash = pd.DataFrame({
        "ticker": "CASH",
        "trade_date": macro["trade_date"],
        "open": price, "high": price, "low": price,
        "close": price, "adj_open": price, "adj_high": price,
        "adj_low": price, "adj_close": price,
        "volume": 0.0,
        "sector": "Cash",
        "selic": macro["selic"], "cdi": macro["cdi"], "ipca": macro["ipca"],
    })

    # Fundamentals are not applicable to risk-free cash; set to 0
    # (StandardScaler treats zero-variance columns as scale_=1.0 in fit_train_scaler)
    for col in FUNDAMENTAL_FEATURES:
        cash[col] = 0.0

    return cash


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
    if len(df) > 0:
        print(f"  Total NaN: {nan_count:,} (~{nan_count/len(df)*100:.2f}%)")
        print(f"  Mean: {df['returns'].mean():.6f}")
        print(f"  Std:  {df['returns'].std():.6f}")
        print(f"  Range: [{df['returns'].min():.4f}, {df['returns'].max():.4f}]")
    else:
        print(f"  Total NaN: 0 (empty dataset)")

    return df


def prepare_training_dataset(
    dataset_path: str = "data/processed/ml_dataset.parquet",
    output_path: str = "data/processed/ml_dataset_training.parquet",
    force_recompute: bool = False
) -> pd.DataFrame:
    """
    Load and prepare dataset for training (ensure returns are computed).
    Adds synthetic CASH asset representing a risk-free SELIC-earning position.

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

    # Synthesize CASH asset (risk-free SELIC-earning position, always active)
    print("\nSynthesizing CASH asset (compounded daily at SELIC rate)...")
    cash_df = synthesize_cash_asset(df)
    df = pd.concat([df, cash_df], ignore_index=True)
    print(f"✓ Added {len(cash_df):,} CASH rows (one per trading date)")

    # Always recompute: returns definition may change (e.g., close → adj_close fix)
    df = compute_returns(df)

    # Sanity check: CASH annualized return should be in the expected range
    cash_returns = df[df["ticker"] == "CASH"]["returns"]
    cash_annual = (1.0 + cash_returns).prod() ** (252 / len(cash_returns)) - 1.0
    if not (CASH_ANNUALIZED_RANGE[0] <= cash_annual <= CASH_ANNUALIZED_RANGE[1]):
        raise ValueError(
            f"CASH annualized return {cash_annual:.2%} out of expected range "
            f"{CASH_ANNUALIZED_RANGE[0]:.1%}-{CASH_ANNUALIZED_RANGE[1]:.1%}. "
            f"Unit error likely (selic should be daily %, not annualized)."
        )
    print(f"✓ CASH annualized return: {cash_annual:.2%} ✓")

    # Save prepared dataset
    print(f"\nSaving to {output_path}...")
    df.to_parquet(output_path, index=False, compression="snappy")
    print(f"✓ Saved prepared dataset: {len(df):,} rows, {len(df.columns)} columns")

    return df


if __name__ == "__main__":
    # One-time preparation
    df = prepare_training_dataset()
