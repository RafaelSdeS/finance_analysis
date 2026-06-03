#!/usr/bin/env python3

"""
test_ticker_data.py
==============

Inspeciona um ticker específico dentro do dataset processado.

Uso:
    python test_ticker.py
    python test_ticker.py --ticker VALE3
    python test_ticker.py --ticker PETR4
"""

from pathlib import Path
import argparse
import pandas as pd

# =============================================================================
# PANDAS SETTINGS
# =============================================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.2f}".format)

# =============================================================================
# ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(
    description="Inspect a ticker from ml_dataset.parquet"
)

parser.add_argument(
    "--ticker",
    default="ABEV3",
    help="Ticker symbol (default: ABEV3)"
)

args = parser.parse_args()
ticker = args.ticker.upper()

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_dataset.parquet"
)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}\n"
    )

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_parquet(DATASET_PATH)

if "ticker" not in df.columns:
    raise ValueError(
        "Column 'ticker' not found in dataset."
    )

data = df[df["ticker"] == ticker].copy()

if data.empty:
    print(f"\nTicker '{ticker}' not found.")
    print(
        "\nAvailable examples:"
    )
    print(
        sorted(df["ticker"].dropna().unique())[:20]
    )
    raise SystemExit(1)

# =============================================================================
# HEADER
# =============================================================================

print()
print("=" * 80)
print(f"TICKER INSPECTION: {ticker}")
print("=" * 80)

print(f"Dataset : {DATASET_PATH}")
print(f"Rows    : {len(data):,}")

# =============================================================================
# 1. IDENTITY & COMPANY INFO
# =============================================================================

print()
print("=" * 80)
print("1. IDENTITY & COMPANY INFO")
print("=" * 80)

info_cols = [
    "ticker",
    "corporate_name",
    "trade_name",
    "sector",
    "cvm_code",
    "cnpj",
    "status",
]

info_cols = [c for c in info_cols if c in data.columns]

if info_cols:
    print(
        data[info_cols]
        .drop_duplicates()
        .to_string(index=False)
    )
else:
    print("No company info columns found.")

# =============================================================================
# 2. DAILY PRICE SAMPLE
# =============================================================================

print()
print("=" * 80)
print("2. DAILY PRICES (first 5 + last 5)")
print("=" * 80)

price_cols = [
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "traded_amount",
]

price_cols = [c for c in price_cols if c in data.columns]

if price_cols:
    sample = pd.concat(
        [data.head(5), data.tail(5)]
    )

    print(
        sample[price_cols]
        .to_string(index=False)
    )
else:
    print("No price columns found.")

# =============================================================================
# 3. FUNDAMENTALS
# =============================================================================

print()
print("=" * 80)
print("3. FUNDAMENTALS PER QUARTER")
print("=" * 80)

fund_cols = [
    "reference_date",
    "close_price",
    "market_cap",
    "shares_outstanding",
    "pl",
    "pvp",
    "roe",
    "roa",
    "roic",
    "net_margin",
    "ebitda_margin",
    "net_debt_ebitda",
    "debt_equity",
    "net_revenue",
    "net_income",
    "ebitda",
    "ebit",
    "total_assets",
    "equity",
    "total_debt",
    "net_debt",
    "cash",
    "cagr_revenue_5y",
    "cagr_earnings_5y",
]

fund_cols = [c for c in fund_cols if c in data.columns]

if fund_cols and "reference_date" in data.columns:

    quarters = (
        data
        .sort_values("reference_date")
        .drop_duplicates(
            subset=["reference_date"]
        )
    )

    print(
        quarters[fund_cols]
        .to_string(index=False)
    )

else:
    print("No fundamentals found.")

# =============================================================================
# 4. NULL CHECK
# =============================================================================

print()
print("=" * 80)
print("4. NULL CHECK")
print("=" * 80)

null_counts = data.isnull().sum()
null_counts = null_counts[null_counts > 0]

if null_counts.empty:
    print("No nulls found.")
else:
    print(null_counts.to_string())

# =============================================================================
# 5. DATE RANGE
# =============================================================================

print()
print("=" * 80)
print("5. DATE RANGE")
print("=" * 80)

if "trade_date" in data.columns:
    print(
        f"trade_date    : "
        f"{data['trade_date'].min().date()} "
        f"→ "
        f"{data['trade_date'].max().date()}"
    )

if "reference_date" in data.columns:

    ref_dates = data["reference_date"].dropna()

    if not ref_dates.empty:
        print(
            f"reference_date: "
            f"{ref_dates.min().date()} "
            f"→ "
            f"{ref_dates.max().date()}"
        )

        print(
            f"Unique quarters: "
            f"{ref_dates.nunique()}"
        )

# =============================================================================
# 6. LATEST QUARTER SNAPSHOT
# =============================================================================

if "reference_date" in data.columns:

    latest_ref = data["reference_date"].max()

    latest = data[
        data["reference_date"] == latest_ref
    ].iloc[0]

    print()
    print("=" * 80)
    print("6. LATEST QUARTER SNAPSHOT")
    print("=" * 80)

    snapshot_cols = [
        "reference_date",
        "close_price",
        "market_cap",
        "shares_outstanding",
        "pl",
        "pvp",
        "roe",
        "roa",
        "net_margin",
        "ebitda_margin",
        "net_revenue",
        "net_income",
        "ebitda",
        "ebit",
        "equity",
        "total_assets",
        "total_debt",
        "net_debt",
        "cash",
    ]

    snapshot_cols = [
        c for c in snapshot_cols
        if c in data.columns
    ]

    print(
        latest[snapshot_cols]
        .to_frame("value")
        .to_string()
    )

print()
print("=" * 80)
print("DONE")
print("=" * 80)