"""
inspect_petr4.py
================
Shows the most important columns for PETR4
to confirm the dataset is correct.

Uso:
    python inspect_petr4.py
"""

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.2f}".format)

OUTPUT_PATH = "../data/processed/ml_dataset.parquet"

df = pd.read_parquet(OUTPUT_PATH)
petr = df[df["ticker"] == "PETR4"].copy()

# =============================================================================
# 1. IDENTITY & COMPANY INFO
# =============================================================================

print()
print("=" * 80)
print("1. IDENTITY & COMPANY INFO")
print("=" * 80)

info_cols = [
    "ticker", "corporate_name", "trade_name",
    "sector", "cvm_code", "cnpj", "status",
]
info_cols = [c for c in info_cols if c in petr.columns]
print(petr[info_cols].drop_duplicates().to_string(index=False))


# =============================================================================
# 2. PRICE SAMPLE (first and last 5 rows)
# =============================================================================

print()
print("=" * 80)
print("2. DAILY PRICES (first 5 + last 5)")
print("=" * 80)

price_cols = [
    "trade_date", "open", "high", "low", "close",
    "adjusted_close", "volume", "traded_amount",
]
price_cols = [c for c in price_cols if c in petr.columns]
sample = pd.concat([petr.head(5), petr.tail(5)])[price_cols]
print(sample.to_string(index=False))


# =============================================================================
# 3. FUNDAMENTALS (one row per unique reference_date)
# =============================================================================

print()
print("=" * 80)
print("3. FUNDAMENTALS PER QUARTER")
print("=" * 80)

fund_cols = [
    "reference_date",
    "close_price", "market_cap", "shares_outstanding",
    "pl", "pvp", "roe", "roa", "roic",
    "net_margin", "ebitda_margin",
    "net_debt_ebitda", "debt_equity",
    "net_revenue", "net_income", "ebitda", "net_debt", "cash",
    "cagr_revenue_5y", "cagr_earnings_5y",
]
fund_cols = [c for c in fund_cols if c in petr.columns]
quarters = petr.drop_duplicates(subset=["reference_date"])[fund_cols]
print(quarters.to_string(index=False))


# =============================================================================
# 4. NULL CHECK
# =============================================================================

print()
print("=" * 80)
print("4. NULL CHECK")
print("=" * 80)

null_counts = petr.isnull().sum()
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

print(f"trade_date    : {petr['trade_date'].min().date()} → {petr['trade_date'].max().date()}")
if "reference_date" in petr.columns:
    print(f"reference_date: {petr['reference_date'].min().date()} → {petr['reference_date'].max().date()}")
    print(f"Unique quarters: {petr['reference_date'].nunique()}")