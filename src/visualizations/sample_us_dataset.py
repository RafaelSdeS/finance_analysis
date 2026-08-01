#!/usr/bin/env python3
"""Subset the US dataset to its top-100-by-traded-volume tickers, for viewing
in Data Wrangler (the full 5.5GB/15.4M-row parquet crashes it).

Ranking is a single total-traded_amount snapshot, not a point-in-time
rebalance like build_top50_universe.py — fine for a look-and-see subset,
not for training. Use build_top50_universe.py's approach if this needs to
become a real no-lookahead universe.
"""
import pyarrow.parquet as pq
from pathlib import Path

from src.build_dataset.paths import US_OUTPUT_PATH

TOP_N = 100
project_root = Path(__file__).resolve().parents[2]
sample_path = project_root / "data/processed/us_ml_dataset_top100.parquet"

print("Ranking tickers by total traded_amount (columnar read, low memory)...")
vol_table = pq.read_table(US_OUTPUT_PATH, columns=["ticker", "traded_amount"])
vol_df = vol_table.to_pandas()
top_tickers = (
    vol_df.groupby("ticker")["traded_amount"].sum().nlargest(TOP_N).index.tolist()
)
del vol_table, vol_df

print(f"Reading full rows for top {TOP_N} tickers only (predicate pushdown)...")
table = pq.read_table(US_OUTPUT_PATH, filters=[("ticker", "in", top_tickers)])
df = table.to_pandas().sort_values(["ticker", "trade_date"])

print(f"Subset: {len(df):,} rows, {df['ticker'].nunique()} tickers, {len(df.columns)} columns")
print(f"Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")

df.to_parquet(sample_path, index=False)
print(f"\n✓ Saved to {sample_path.relative_to(project_root)}")
print(f"  Size: {sample_path.stat().st_size / 1e6:.1f} MB")
