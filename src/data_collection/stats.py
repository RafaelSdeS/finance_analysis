"""
stats.py — post-collection data audit.

Usage:
    python -m src.data_collection.stats
    python -m src.data_collection.stats --mode full_scale
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config

STALE_DAYS = 7


def _date_col(df: pd.DataFrame) -> str | None:
    for col in ("trade_date", "reference_date", "date"):
        if col in df.columns:
            return col
    return None


def _stats_line(path: Path) -> str:
    df = pd.read_parquet(path)
    rows = len(df)
    dcol = _date_col(df)
    if dcol:
        dates = pd.to_datetime(df[dcol])
        lo, hi = dates.min().date(), dates.max().date()
        age = (datetime.now(timezone.utc).date() - hi).days
        stale = "  ⚠ STALE" if age > STALE_DAYS else ""
        nulls = df.drop(columns=[dcol]).isnull().sum().sum()
        return f"  {path.stem:<12} {rows:>6} rows  {lo} → {hi}  nulls={nulls}{stale}"
    else:
        # company_info: no date column
        nulls = df.isnull().sum().sum()
        return f"  {path.stem:<12} {rows:>6} rows  nulls={nulls}"


def print_stats():
    sections = [
        ("macro",        config.MACRO_DIR.glob("*.parquet")),
        ("company_info", config.COMPANY_DIR.glob("*.parquet")),
        ("prices",       sorted(config.PRICES_DIR.glob("*.parquet"))),
        ("fundamentals", sorted(config.FUND_DIR.glob("*.parquet"))),
    ]
    for title, files in sections:
        files = list(files)
        if not files:
            print(f"\n{title}: (no files)")
            continue
        print(f"\n{title}:")
        for f in files:
            print(_stats_line(f))


def main():
    p = argparse.ArgumentParser(description="Audit collected raw data")
    p.add_argument("--mode", choices=["prototype", "full_scale"], default="prototype",
                   help="Informational only — stats always scan all files in data/raw/")
    args = p.parse_args()
    print(f"=== data audit  mode={args.mode}  {datetime.now():%Y-%m-%d %H:%M} ===")
    print_stats()
    print()


if __name__ == "__main__":
    main()
