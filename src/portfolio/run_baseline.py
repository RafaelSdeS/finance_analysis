"""
run_baseline.py -- proposal Phase 2.3 "Done when" deliverable: runs the
equal-weight baseline (on the point-in-time liquid universe) plus the
BOVA11 buy-and-hold and 100%-CDI benchmarks on the REAL dataset, and prints
the full §8 metric panel for each. Proves the harness end-to-end before any
α/Σ/optimizer work (Phase 2.4+) gets layered on top of it.

Run: python -m src.portfolio.run_baseline [--top-n 50]
"""

import argparse

import pandas as pd

from src.build_dataset.paths import MACRO_DIR, OUTPUT_PATH, PRICES_DIR
from src.portfolio import universe
from src.portfolio.backtest import buy_and_hold_curve, cdi_curve, equal_weight_fn, run_backtest
from src.portfolio.metrics import full_report, print_report

_NO_REBALANCE_LOG = pd.DataFrame({"turnover": [0.0]})


def main(top_n: int = 50):
    print("Loading dataset...")
    df = pd.read_parquet(OUTPUT_PATH, columns=["ticker", "trade_date", "adj_close", "traded_amount", "cdi"])
    cdi = df[["trade_date", "cdi"]].drop_duplicates().sort_values("trade_date")
    prices = df[["ticker", "trade_date", "adj_close"]]

    print(f"Building point-in-time liquid universe (top_n={top_n})...")
    membership = universe.liquid_universe(df[["ticker", "trade_date", "traded_amount"]], top_n=top_n)

    print("Running equal-weight backtest...")
    eq_curve, eq_log = run_backtest(prices, cdi, membership, equal_weight_fn)
    print(f"  {len(eq_curve)} daily observations, {eq_curve.index.min().date()} -> {eq_curve.index.max().date()}")

    print("Loading BOVA11 benchmark...")
    bova = pd.read_parquet(PRICES_DIR / "BOVA11.parquet", columns=["trade_date", "adj_close"])
    bova_series = bova.set_index("trade_date")["adj_close"].reindex(eq_curve.index).ffill()
    bova_curve = buy_and_hold_curve(bova_series)

    print("Building 100% CDI curve...")
    cdi_in_range = cdi[cdi["trade_date"].isin(eq_curve.index)]
    cdi_only_curve = cdi_curve(cdi_in_range)

    selic = pd.read_parquet(MACRO_DIR / "selic.parquet").rename(columns={"reference_date": "trade_date"})
    selic_daily = selic.set_index("trade_date")["selic"]

    for name, curve, log in (
        ("Equal-weight (liquid universe)", eq_curve, eq_log),
        ("BOVA11 buy-and-hold", bova_curve, _NO_REBALANCE_LOG),
        ("100% CDI", cdi_only_curve, _NO_REBALANCE_LOG),
    ):
        print_report(name, full_report(curve, log, selic_daily=selic_daily))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()
    main(top_n=args.top_n)
