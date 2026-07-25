"""
run_full_backtest.py -- proposal Phase 2.7 "Done when" deliverable: the
full predict-then-optimize pipeline (alpha -> Sigma -> optimizer) wired end
to end through the backtest harness, on the real dataset. Reports the full
§8 panel including the cost-sensitivity curve, vs. all three baselines.

Run: python -m src.portfolio.run_full_backtest [--top-n 50] [--horizon-td 252]
"""

import argparse
import pprint

import pandas as pd

from src.build_dataset.paths import MACRO_DIR, OUTPUT_PATH, PRICES_DIR
from src.portfolio import alpha, universe
from src.portfolio.backtest import buy_and_hold_curve, cdi_curve, equal_weight_fn, run_backtest
from src.portfolio.features import feature_columns
from src.portfolio.labels import forward_excess_return
from src.portfolio.metrics import full_report
from src.portfolio.pipeline import make_full_weights_fn

ONE_WAY_COST_SWEEP = [0.0003, 0.0015, 0.003]  # 0.03% floor -> 0.15% -> 0.3%
_NO_REBALANCE_LOG = pd.DataFrame({"turnover": [0.0]})


def main(top_n: int = 50, horizon_td: int = 252):
    print("Loading dataset (full feature set)...")
    base_cols = ["ticker", "trade_date", "adj_close", "traded_amount"]
    all_cols = sorted(set(base_cols) | set(feature_columns(include_sector=False)))
    df = pd.read_parquet(OUTPUT_PATH, columns=all_cols)

    print(f"Building point-in-time liquid universe (top_n={top_n})...")
    membership = universe.liquid_universe(df[["ticker", "trade_date", "traded_amount"]], top_n=top_n)
    df = universe.restrict_to_universe(df, membership)
    print(f"  restricted to {len(df)} rows, {df['ticker'].nunique()} tickers ever in the universe")

    print(f"Building the {horizon_td}-day forward-excess-return label...")
    df["label"] = forward_excess_return(df, horizon_td=horizon_td)

    reb_dates = universe.rebalance_dates(membership)
    print(f"Walk-forward alpha training/prediction across {len(reb_dates)} rebalance dates "
          "(reused unchanged across the cost sweep below -- alpha doesn't depend on cost)...")
    preds = alpha.walk_forward_predict(df, reb_dates, horizon_td=horizon_td, n_estimators=200)
    preds_by_date = {d: g.set_index("ticker")["alpha"].to_dict() for d, g in preds.groupby("date")}
    print(f"  {len(preds_by_date)} dates produced a prediction")

    prices = df[["ticker", "trade_date", "adj_close"]]
    price_wide = prices.pivot(index="trade_date", columns="ticker", values="adj_close")  # NOT ffilled -- see pipeline.py
    cdi = pd.read_parquet(OUTPUT_PATH, columns=["trade_date", "cdi"]).drop_duplicates().sort_values("trade_date")
    selic = pd.read_parquet(MACRO_DIR / "selic.parquet").rename(columns={"reference_date": "trade_date"})
    selic_daily = selic.set_index("trade_date")["selic"]

    print("\nRunning equal-weight baseline...")
    eq_curve, eq_log = run_backtest(prices, cdi, membership, equal_weight_fn)

    print("Loading BOVA11 / 100% CDI baselines...")
    bova = pd.read_parquet(PRICES_DIR / "BOVA11.parquet", columns=["trade_date", "adj_close"])
    bova_curve = buy_and_hold_curve(bova.set_index("trade_date")["adj_close"].reindex(eq_curve.index).ffill())
    cdi_only_curve = cdi_curve(cdi[cdi["trade_date"].isin(eq_curve.index)])

    print("\n=== Equal-weight baseline ===")
    pprint.pprint(full_report(eq_curve, eq_log, selic_daily=selic_daily))
    print("\n=== BOVA11 buy-and-hold ===")
    pprint.pprint(full_report(bova_curve, _NO_REBALANCE_LOG, selic_daily=selic_daily))
    print("\n=== 100% CDI ===")
    pprint.pprint(full_report(cdi_only_curve, _NO_REBALANCE_LOG, selic_daily=selic_daily))

    print("\n=== Full pipeline (alpha -> Sigma -> optimizer): cost-sensitivity curve ===")
    for c1 in ONE_WAY_COST_SWEEP:
        weights_fn = make_full_weights_fn(preds_by_date, price_wide, sigma_window=252,
                                           horizon_td=horizon_td, c1=c1, lam=1.0, w_max=0.1)
        curve, log = run_backtest(prices, cdi, membership, weights_fn, one_way_cost=c1)
        report = full_report(curve, log, selic_daily=selic_daily)
        print(f"\n-- one-way c1={c1:.4%} (round-trip equivalent {2 * c1:.4%}) --")
        pprint.pprint(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--horizon-td", type=int, default=252)
    args = parser.parse_args()
    main(top_n=args.top_n, horizon_td=args.horizon_td)
