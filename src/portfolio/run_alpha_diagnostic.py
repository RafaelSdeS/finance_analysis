"""
run_alpha_diagnostic.py -- proposal Phase 2.6 "Done when" deliverable: runs
the LightGBM walk-forward forecaster on the REAL point-in-time liquid
universe, reports out-of-sample rank-IC, and compares a simple alpha-ranked
(top-half, equal-weight -- "still equal-ish, pre-optimizer") strategy
against the Phase 2.3 equal-weight floor, same universe/dates/costs.

Run: python -m src.portfolio.run_alpha_diagnostic [--top-n 50] [--horizon-td 252]
"""

import argparse
import pprint

import pandas as pd

from src.build_dataset.paths import OUTPUT_PATH
from src.portfolio import alpha, universe
from src.portfolio.backtest import equal_weight_fn, run_backtest
from src.portfolio.features import feature_columns
from src.portfolio.labels import forward_excess_return
from src.portfolio.metrics import full_report


def make_alpha_weighted_fn(preds_by_date: dict, top_frac: float = 0.5):
    """Falls back to equal-weight on any date without a prediction yet
    (early history, before the model has min_train_rows)."""
    def weights_fn(date, uni, state):
        preds = preds_by_date.get(date, {})
        ranked = sorted((t for t in uni if t in preds), key=lambda t: preds[t], reverse=True)
        if not ranked:
            n = len(uni)
            return {t: 1.0 / n for t in uni} if n else {}
        chosen = ranked[: max(1, int(len(ranked) * top_frac))]
        w = 1.0 / len(chosen)
        return {t: w for t in chosen}
    return weights_fn


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
    print(f"Walk-forward training/prediction across {len(reb_dates)} rebalance dates...")
    preds = alpha.walk_forward_predict(df, reb_dates, horizon_td=horizon_td, n_estimators=200)
    print(f"  {preds['date'].nunique()} dates produced a prediction "
          f"(first {len(reb_dates) - preds['date'].nunique()} skipped -- not enough training history yet)")

    ic = alpha.rank_ic(preds, df)
    print("\n=== Out-of-sample rank-IC ===")
    print(f"mean={ic.mean():.4f}  median={ic.median():.4f}  "
          f"fraction of dates with IC>0={float((ic > 0).mean()):.2f}  n_dates={ic.notna().sum()}")

    preds_by_date = {
        d: g.set_index("ticker")["alpha"].to_dict() for d, g in preds.groupby("date")
    }
    alpha_fn = make_alpha_weighted_fn(preds_by_date, top_frac=0.5)

    cdi = pd.read_parquet(OUTPUT_PATH, columns=["trade_date", "cdi"]).drop_duplicates().sort_values("trade_date")
    prices = df[["ticker", "trade_date", "adj_close"]]

    print("\nRunning equal-weight baseline (same universe/dates/costs)...")
    eq_curve, eq_log = run_backtest(prices, cdi, membership, equal_weight_fn)
    print("Running alpha-weighted (top-half, equal-weight) strategy...")
    alpha_curve, alpha_log = run_backtest(prices, cdi, membership, alpha_fn)

    print("\n=== Equal-weight baseline ===")
    pprint.pprint(full_report(eq_curve, eq_log))
    print("\n=== Alpha-weighted (top-half) ===")
    pprint.pprint(full_report(alpha_curve, alpha_log))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--horizon-td", type=int, default=252)
    args = parser.parse_args()
    main(top_n=args.top_n, horizon_td=args.horizon_td)
