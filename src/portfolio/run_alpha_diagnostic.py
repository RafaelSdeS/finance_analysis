"""
run_alpha_diagnostic.py -- proposal Phase 2.6 "Done when" deliverable: runs
the LightGBM walk-forward forecaster on the REAL point-in-time liquid
universe, reports out-of-sample rank-IC, and compares a simple alpha-ranked
(top-half, equal-weight -- "still equal-ish, pre-optimizer") strategy
against the Phase 2.3 equal-weight floor, same universe/dates/costs.

Run: python -m src.portfolio.run_alpha_diagnostic [--top-n 50] [--horizon-td 252]
"""

import argparse

import pandas as pd

from src.build_dataset.paths import OUTPUT_PATH
from src.portfolio import alpha, universe
from src.portfolio.backtest import equal_weight_fn, run_backtest
from src.portfolio.features import feature_columns
from src.portfolio.labels import forward_excess_return
from src.portfolio.metrics import full_report, print_report


def make_alpha_weighted_fn(preds_by_date: dict, top_frac: float = 0.5, hold_frac: float | None = None):
    """The honest baseline bar (plan Phase 1.1): quintile/top-half sort,
    equal-weight, PLUS a no-trade band -- a name already held stays as long
    as it's still within the looser `hold_frac` cut, not just the tighter
    `top_frac` entry cut. Without this, a name bouncing between rank 49 and
    51 out of 100 gets fully sold and rebought every quarter on pure ranking
    noise near the cutoff -- exactly the churn a Buffett-shaped "own the
    best, hold" bar should not have. `hold_frac=None` (default) reproduces
    the pre-band behavior (hold_frac == top_frac, no slack).
    Falls back to equal-weight on any date without a prediction yet (early
    history, before the model has min_train_rows)."""
    hold_frac = top_frac if hold_frac is None else hold_frac
    def weights_fn(date, uni, state):
        preds = preds_by_date.get(date, {})
        ranked = sorted((t for t in uni if t in preds), key=lambda t: preds[t], reverse=True)
        if not ranked:
            n = len(uni)
            return {t: 1.0 / n for t in uni} if n else {}
        n_buy = max(1, int(len(ranked) * top_frac))
        n_hold = max(n_buy, int(len(ranked) * hold_frac))
        buy_set = set(ranked[:n_buy])
        hold_set = set(ranked[:n_hold])
        currently_held = set(state.get("prev_weights", {}))
        chosen = buy_set | (currently_held & hold_set)
        w = 1.0 / len(chosen)
        return {t: w for t in chosen}
    return weights_fn


def main(top_n: int = 50, horizon_td: int = 252, top_frac: float = 0.5, hold_frac: float = 0.65):
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
    preds = alpha.walk_forward_predict(df, reb_dates, horizon_td=horizon_td)  # n_estimators: alpha.DEFAULT_N_ESTIMATORS
    print(f"  {preds['date'].nunique()} dates produced a prediction "
          f"(first {len(reb_dates) - preds['date'].nunique()} skipped -- not enough training history yet)")

    ic = alpha.rank_ic(preds, df)
    print("\n=== Out-of-sample rank-IC ===")
    print(f"mean={ic.mean():.4f}  median={ic.median():.4f}  "
          f"fraction of dates with IC>0={float((ic > 0).mean()):.2f}  n_dates={ic.notna().sum()}")

    preds_by_date = {
        d: g.set_index("ticker")["alpha"].to_dict() for d, g in preds.groupby("date")
    }
    alpha_fn = make_alpha_weighted_fn(preds_by_date, top_frac=top_frac, hold_frac=hold_frac)

    cdi = pd.read_parquet(OUTPUT_PATH, columns=["trade_date", "cdi"]).drop_duplicates().sort_values("trade_date")
    prices = df[["ticker", "trade_date", "adj_close"]]

    print("\nRunning equal-weight baseline (same universe/dates/costs)...")
    eq_curve, eq_log = run_backtest(prices, cdi, membership, equal_weight_fn)
    print(f"Running alpha-sort (honest baseline bar: top {top_frac:.0%} buy / "
          f"hold to {hold_frac:.0%}, equal-weight)...")
    alpha_curve, alpha_log = run_backtest(prices, cdi, membership, alpha_fn)

    cdi_series = cdi.set_index("trade_date")["cdi"]
    eq_returns = eq_curve.pct_change().dropna()
    print_report("Equal-weight baseline", full_report(eq_curve, eq_log, cdi_daily=cdi_series))
    print_report(f"Alpha-sort (top {top_frac:.0%} buy / hold {hold_frac:.0%})",
                 full_report(alpha_curve, alpha_log, cdi_daily=cdi_series, benchmark_returns=eq_returns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--horizon-td", type=int, default=252)
    parser.add_argument("--top-frac", type=float, default=0.5, help="buy cutoff, e.g. 0.5 = top half")
    parser.add_argument("--hold-frac", type=float, default=0.65,
                         help="no-trade band: an already-held name stays until it drops below this "
                              "looser cutoff, not just below --top-frac")
    args = parser.parse_args()
    main(top_n=args.top_n, horizon_td=args.horizon_td, top_frac=args.top_frac, hold_frac=args.hold_frac)
