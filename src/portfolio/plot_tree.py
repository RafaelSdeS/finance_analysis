"""
plot_tree.py -- renders one tree from the alpha model (src/portfolio/alpha.py) as
a graphviz image. n_estimators=N means the model is an ensemble of N of these
trees added together; this shows exactly one of them so you can see what a
single split sequence actually looks like (feature, threshold, leaf value).

Run: python -m src.portfolio.plot_tree [--top-n 50] [--horizon-td 21]
                                        [--rebalance-freq M] [--tree-index 0]
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from src.build_dataset.paths import OUTPUT_PATH
from src.build_dataset.terminal_events import load_terminal_events
from src.portfolio import alpha, universe
from src.portfolio.features import feature_columns
from src.portfolio.labels import forward_excess_return

OUT_DIR = "artifacts/models"


def main(top_n: int = 50, horizon_td: int = 21, rebalance_freq: str = "M",
         tree_index: int = 0, n_estimators: int = 50):
    print("Loading dataset...")
    base_cols = ["ticker", "trade_date", "adj_close", "traded_amount"]
    all_cols = sorted(set(base_cols) | set(feature_columns(include_sector=False)))
    df = pd.read_parquet(OUTPUT_PATH, columns=all_cols)

    membership = universe.liquid_universe(df[["ticker", "trade_date", "traded_amount"]],
                                           top_n=top_n, rebalance_freq=rebalance_freq)
    df = universe.restrict_to_universe(df, membership)
    df["label"] = forward_excess_return(df, horizon_td=horizon_td, terminal_events=load_terminal_events())

    reb_dates = universe.rebalance_dates(membership)
    as_of = reb_dates[-1]  # latest rebalance -- the model with the most training history
    print(f"Fitting one model as of {as_of.date()} (n_estimators={n_estimators})...")
    model = alpha.fit(df, as_of, horizon_td=horizon_td, n_estimators=n_estimators)
    if model is None:
        raise SystemExit("Not enough purged training history at this as_of date.")

    graph = lgb.create_tree_digraph(model, tree_index=tree_index)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = f"{OUT_DIR}/alpha_tree_{tree_index}"
    rendered = graph.render(out_path, format="png", cleanup=True)
    print(f"Wrote {rendered} (tree {tree_index} of {model.n_estimators_})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--horizon-td", type=int, default=21)
    parser.add_argument("--rebalance-freq", type=str, default="M")
    parser.add_argument("--tree-index", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=50)
    args = parser.parse_args()
    main(top_n=args.top_n, horizon_td=args.horizon_td, rebalance_freq=args.rebalance_freq,
         tree_index=args.tree_index, n_estimators=args.n_estimators)
