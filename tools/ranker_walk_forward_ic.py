"""
DIAGNOSIS_PLAN.md Phase 0.4: walk-forward ranker IC per rolling window.

Reuses ranker_baseline's honest (time-based) train/test machinery, but applied per the
RL pipeline's own anchored rolling windows (config.generate_windows) instead of one global
50/50 split — train <= that window's train_end, test = that window's [test_start, test_end],
matching exactly what the agent itself trains/evaluates on.

Usage:
    python tools/ranker_walk_forward_ic.py [--horizon 21]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root, for `src.*` imports

from src.agent.config import DEFAULT_CONFIG, generate_windows, window_to_config
from src.agent.ic_analysis import SKIP_COLS
from src.agent.ranker_baseline import _compute_forward_returns, _score_predictions, _select_features, load_dataset

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward ranker IC per rolling window (DIAGNOSIS_PLAN Phase 0.4)")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/ml_dataset.parquet"))
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--min-names", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("artifacts/backtest/ranker_walk_forward_ic.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    df = load_dataset(args.dataset)
    candidates = [c for c in df.columns if df[c].dtype.kind == "f" and c not in SKIP_COLS]
    df = _compute_forward_returns(df, args.horizon)
    df_valid = df.loc[df["fwd_ret"].notna()].copy()
    logger.info("Valid rows with %dd forward targets: %d", args.horizon, len(df_valid))

    base = DEFAULT_CONFIG
    windows = generate_windows(base.dataset_start, base.dataset_end, base.window_train_years, base.window_test_years)

    X_all = df_valid[candidates].values
    y_all = df_valid["fwd_ret"].values
    dates_all = df_valid["trade_date"].to_numpy()

    rows = []
    for w in windows:
        wc = window_to_config(w, base)
        train_mask = dates_all <= pd.Timestamp(wc.train_end).to_datetime64()
        test_mask = (dates_all >= pd.Timestamp(wc.test_start).to_datetime64()) & (dates_all <= pd.Timestamp(wc.test_end).to_datetime64())

        row = {
            "window": w.window_id, "test_start": wc.test_start, "test_end": wc.test_end,
            "n_train_rows": int(train_mask.sum()), "n_test_rows": int(test_mask.sum()),
        }
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            logger.warning("window %d: empty train or test slice, skipping", w.window_id)
            rows.append(row)
            continue

        X_clean, cols_used = _select_features(X_all, candidates, train_mask)
        model = HistGradientBoostingRegressor(max_iter=50, early_stopping=False, random_state=base.seed)
        model.fit(X_clean[train_mask], y_all[train_mask])
        pred_test = model.predict(X_clean[test_mask])

        result = _score_predictions(dates_all[test_mask], pred_test, y_all[test_mask], min_names=args.min_names)
        if result is not None:
            row.update(result)
        rows.append(row)
        logger.info("  window %d done (%d train rows, %d test rows, %d features)",
                    w.window_id, train_mask.sum(), test_mask.sum(), X_clean.shape[1])

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print("\n" + "=" * 100)
    print(f"WALK-FORWARD RANKER IC — anchored rolling windows (train <= train_end, "
          f"test = [test_start, test_end]), horizon={args.horizon}d")
    print("=" * 100)
    print(table.round(4).to_string(index=False))
    if "mean_ic" in table.columns:
        print(f"\nMean |IC| across windows: {table['mean_ic'].abs().mean():.4f}")
        print(f"Mean t-stat across windows: {table['t_stat'].mean():.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(table.to_dict(orient="records"), f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
