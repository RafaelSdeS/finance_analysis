"""
Supervised ranker baseline: can HistGradientBoosting extract signal?

Simple approach: train on first half, test on second half.
Compute daily IC (correlation between predictions and forward returns).
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.agent.ic_analysis import SKIP_COLS

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load ml_dataset.parquet."""
    df = pd.read_parquet(dataset_path)
    if "ticker" in df.columns:
        df = df[df["ticker"] != "CASH"].reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} cols")
    return df


def compute_ranker_ic(df: pd.DataFrame, horizon: int = 21):
    """Train on first 50% of data, test on last 50%. Compute IC."""
    # Build features and targets
    candidates = [c for c in df.columns if df[c].dtype.kind == "f" and c not in SKIP_COLS]
    logger.info(f"Candidate features: {len(candidates)}")

    # Compute forward returns
    price_by_ticker = df.groupby("ticker")["adj_close"].apply(list).to_dict()
    dates_by_ticker = df.groupby("ticker")["trade_date"].apply(list).to_dict()

    fwd_rets = []
    for idx, row in df.iterrows():
        ticker = row["ticker"]
        try:
            current_idx = dates_by_ticker[ticker].index(row["trade_date"])
            future_idx = current_idx + horizon
            if future_idx < len(price_by_ticker[ticker]):
                future_price = price_by_ticker[ticker][future_idx]
                current_price = row["adj_close"]
                if current_price > 0 and future_price > 0:
                    fwd_rets.append(np.log(future_price / current_price))
                else:
                    fwd_rets.append(np.nan)
            else:
                fwd_rets.append(np.nan)
        except (ValueError, KeyError):
            fwd_rets.append(np.nan)

    df = df.copy()
    df["fwd_ret"] = fwd_rets
    df_valid = df.loc[df["fwd_ret"].notna()].copy()

    logger.info(f"Valid rows with targets: {len(df_valid)}")

    X = df_valid[candidates].values
    y = df_valid["fwd_ret"].values
    dates = df_valid["trade_date"].values

    # Filter to valid features
    X_clean = []
    cols_used = []
    for i in range(X.shape[1]):
        col_data = X[:, i]
        valid = ~np.isnan(col_data)
        if valid.sum() > 0:
            distinct = len(np.unique(col_data[valid]))
            if distinct >= 2:
                X_clean.append(col_data)
                cols_used.append(candidates[i])

    X_clean = np.column_stack(X_clean)
    logger.info(f"After filtering: {X_clean.shape[1]} features")

    # Split: first 50% train, last 50% test
    split_idx = len(X_clean) // 2
    X_train, X_test = X_clean[:split_idx], X_clean[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]

    logger.info(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    # Train ranker
    logger.info("Training HistGradientBoosting...")
    model = HistGradientBoostingRegressor(max_iter=50, early_stopping=False, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test
    logger.info("Predicting on test set...")
    pred_test = model.predict(X_test)

    # Compute daily IC
    df_test = df_valid.iloc[split_idx:].copy()
    df_test["pred"] = pred_test
    df_test["actual"] = y_test

    ic_daily = []
    for date in sorted(df_test["trade_date"].unique()):
        day_data = df_test[df_test["trade_date"] == date]
        if len(day_data) >= 10:  # Min tickers
            pred_rank = day_data["pred"].rank(method="average").values
            actual_rank = day_data["actual"].rank(method="average").values
            ic = np.corrcoef(pred_rank, actual_rank)[0, 1]
            if not np.isnan(ic):
                ic_daily.append(ic)

    if not ic_daily:
        logger.error("No daily IC computed")
        return None

    ic_series = pd.Series(ic_daily)
    mean_ic = float(ic_series.mean())
    std_ic = float(ic_series.std())
    n_days = len(ic_series)
    t_stat = mean_ic / (std_ic / np.sqrt(n_days)) if std_ic > 0 else 0.0

    logger.info(f"Ranker IC: mean={mean_ic:.4f}, std={std_ic:.4f}, t-stat={t_stat:.2f}")

    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "n_days": n_days,
        "t_stat": t_stat,
        "max_ic": float(ic_series.max()),
        "min_ic": float(ic_series.min()),
    }


def main():
    parser = argparse.ArgumentParser(description="Supervised ranker baseline")
    parser.add_argument("--dataset", type=Path, default=_PROJECT_ROOT / "data/processed/ml_dataset.parquet")
    parser.add_argument("--out", type=Path, default=_PROJECT_ROOT / "data/backtest/ranker_metrics.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    df = load_dataset(args.dataset)
    result = compute_ranker_ic(df, horizon=21)

    if result is None:
        return

    print("\n" + "=" * 70)
    print("RANKER BASELINE IC (50/50 train/test split)")
    print("=" * 70)
    print(f"{'Mean IC':<30} {result['mean_ic']:>14.4f}")
    print(f"{'Std IC':<30} {result['std_ic']:>14.4f}")
    print(f"{'t-stat':<30} {result['t_stat']:>14.2f}")
    print(f"{'N days':<30} {result['n_days']:>14.0f}")
    print()

    if result['mean_ic'] > 0.05:
        print("✓ STRONG SIGNAL (IC > 0.05): Model extracted significant alpha")
    elif result['mean_ic'] > 0.02:
        print("✓ MODERATE SIGNAL (IC > 0.02): Model extracted exploitable alpha")
    elif result['mean_ic'] > 0:
        print("~ WEAK SIGNAL (IC ≈ 0.01): Marginal after costs")
    else:
        print("✗ NO SIGNAL (IC ≤ 0): No extractable alpha")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"ranker": result}, f, indent=2)
    logger.info(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
