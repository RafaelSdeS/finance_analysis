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


def _compute_forward_returns(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add fwd_ret: horizon-day-ahead log return per (ticker, trade_date) row."""
    price_by_ticker = df.groupby("ticker")["adj_close"].apply(list).to_dict()
    dates_by_ticker = df.groupby("ticker")["trade_date"].apply(list).to_dict()

    fwd_rets = []
    for _, row in df.iterrows():
        ticker = row["ticker"]
        try:
            current_idx = dates_by_ticker[ticker].index(row["trade_date"])
            future_idx = current_idx + horizon
            if future_idx < len(price_by_ticker[ticker]):
                future_price = price_by_ticker[ticker][future_idx]
                current_price = row["adj_close"]
                fwd_rets.append(np.log(future_price / current_price) if current_price > 0 and future_price > 0 else np.nan)
            else:
                fwd_rets.append(np.nan)
        except (ValueError, KeyError):
            fwd_rets.append(np.nan)

    df = df.copy()
    df["fwd_ret"] = fwd_rets
    return df


def _select_features(X: np.ndarray, candidates: list[str], train_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Drop constant/all-NaN columns, using TRAIN rows only (no test-set peeking)."""
    X_train = X[train_mask]
    X_clean_cols, cols_used = [], []
    for i in range(X.shape[1]):
        col_train = X_train[:, i]
        valid = ~np.isnan(col_train)
        if valid.sum() > 0 and len(np.unique(col_train[valid])) >= 2:
            X_clean_cols.append(X[:, i])
            cols_used.append(candidates[i])
    return np.column_stack(X_clean_cols), cols_used


def _score_predictions(dates_test: np.ndarray, pred_test: np.ndarray, y_test: np.ndarray, min_names: int = 10) -> dict | None:
    """Daily cross-sectional rank-IC of predictions vs realized forward returns."""
    df_test = pd.DataFrame({"trade_date": dates_test, "pred": pred_test, "actual": y_test})
    ic_daily = []
    for date in sorted(df_test["trade_date"].unique()):
        day_data = df_test[df_test["trade_date"] == date]
        if len(day_data) >= min_names:
            pred_rank = day_data["pred"].rank(method="average").values
            actual_rank = day_data["actual"].rank(method="average").values
            ic = np.corrcoef(pred_rank, actual_rank)[0, 1]
            if not np.isnan(ic):
                ic_daily.append(ic)

    if not ic_daily:
        return None

    ic_series = pd.Series(ic_daily)
    mean_ic, std_ic, n_days = float(ic_series.mean()), float(ic_series.std()), len(ic_series)
    t_stat = mean_ic / (std_ic / np.sqrt(n_days)) if std_ic > 0 else 0.0
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "n_days": n_days,
        "t_stat": t_stat,
        "max_ic": float(ic_series.max()),
        "min_ic": float(ic_series.min()),
    }


def compute_ranker_ic(df: pd.DataFrame, horizon: int = 21, split: str = "time", seed: int = 42) -> dict | None:
    """Train a ranker on one half of the data, test on the other, compute daily rank-IC.

    Args:
        split: "time" (default, honest) — train on the calendar-earlier half of dates,
            test on the strictly later half (train_end < test_start, zero date overlap).
            "row" (legacy — kept ONLY to demonstrate the leak it has; do not use for a
            real signal claim, see DIAGNOSIS_PLAN.md F2) — splits by row POSITION after
            the parquet's natural (ticker, trade_date) sort order, which is effectively a
            TICKER-holdout with FULL date overlap: every one of the dataset's ~6,568
            trading dates appears on both sides (measured). A model can pick up a
            contemporaneous/common-mode shock ("the whole market moved on this date")
            from train tickers and trivially "predict" the same date's outcome for
            held-out tickers — that is not forward-looking skill, and is the leak that
            produced this script's original t=24 headline number.
    """
    candidates = [c for c in df.columns if df[c].dtype.kind == "f" and c not in SKIP_COLS]
    logger.info(f"Candidate features: {len(candidates)}")

    df = _compute_forward_returns(df, horizon)
    df_valid = df.loc[df["fwd_ret"].notna()].copy()
    logger.info(f"Valid rows with targets: {len(df_valid)}")
    if len(df_valid) == 0:
        logger.warning("No rows with a valid %d-day forward return (need > %d dates per ticker)", horizon, horizon)
        return None

    if split == "time":
        dates_sorted = np.sort(df_valid["trade_date"].unique())
        cutoff = dates_sorted[len(dates_sorted) // 2]
        train_mask = (df_valid["trade_date"] <= cutoff).to_numpy()
    elif split == "row":
        train_mask = np.arange(len(df_valid)) < len(df_valid) // 2
    else:
        raise ValueError(f"split must be 'time' or 'row', got {split!r}")
    test_mask = ~train_mask

    X = df_valid[candidates].values
    y = df_valid["fwd_ret"].values
    X_clean, cols_used = _select_features(X, candidates, train_mask)
    logger.info(f"After filtering: {X_clean.shape[1]} features")

    X_train, X_test = X_clean[train_mask], X_clean[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    dates_test = df_valid["trade_date"].to_numpy()[test_mask]
    logger.info(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows (split={split})")

    logger.info("Training HistGradientBoosting...")
    model = HistGradientBoostingRegressor(max_iter=50, early_stopping=False, random_state=seed)
    model.fit(X_train, y_train)

    logger.info("Predicting on test set...")
    pred_test = model.predict(X_test)

    result = _score_predictions(dates_test, pred_test, y_test)
    if result is None:
        logger.error("No daily IC computed")
        return None

    logger.info(f"Ranker IC: mean={result['mean_ic']:.4f}, std={result['std_ic']:.4f}, t-stat={result['t_stat']:.2f}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Supervised ranker baseline")
    parser.add_argument("--dataset", type=Path, default=_PROJECT_ROOT / "data/processed/ml_dataset.parquet")
    parser.add_argument("--split", choices=["time", "row"], default="time",
                         help="'time' (default, honest): train/test split by calendar date, no overlap. "
                              "'row' (legacy, DEPRECATED): ticker-holdout with full date overlap — "
                              "see DIAGNOSIS_PLAN.md F2, kept only for comparison")
    parser.add_argument("--out", type=Path, default=_PROJECT_ROOT / "artifacts/backtest/ranker_metrics.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    df = load_dataset(args.dataset)
    result = compute_ranker_ic(df, horizon=21, split=args.split)

    if result is None:
        return

    print("\n" + "=" * 70)
    print(f"RANKER BASELINE IC (50/50 train/test split, split={args.split})")
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
        json.dump({"ranker": {**result, "split": args.split}}, f, indent=2)
    logger.info(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
