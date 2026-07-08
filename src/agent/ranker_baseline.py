"""
Supervised ranker baseline: gradient boosting on cross-sectional returns.

Upper-bound estimate of what exploitable signal exists in the feature set.
Trains HistGradientBoostingRegressor per anchored window on 21d forward returns.
Walk-forward backtests with 21-day rebalance cadence, same costing as the agent.
Compares vs all-active equal-weight baseline (same cadence, same costs).

Output: data/backtest/ranker_{metrics.json,results.parquet} + t-test on daily excess returns.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

from src.agent.config import DEFAULT_CONFIG, generate_windows, window_to_config, AgentConfig
from src.agent.ic_analysis import SKIP_COLS
from src.agent.metrics import compute_all
import dataclasses

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load ml_dataset.parquet (not training version with synthetic CASH)."""
    df = pd.read_parquet(dataset_path)
    if "ticker" in df.columns:
        df = df[df["ticker"] != "CASH"].reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} cols from {dataset_path.name}")
    return df


def build_features_and_targets(df: pd.DataFrame, horizon: int = 21):
    """Build feature matrix (X) and forward-return targets (y).

    Returns:
        (X, y, dates, tickers): feature matrix, targets, date index, ticker index
    """
    candidates = [c for c in df.columns if df[c].dtype.kind == "f" and c not in SKIP_COLS]
    logger.info(f"Using {len(candidates)} features for ranker")

    # Forward return target per ticker
    price_by_ticker = df.groupby("ticker")["adj_close"].apply(list).to_dict()
    dates_by_ticker = df.groupby("ticker")["trade_date"].apply(list).to_dict()

    fwd_rets = []
    valid_rows = []
    for idx, row in df.iterrows():
        ticker = row["ticker"]
        current_idx = dates_by_ticker[ticker].index(row["trade_date"])
        future_idx = current_idx + horizon

        if future_idx < len(price_by_ticker[ticker]):
            future_price = price_by_ticker[ticker][future_idx]
            current_price = row["adj_close"]
            fwd_ret = np.log(future_price / current_price)
            fwd_rets.append(fwd_ret)
            valid_rows.append(idx)
        else:
            fwd_rets.append(np.nan)

    df["fwd_ret"] = fwd_rets
    df_valid = df.loc[df["fwd_ret"].notna()].copy()

    X = df_valid[candidates].values
    y = df_valid["fwd_ret"].values
    dates = df_valid["trade_date"].values
    tickers = df_valid["ticker"].values

    logger.info(f"Training set: {len(X)} rows, {X.shape[1]} features, {len(set(tickers))} tickers")
    return X, y, dates, tickers


def portfolio_simulator(predictions: np.ndarray, dates: np.ndarray, tickers: np.ndarray,
                       all_returns: dict, initial_capital: float = 100_000,
                       rebalance_days: int = 21, cost_bps: float = 10.0):
    """Simulate portfolio: predict forward returns, top quintile equal-weight, drift between rebalances."""
    unique_dates = sorted(set(dates))
    daily_log_rets = []
    daily_portfolio_values = [initial_capital]

    portfolio_value = initial_capital
    target_weights = {}
    prev_weights = {}

    for date_idx, date in enumerate(unique_dates):
        # Get current predictions for this date
        mask = dates == date
        pred_for_date = predictions[mask]
        tickers_for_date = tickers[mask]

        # Top quintile by prediction
        if len(pred_for_date) > 0:
            quintile_threshold = np.percentile(pred_for_date, 80)
            top_mask = pred_for_date >= quintile_threshold
            top_names = tickers_for_date[top_mask]

            if len(top_names) >= 5:  # Min 5 names per quintile
                target_weights = {name: 1.0 / len(top_names) for name in top_names}
            else:
                target_weights = {}
        else:
            target_weights = {}

        # Initialize prev weights on first date
        if date_idx == 0:
            prev_weights = target_weights.copy()

        # Daily return: compute for active names only
        active_names = set(prev_weights.keys())
        daily_return = 0.0
        cost = 0.0

        if active_names:
            for name in active_names:
                weight = prev_weights.get(name, 0.0)
                ret = all_returns.get((date, name), 0.0)
                daily_return += weight * ret

            # Cost: one-way turnover vs previous weights
            if date_idx % rebalance_days == 0:
                traded = sum(abs(target_weights.get(name, 0.0) - prev_weights.get(name, 0.0))
                           for name in set(list(prev_weights.keys()) + list(target_weights.keys())))
                cost = (traded / 2) * (cost_bps / 1e4)
                daily_return -= cost

        daily_return = max(daily_return, -0.9999)  # Clip catastrophic days
        log_return = np.log1p(daily_return)
        daily_log_rets.append(log_return)

        portfolio_value *= 1.0 + daily_return
        daily_portfolio_values.append(portfolio_value)

        # Drift weights if holding, rebalance on interval
        if date_idx % rebalance_days == 0 or not active_names:
            prev_weights = target_weights.copy()
        else:
            # Drift: w_i ← w_i * (1 + r_i) / (1 + r_p)
            r_p = daily_return + cost  # Pre-cost return for drift
            if r_p > -0.9999:
                for name in prev_weights:
                    r_i = all_returns.get((date, name), 0.0)
                    prev_weights[name] *= (1.0 + r_i) / (1.0 + r_p)

    return np.array(daily_log_rets), daily_portfolio_values[1:]


def main():
    parser = argparse.ArgumentParser(description="Supervised ranker baseline")
    parser.add_argument("--dataset", type=Path, default=_PROJECT_ROOT / "data/processed/ml_dataset.parquet")
    parser.add_argument("--top-frac", type=float, default=0.2, help="Top fraction for quintile")
    parser.add_argument("--rebalance-days", type=int, default=21)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--train-years", type=int, default=10)
    parser.add_argument("--test-years", type=int, default=2)
    parser.add_argument("--out", type=Path, default=_PROJECT_ROOT / "data/backtest/ranker_metrics.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    df = load_dataset(args.dataset)

    # Build all-dates price dict for returns
    price_dict = df.set_index(["trade_date", "ticker"])["adj_close"].to_dict()

    # Generate windows
    base_config = DEFAULT_CONFIG
    windows = generate_windows(base_config.dataset_start, base_config.dataset_end,
                              args.train_years, args.test_years)
    logger.info(f"Generated {len(windows)} windows")

    all_rets_ranker = []
    all_rets_ew = []
    all_dates = []

    for w_idx, window in enumerate(windows):
        logger.info(f"\n--- Window {w_idx} ---")

        # Train on full train span
        train_df = df[(df["trade_date"] >= window.train_start) & (df["trade_date"] <= window.train_end)]
        X_train, y_train, _, _ = build_features_and_targets(train_df, horizon=21)

        if len(X_train) == 0:
            logger.warning(f"Window {w_idx}: no train data")
            continue

        # Train ranker
        model = HistGradientBoostingRegressor(max_iter=300, early_stopping=False, random_state=42)
        model.fit(X_train, y_train)
        logger.info(f"Ranker trained: {len(X_train)} samples")

        # Test on OOS span
        test_df = df[(df["trade_date"] >= window.test_start) & (df["trade_date"] <= window.test_end)]
        if len(test_df) == 0:
            logger.warning(f"Window {w_idx}: no test data")
            continue

        candidates = [c for c in test_df.columns if test_df[c].dtype.kind == "f" and c not in SKIP_COLS]
        X_test = test_df[candidates].values
        pred_test = model.predict(X_test)

        # Build returns dict for simulator
        returns_dict = {}
        for _, row in test_df.iterrows():
            date = row["trade_date"]
            ticker = row["ticker"]
            next_date_idx = test_df[test_df["ticker"] == ticker]["trade_date"].tolist().index(date) + 1
            all_dates_ticker = test_df[test_df["ticker"] == ticker]["trade_date"].tolist()

            if next_date_idx < len(all_dates_ticker):
                next_date = all_dates_ticker[next_date_idx]
                try:
                    next_price = price_dict[(next_date, ticker)]
                    returns_dict[(date, ticker)] = np.log(next_price / row["adj_close"])
                except KeyError:
                    pass

        # Simulate ranker
        ranker_rets, ranker_vals = portfolio_simulator(
            pred_test, test_df["trade_date"].values, test_df["ticker"].values,
            returns_dict, rebalance_days=args.rebalance_days, cost_bps=args.cost_bps
        )
        all_rets_ranker.extend(ranker_rets)

        # Simulate all-active EW (same cadence, costs)
        ew_pred = np.ones_like(pred_test)  # Uniform predictions
        ew_rets, ew_vals = portfolio_simulator(
            ew_pred, test_df["trade_date"].values, test_df["ticker"].values,
            returns_dict, rebalance_days=args.rebalance_days, cost_bps=args.cost_bps
        )
        all_rets_ew.extend(ew_rets)

        unique_dates_test = sorted(set(test_df["trade_date"]))
        all_dates.extend(unique_dates_test[:len(ranker_rets)])

        logger.info(f"Window {w_idx} test: {len(ranker_rets)} days")

    # Metrics
    ranker_rets = np.array(all_rets_ranker)
    ew_rets = np.array(all_rets_ew)
    excess_rets = ranker_rets - ew_rets

    if len(ranker_rets) > 0:
        ranker_metrics = compute_all(ranker_rets, np.exp(np.cumsum(ranker_rets)))
        ew_metrics = compute_all(ew_rets, np.exp(np.cumsum(ew_rets)))

        print("\n" + "=" * 60)
        print("RANKER VS EQUAL-WEIGHT")
        print("=" * 60)
        print(f"{'Metric':<25} {'Ranker':>12} {'EW':>12}")
        print("-" * 50)
        for key in ["annualized_return", "sharpe", "max_drawdown", "win_rate"]:
            print(f"{key:<25} {ranker_metrics.get(key, 0):>12.4f} {ew_metrics.get(key, 0):>12.4f}")

        # T-test
        t_stat, p_val = stats.ttest_1samp(excess_rets, popmean=0.0)
        print(f"\nDaily excess (ranker - EW):")
        print(f"  Mean: {excess_rets.mean() * 1e4:.2f} bps/day")
        print(f"  t-stat: {t_stat:.3f}, p-value: {p_val:.4f}")
        print(f"  Interpretation: {'SIGNAL EXISTS' if abs(excess_rets.mean()) > 0.0001 else 'NO SIGNAL'}")

        # Save
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"ranker": ranker_metrics, "equal_weight": ew_metrics,
                      "t_test": {"t_stat": float(t_stat), "p_value": float(p_val),
                                "mean_excess_bps": float(excess_rets.mean() * 1e4)}}, f, indent=2)
        logger.info(f"Metrics saved → {args.out}")


if __name__ == "__main__":
    main()
