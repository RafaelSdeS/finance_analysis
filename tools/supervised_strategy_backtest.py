"""
DIAGNOSIS_PLAN.md Phase 1.1: walk-forward top-k equal-weight strategy on the point-in-time
universe -- the ceiling any RL work (Phase 2) has to beat.

Per rolling window: trains a HistGradientBoostingRegressor teacher on TRAIN-span features
only (features -> forward `horizon`-day log return, PIT-active non-cash names -- tree
models are scale-invariant per feature, so no scaler is needed, matching the ic_analysis.py/
ranker_baseline.py convention already used in Phase 0), then backtests a top-k equal-weight
strategy (supervised_strategy.backtest_topk_strategy) on the TEST span, for k in {10, 20}.
Pools all 8 windows and scores with Phase 0's primary metrics: tilt-IC (HAC-tested) plus
excess-of-EW daily-return HAC/bootstrap and deflated Sharpe across the k-sweep.

No production artifacts touched: PIT tensors are built in-memory only (build_tensors).

Usage:
    python tools/supervised_strategy_backtest.py [--universe-size 50] [--horizon 21]
"""

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root, for `src.*` imports

from src.agent.config import DEFAULT_CONFIG, generate_windows, window_to_config
from src.agent.data_pipeline import build_tensors
from src.agent.metrics import (
    TRADING_DAYS,
    block_bootstrap_mean_ci,
    cumulative_return,
    deflated_sharpe_ratio,
    hac_mean_test,
    sharpe_ratio,
    tilt_ic_significance,
)
from src.agent.ranker_baseline import _select_features
from src.agent.supervised_strategy import backtest_topk_strategy

logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 100_000.0


def _simple_returns_from_tensors(tensors: dict) -> np.ndarray:
    returns, mask = tensors["returns"], tensors["mask"]
    return np.where(mask, np.expm1(np.nan_to_num(returns, nan=0.0)), 0.0)


def _train_teacher(features: np.ndarray, active: np.ndarray, simple_rets: np.ndarray,
                    train_idx: np.ndarray, horizon: int, seed: int):
    """Fit on (day, ticker) rows with day in train_idx, day+horizon within range, and the
    ticker PIT-active + non-cash that day. Mirrors bc_pretrain.train_teacher's target
    construction (forward cumulative log return), applied here to raw (unscaled) features.

    Drops constant/all-NaN feature columns using TRAIN rows only (reuses
    ranker_baseline._select_features, same fix as Phase 0.4) -- early PIT windows can have
    thin enough history that a feature is genuinely constant across the whole train span
    (e.g. a macro series with too few distinct values yet), which otherwise crashes
    HistGradientBoostingRegressor's binning step.
    """
    X_rows, y_rows = [], []
    for t in train_idx:
        if t + horizon >= simple_rets.shape[0]:
            continue
        active_t = active[t]
        if not active_t.any():
            continue
        fwd_log = np.log1p(simple_rets[t + 1: t + horizon + 1]).sum(axis=0)  # [N]
        X_rows.append(features[t][active_t])
        y_rows.append(fwd_log[active_t])

    X_train_raw = np.concatenate(X_rows, axis=0)
    y_train = np.concatenate(y_rows, axis=0)

    n_features = X_train_raw.shape[1]
    X_train, feature_idx = _select_features(X_train_raw, list(range(n_features)), np.ones(len(X_train_raw), dtype=bool))

    model = HistGradientBoostingRegressor(max_iter=100, random_state=seed)
    model.fit(X_train, y_train)
    return model, np.array(feature_idx, dtype=np.int64), len(X_train)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward top-k EW strategy on the PIT universe (Phase 1.1)")
    parser.add_argument("--universe-size", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--rebalance-days", type=int, default=21)
    parser.add_argument("--out", type=Path, default=Path("artifacts/backtest/supervised_strategy.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    base = DEFAULT_CONFIG
    pit_config = dataclasses.replace(base, universe_size=args.universe_size, pit_universe=True)
    logger.info("Building PIT tensors (top-%d per window)...", args.universe_size)
    tensors = build_tensors(pit_config)

    dates = pd.to_datetime(tensors["dates"])
    tickers = tensors["tickers"]
    is_cash = tickers == "CASH"
    features = tensors["features"]  # raw, unnormalized -- fine for a tree model
    mask = tensors["mask"]
    simple_rets = _simple_returns_from_tensors(tensors)
    pit_cutoffs = list(tensors["pit_cutoffs"])
    pit_membership = tensors["pit_membership_mask"]

    windows = generate_windows(base.dataset_start, base.dataset_end, base.window_train_years, base.window_test_years)

    # Pooled (concatenated across all windows) series per k, for the final significance tests
    pooled = {k: {"daily_rets": [], "rebal_w": [], "rebal_ew": [], "rebal_fwd": []} for k in args.ks}
    per_window_rows = []

    for w in windows:
        wc = window_to_config(w, base)
        cutoff = wc.train_end
        if cutoff not in pit_cutoffs:
            logger.warning("window %d: no PIT membership row for cutoff=%s, skipping", w.window_id, cutoff)
            continue
        pit_row = pit_membership[pit_cutoffs.index(cutoff)]
        active_full = mask & pit_row[np.newaxis, :] & ~is_cash[np.newaxis, :]

        train_idx = np.where(dates <= pd.Timestamp(wc.train_end))[0]
        test_idx = np.where((dates >= pd.Timestamp(wc.test_start)) & (dates <= pd.Timestamp(wc.test_end)))[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            logger.warning("window %d: empty train or test slice, skipping", w.window_id)
            continue

        model, feature_idx, n_train_rows = _train_teacher(features, active_full, simple_rets, train_idx, args.horizon, base.seed)
        logger.info("window %d: teacher trained on %d (day,ticker) rows, %d/%d features kept",
                    w.window_id, n_train_rows, len(feature_idx), features.shape[2])

        active_test = active_full[test_idx]
        simple_test = simple_rets[test_idx]
        scores_test = np.zeros((len(test_idx), len(tickers)))
        for i, t in enumerate(test_idx):
            active_t = active_full[t]
            if active_t.any():
                scores_test[i, active_t] = model.predict(features[t][active_t][:, feature_idx])

        row = {"window": w.window_id, "test_start": wc.test_start, "test_end": wc.test_end, "n_train_rows": n_train_rows}
        for k in args.ks:
            res = backtest_topk_strategy(
                scores_test, active_test, simple_test, k=k, cap=base.max_position_weight,
                cost_bps=args.cost_bps, rebalance_interval_days=args.rebalance_days, turnover_band=True,
            )
            values = INITIAL_CAPITAL * np.exp(np.concatenate([[0.0], np.cumsum(res["daily_log_returns"])]))
            row[f"k{k}_cum_return"] = cumulative_return(values)
            row[f"k{k}_sharpe"] = sharpe_ratio(res["daily_log_returns"])
            row[f"k{k}_n_rebalances"] = len(res["rebal_weights"])

            pooled[k]["daily_rets"].append(res["daily_log_returns"])
            pooled[k]["rebal_w"].append(res["rebal_weights"])
            pooled[k]["rebal_fwd"].append(res["rebal_fwd_rets"])
            ew_at_decision = active_test[res["rebal_t"]].astype(np.float64)
            ew_at_decision /= np.maximum(ew_at_decision.sum(axis=1, keepdims=True), 1.0)
            pooled[k]["rebal_ew"].append(ew_at_decision)
        per_window_rows.append(row)
        logger.info("  window %d done", w.window_id)

    # --- pooled significance tests per k ---
    summary = {}
    for k in args.ks:
        daily_rets = np.concatenate(pooled[k]["daily_rets"])
        rebal_w = np.concatenate(pooled[k]["rebal_w"], axis=0)
        rebal_ew = np.concatenate(pooled[k]["rebal_ew"], axis=0)
        rebal_fwd = np.concatenate(pooled[k]["rebal_fwd"], axis=0)

        hac = hac_mean_test(daily_rets, lag=args.rebalance_days)
        boot = block_bootstrap_mean_ci(daily_rets, block_size=args.rebalance_days, n_resamples=2000, seed=base.seed)
        tilt = tilt_ic_significance(rebal_w, rebal_ew, rebal_fwd, active=(rebal_ew > 0), min_names=10, lag=1)

        values = INITIAL_CAPITAL * np.exp(np.concatenate([[0.0], np.cumsum(daily_rets)]))
        summary[f"k{k}"] = {
            "cumulative_return": cumulative_return(values),
            "annualized_sharpe": sharpe_ratio(daily_rets) if len(daily_rets) > 1 else 0.0,
            "hac_mean_daily_return": hac["mean"], "hac_p_value": hac["p_value"],
            "bootstrap_ci_low": boot["ci_low"], "bootstrap_ci_high": boot["ci_high"],
            "tilt_ic_mean": tilt["mean"], "tilt_ic_p_value": tilt["p_value"], "tilt_ic_n_days": tilt["n_valid_days"],
            "n_daily_obs": len(daily_rets),
        }

    trial_sharpes = np.array([summary[f"k{k}"]["annualized_sharpe"] for k in args.ks]) / np.sqrt(TRADING_DAYS)
    for k in args.ks:
        if len(trial_sharpes) >= 2:
            daily_rets = np.concatenate(pooled[k]["daily_rets"])
            summary[f"k{k}"]["deflated_sharpe_ratio"] = deflated_sharpe_ratio(daily_rets, trial_sharpes)

    table = pd.DataFrame(per_window_rows)
    pd.set_option("display.width", 140)
    print("\n" + "=" * 110)
    print("WALK-FORWARD TOP-K SUPERVISED STRATEGY — per window")
    print("=" * 110)
    print(table.round(4).to_string(index=False))

    print("\n" + "=" * 110)
    print("POOLED SIGNIFICANCE (all windows concatenated)")
    print("=" * 110)
    print(pd.DataFrame(summary).T.round(4).to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"per_window": per_window_rows, "pooled": summary}, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
