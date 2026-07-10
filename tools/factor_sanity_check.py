"""
DIAGNOSIS_PLAN.md Phase 1.3: is the alpha waterfall's long-short signal (1.2) a genuine,
independent edge, or just a size/momentum style tilt in disguise?

For each rolling window, reuses 1.1/1.2's exact per-window teacher to get the long-short
strategy's period-return series, then regresses it (pooled across windows) on two
independent factor-portfolio return series built from the SAME PIT-active universe:
  size:     smallest-market-cap decile return - largest-market-cap decile return (SMB-style)
  momentum: highest vs lowest momentum_vs_market_3m decile return (already a state feature)

Usage:
    python tools/factor_sanity_check.py [--universe-size 50] [--horizon 21] [--k 5]
"""

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root, for `src.*` imports

from src.agent.config import DEFAULT_CONFIG, generate_windows, window_to_config
from src.agent.data_pipeline import build_tensors
from src.agent.supervised_strategy import compute_alpha_waterfall, regress_on_factors
from tools.supervised_strategy_backtest import _simple_returns_from_tensors, _train_teacher

logger = logging.getLogger(__name__)


def _decile_spread_factor(factor_values: np.ndarray, active_idx: np.ndarray, fwd_cum: np.ndarray) -> float:
    """Independent factor-portfolio period return: EW(top decile by factor_values) -
    EW(bottom decile by factor_values), among active_idx names, using their OWN period
    return fwd_cum -- NOT the strategy's picks, so this is a genuinely separate portfolio."""
    vals = factor_values[active_idx]
    valid = ~np.isnan(vals)
    idx = active_idx[valid]
    if len(idx) < 10:
        return np.nan
    order = idx[np.argsort(vals[valid])]
    decile = max(len(order) // 10, 1)
    bottom, top = order[:decile], order[-decile:]
    return float(fwd_cum[top].mean() - fwd_cum[bottom].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Factor sanity check on the waterfall's long-short signal (Phase 1.3)")
    parser.add_argument("--universe-size", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--rebalance-days", type=int, default=21)
    parser.add_argument("--out", type=Path, default=Path("artifacts/backtest/factor_sanity_check.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    base = DEFAULT_CONFIG
    pit_config = dataclasses.replace(base, universe_size=args.universe_size, pit_universe=True)
    logger.info("Building PIT tensors (top-%d per window)...", args.universe_size)
    tensors = build_tensors(pit_config)

    dates = pd.to_datetime(tensors["dates"])
    tickers = tensors["tickers"]
    is_cash = tickers == "CASH"
    features = tensors["features"]
    mask = tensors["mask"]
    simple_rets = _simple_returns_from_tensors(tensors)
    pit_cutoffs = list(tensors["pit_cutoffs"])
    pit_membership = tensors["pit_membership_mask"]

    mom_idx = base.state_features.index("momentum_vs_market_3m")
    caps = pd.read_parquet(base.dataset_path, columns=["ticker", "trade_date", "market_cap"])
    cap_pivot = caps.pivot(index="trade_date", columns="ticker", values="market_cap").reindex(
        index=dates, columns=tickers
    ).ffill().to_numpy()

    windows = generate_windows(base.dataset_start, base.dataset_end, base.window_train_years, base.window_test_years)

    strategy_rets, size_factor, mom_factor = [], [], []
    for w in windows:
        wc = window_to_config(w, base)
        cutoff = wc.train_end
        if cutoff not in pit_cutoffs:
            continue
        pit_row = pit_membership[pit_cutoffs.index(cutoff)]
        active_full = mask & pit_row[np.newaxis, :] & ~is_cash[np.newaxis, :]

        train_idx = np.where(dates <= pd.Timestamp(wc.train_end))[0]
        test_idx = np.where((dates >= pd.Timestamp(wc.test_start)) & (dates <= pd.Timestamp(wc.test_end)))[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        model, feature_idx, _ = _train_teacher(features, active_full, simple_rets, train_idx, args.horizon, base.seed)

        active_test = active_full[test_idx]
        simple_test = simple_rets[test_idx]
        scores_test = np.zeros((len(test_idx), len(tickers)))
        for i, t in enumerate(test_idx):
            active_t = active_full[t]
            if active_t.any():
                scores_test[i, active_t] = model.predict(features[t][active_t][:, feature_idx])

        wf = compute_alpha_waterfall(scores_test, active_test, simple_test, k=args.k,
                                      cap=base.max_position_weight, cost_bps=10.0,
                                      rebalance_interval_days=args.rebalance_days)
        n_steps = len(test_idx) - 1
        for local_t in wf["long_short_t"]:
            global_t = test_idx[local_t]
            n_days = min(args.rebalance_days, n_steps - local_t)
            fwd_cum = np.prod(1.0 + simple_test[local_t + 1: local_t + n_days + 1], axis=0) - 1.0
            active_idx = np.where(active_test[local_t])[0]

            size_factor.append(_decile_spread_factor(-cap_pivot[global_t], active_idx, fwd_cum))  # smallest-first
            mom_factor.append(_decile_spread_factor(features[global_t, :, mom_idx], active_idx, fwd_cum))
        strategy_rets.append(wf["long_short_series"])
        logger.info("window %d: %d periods", w.window_id, wf["n_periods"])

    strategy_rets = np.concatenate(strategy_rets)
    size_factor = np.array(size_factor)
    mom_factor = np.array(mom_factor)

    valid = ~(np.isnan(size_factor) | np.isnan(mom_factor))
    strategy_rets, size_factor, mom_factor = strategy_rets[valid], size_factor[valid], mom_factor[valid]
    logger.info("Pooled %d valid periods for regression", len(strategy_rets))

    factors = np.column_stack([size_factor, mom_factor])
    fit = regress_on_factors(strategy_rets, factors)

    print("\n" + "=" * 90)
    print("FACTOR SANITY CHECK — is the long-short signal a size/momentum tilt in disguise?")
    print("=" * 90)
    print(f"  n_periods:              {len(strategy_rets)}")
    print(f"  strategy mean return:   {strategy_rets.mean():+.5f}")
    print(f"  R-squared (size+mom):   {fit['r_squared']:.4f}")
    print(f"  beta (size):            {fit['betas'][0]:+.4f}")
    print(f"  beta (momentum):        {fit['betas'][1]:+.4f}")
    print(f"  alpha (intercept):      {fit['alpha']:+.5f}")
    print(f"  residual mean:          {fit['residual_mean']:+.5f}")
    if fit["r_squared"] > 0.3:
        print("\n  -> meaningful factor exposure: a material share of the signal IS a size/momentum tilt")
    else:
        print("\n  -> low factor exposure: the signal is largely independent of size/momentum "
              "(doesn't mean it's ROBUST, just that it isn't explained by these two)")

    out = {
        "n_periods": len(strategy_rets), "strategy_mean_return": float(strategy_rets.mean()),
        "r_squared": fit["r_squared"], "beta_size": fit["betas"][0], "beta_momentum": fit["betas"][1],
        "alpha": fit["alpha"], "residual_mean": fit["residual_mean"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
