"""
DIAGNOSIS_PLAN.md Phase 1.2: alpha waterfall, per rolling window and pooled -- isolates
WHERE the raw cross-sectional signal (0.3/0.4) is destroyed by realistic portfolio
construction, mechanically explaining 1.1's finding that top-k EW loses to plain EW.

Reuses the SAME per-window teacher training as 1.1 (tools/supervised_strategy_backtest.py)
so the waterfall is scored on the identical predictions, not a separately-fit model.
Uses k=5 (not 1.1's 10/20) so the 0.10 cap stage is actually triggered (1/5=0.20 > cap).

Usage:
    python tools/alpha_waterfall.py [--universe-size 50] [--horizon 21] [--k 5]
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
from src.agent.supervised_strategy import compute_alpha_waterfall
from tools.supervised_strategy_backtest import _simple_returns_from_tensors, _train_teacher

logger = logging.getLogger(__name__)

STAGES = ["long_short", "long_only", "long_only_capped", "plus_costs", "plus_holding"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha waterfall per rolling window (DIAGNOSIS_PLAN Phase 1.2)")
    parser.add_argument("--universe-size", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--k", type=int, default=5, help="Long/short leg size (small enough to trigger the 0.10 cap)")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--rebalance-days", type=int, default=21)
    parser.add_argument("--out", type=Path, default=Path("artifacts/backtest/alpha_waterfall.json"))
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

    windows = generate_windows(base.dataset_start, base.dataset_end, base.window_train_years, base.window_test_years)

    rows = []
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

        active_test = active_full[test_idx]
        simple_test = simple_rets[test_idx]
        scores_test = np.zeros((len(test_idx), len(tickers)))
        for i, t in enumerate(test_idx):
            active_t = active_full[t]
            if active_t.any():
                scores_test[i, active_t] = model.predict(features[t][active_t][:, feature_idx])

        wf = compute_alpha_waterfall(scores_test, active_test, simple_test, k=args.k,
                                      cap=base.max_position_weight, cost_bps=args.cost_bps,
                                      rebalance_interval_days=args.rebalance_days)
        rows.append({"window": w.window_id, "test_start": wc.test_start, "test_end": wc.test_end, **wf})
        logger.info("  window %d: %s", w.window_id, {s: round(wf[s], 5) for s in STAGES})

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print("\n" + "=" * 120)
    print(f"ALPHA WATERFALL — per window (k={args.k}, mean period excess return vs EW-of-active)")
    print("=" * 120)
    print(table.round(5).to_string(index=False))

    # Pooled: weight each window's stage value by its own n_periods (fair average across
    # windows with different numbers of rebalance decisions).
    weights = table["n_periods"].to_numpy(dtype=np.float64)
    pooled = {s: float(np.average(table[s], weights=weights)) for s in STAGES}
    print("\n" + "=" * 120)
    print("POOLED (n_periods-weighted mean across windows)")
    print("=" * 120)
    for s in STAGES:
        print(f"  {s:<20} {pooled[s]:+.5f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"per_window": rows, "pooled": pooled}, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
