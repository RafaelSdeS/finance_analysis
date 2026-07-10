"""
DIAGNOSIS_PLAN.md Phase 0.2: quantify the survivorship/lookahead premium (F1).

Rolls a plain equal-weight strategy through each rolling window's test span
twice: once using the CURRENT universe selector (top-N by full-sample mean
market cap — the same static universe for every window, includes future
information), once using a genuinely point-in-time (PIT) universe (top-N by
mean market cap using only data <= that window's train_end). The gap between
the two is the survivorship-lookahead premium baked into every backtest
number produced before this fix.

No production artifacts are touched: tensors are built in-memory only
(build_tensors), nothing is written to data/processed/agent_tensors.npz.

Usage:
    python tools/quantify_survivorship_bias.py [--universe-size 50]
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
from src.agent.metrics import sharpe_ratio, cumulative_return, annualized_return

logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 100_000.0


def _ew_log_returns(tensors: dict, mask: np.ndarray, start: str, end: str) -> np.ndarray:
    """Equal-weight (active, non-CASH) daily log-return series over [start, end],
    using the supplied mask instead of tensors['mask'] (so callers can pass a
    PIT-restricted mask without mutating the shared tensors dict)."""
    dates = pd.to_datetime(tensors["dates"])
    sel = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    is_cash = tensors["tickers"] == "CASH"

    returns = tensors["returns"][sel]
    sliced_mask = mask[sel]
    active_noncash = sliced_mask & ~is_cash

    simple = np.where(active_noncash, np.expm1(np.nan_to_num(returns, nan=0.0)), 0.0)
    n_active = active_noncash.sum(axis=1)
    ew_simple = np.where(n_active > 0, simple.sum(axis=1) / np.maximum(n_active, 1), 0.0)
    return np.log1p(np.maximum(ew_simple, -0.9999))


def _values(log_returns: np.ndarray) -> np.ndarray:
    return INITIAL_CAPITAL * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify survivorship/lookahead bias (DIAGNOSIS_PLAN Phase 0.2)")
    parser.add_argument("--universe-size", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("artifacts/backtest/survivorship_bias.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    base = DEFAULT_CONFIG
    current_config = dataclasses.replace(base, universe_size=args.universe_size, pit_universe=False)
    pit_config = dataclasses.replace(base, universe_size=args.universe_size, pit_universe=True)

    logger.info("Building current-mode (full-sample top-%d) tensors...", args.universe_size)
    current_tensors = build_tensors(current_config)
    logger.info("Building point-in-time (top-%d per window) tensors...", args.universe_size)
    pit_tensors = build_tensors(pit_config)

    pit_cutoffs = list(pit_tensors["pit_cutoffs"])
    pit_membership = pit_tensors["pit_membership_mask"]

    windows = generate_windows(base.dataset_start, base.dataset_end, base.window_train_years, base.window_test_years)

    rows = []
    for w in windows:
        wc = window_to_config(w, base)
        cutoff = wc.train_end

        cur_log_rets = _ew_log_returns(current_tensors, current_tensors["mask"], wc.test_start, wc.test_end)

        if cutoff not in pit_cutoffs:
            logger.warning("window %d: no PIT membership row for cutoff=%s, skipping", w.window_id, cutoff)
            continue
        pit_row = pit_membership[pit_cutoffs.index(cutoff)]
        pit_mask = pit_tensors["mask"] & pit_row[np.newaxis, :]
        pit_log_rets = _ew_log_returns(pit_tensors, pit_mask, wc.test_start, wc.test_end)

        cur_values, pit_values = _values(cur_log_rets), _values(pit_log_rets)
        cur_cum, pit_cum = cumulative_return(cur_values), cumulative_return(pit_values)
        cur_sharpe, pit_sharpe = sharpe_ratio(cur_log_rets), sharpe_ratio(pit_log_rets)

        rows.append({
            "window": w.window_id,
            "test_start": wc.test_start,
            "test_end": wc.test_end,
            "current_cum_return": cur_cum,
            "pit_cum_return": pit_cum,
            "gap_cum_return": cur_cum - pit_cum,
            "current_annualized_return": annualized_return(cur_values),
            "pit_annualized_return": annualized_return(pit_values),
            "current_sharpe": cur_sharpe,
            "pit_sharpe": pit_sharpe,
            "gap_sharpe": cur_sharpe - pit_sharpe,
        })

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print("\n" + "=" * 100)
    print(f"SURVIVORSHIP / LOOKAHEAD BIAS — equal-weight, current (full-sample top-{args.universe_size}) "
          f"vs PIT top-{args.universe_size}, per window")
    print("=" * 100)
    print(table.round(4).to_string(index=False))
    print(f"\nMean gap (current − PIT), cumulative return: {table['gap_cum_return'].mean():+.4f}")
    print(f"Mean gap (current − PIT), Sharpe:             {table['gap_sharpe'].mean():+.4f}")
    print(f"Windows where current-mode overstates Sharpe: "
          f"{(table['gap_sharpe'] > 0).sum()}/{len(table)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(table.to_dict(orient="records"), f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
