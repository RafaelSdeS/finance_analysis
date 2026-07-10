"""
DIAGNOSIS_PLAN.md Phase 0.3: does the feature signal (rank-IC historically measured at
0.03-0.10 on the full ~288-name universe) survive when restricted to the top-N the agent
actually trades? (F2)

For each rolling window, computes mean |rank-IC| + non-overlap t-stat across the agent's
state_features, over that window's test date range, twice: once using the full listed
universe, once restricted to that window's point-in-time top-N (data_pipeline.select_universe,
same selector as --pit-universe).

Usage:
    python tools/tradable_universe_ic_audit.py [--universe-size 50] [--horizon 21]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root, for `src.*` imports

from src.agent.config import DEFAULT_CONFIG, generate_windows, window_to_config
from src.agent.data_pipeline import select_universe
from src.agent.ic_analysis import daily_rank_ic, load_panel, summarize

logger = logging.getLogger(__name__)


def _aligned_window(feat_mat: pd.DataFrame, fwd_mat: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice both matrices to [start, end] and align to their common date index AND
    ticker columns.

    Fundamental-feature pivots (e.g. roe) have a SHORTER date index and FEWER ticker
    columns than price-derived ones: pivot_table drops any date/ticker slice that's
    entirely NaN for that feature (pre-2011 CVM digital-filing era; a handful of
    tickers with no fundamentals ever). daily_rank_ic requires feat_mat/fwd_mat to
    share an identical index and column set, so the intersection has to be taken per
    feature rather than reusing one boolean mask sized for the price-based matrices.
    """
    f = feat_mat.loc[(feat_mat.index >= pd.Timestamp(start)) & (feat_mat.index <= pd.Timestamp(end))]
    r = fwd_mat.loc[(fwd_mat.index >= pd.Timestamp(start)) & (fwd_mat.index <= pd.Timestamp(end))]
    common_idx = f.index.intersection(r.index)
    common_cols = f.columns.intersection(r.columns)
    return f.loc[common_idx, common_cols], r.loc[common_idx, common_cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tradable-universe IC audit (DIAGNOSIS_PLAN Phase 0.3)")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/ml_dataset.parquet"))
    parser.add_argument("--universe-size", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--min-names", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("artifacts/backtest/tradable_universe_ic.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    base = DEFAULT_CONFIG
    # 'returns' is the contemporaneous (t=0) return, not a forward-looking predictive
    # feature; excluded here the same way ic_analysis's own SKIP_COLS logic would treat
    # a realized outcome rather than a signal.
    feature_set = [f for f in base.state_features if f != "returns"]

    logger.info("Loading full panel (horizon=%dd, all candidate features)...", args.horizon)
    features, fwd_rets = load_panel(args.dataset, horizons=[args.horizon])
    fwd = fwd_rets[args.horizon]

    available = [f for f in feature_set if f in features]
    missing = [f for f in feature_set if f not in features]
    if missing:
        logger.warning("state_features not found in IC candidate set (skipped): %s", missing)
    logger.info("Auditing %d of %d state_features: %s", len(available), len(feature_set), available)

    caps = pd.read_parquet(args.dataset, columns=["ticker", "trade_date", "market_cap"])
    all_tickers = sorted(set(fwd.columns))

    windows = generate_windows(base.dataset_start, base.dataset_end, base.window_train_years, base.window_test_years)

    rows = []
    for w in windows:
        wc = window_to_config(w, base)
        pit_universe = select_universe(caps, all_tickers, args.universe_size, cutoff=wc.train_end)

        full_abs_ics, pit_abs_ics, full_ts, pit_ts = [], [], [], []
        for feat_name in available:
            feat_window, fwd_window = _aligned_window(features[feat_name], fwd, wc.test_start, wc.test_end)

            s_full = summarize(daily_rank_ic(feat_window, fwd_window, args.min_names), args.horizon, feat_name)
            full_abs_ics.append(abs(s_full.get("mean_ic", 0.0)))
            full_ts.append(s_full.get("non_overlap_t_stat", 0.0))

            pit_cols = [c for c in pit_universe if c in feat_window.columns and c in fwd_window.columns]
            s_pit = summarize(
                daily_rank_ic(feat_window[pit_cols], fwd_window[pit_cols], args.min_names),
                args.horizon, feat_name,
            )
            pit_abs_ics.append(abs(s_pit.get("mean_ic", 0.0)))
            pit_ts.append(s_pit.get("non_overlap_t_stat", 0.0))

        rows.append({
            "window": w.window_id,
            "test_start": wc.test_start,
            "test_end": wc.test_end,
            "n_pit_tickers": len(pit_universe),
            "full_mean_abs_ic": float(np.mean(full_abs_ics)),
            "pit_mean_abs_ic": float(np.mean(pit_abs_ics)),
            "full_mean_nonoverlap_t": float(np.mean(full_ts)),
            "pit_mean_nonoverlap_t": float(np.mean(pit_ts)),
        })
        logger.info("  window %d done", w.window_id)

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print("\n" + "=" * 100)
    print(f"TRADABLE-UNIVERSE IC AUDIT — full listed universe vs PIT top-{args.universe_size}, "
          f"per window, horizon={args.horizon}d, mean |rank-IC| across {len(available)} state_features")
    print("=" * 100)
    print(table.round(4).to_string(index=False))
    print(f"\nMean across windows: full |IC|={table['full_mean_abs_ic'].mean():.4f}  "
          f"pit |IC|={table['pit_mean_abs_ic'].mean():.4f}")
    print(f"Mean non-overlap t-stat: full={table['full_mean_nonoverlap_t'].mean():.2f}  "
          f"pit={table['pit_mean_nonoverlap_t'].mean():.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(table.to_dict(orient="records"), f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
