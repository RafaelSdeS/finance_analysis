"""
pretrain_pilot.py -- Phase 1's pilot/dry-run (docs/conviction_model/CONVICTION_MODEL_PLAN.md),
same convention as rl_agent/experiment.py --dry-run: rehearse the real plumbing on a small
slice before the full run, to catch bugs and get a wall-clock estimate.

Scoped to what's actually built so far -- Stage 1A (CPC) only. The full pilot item in the
plan ("all 4 stages, all diagnostics, the trajectory plot") can't run yet: Stages 1B-1D's
losses aren't written. This rehearses real-data batch assembly (data.py + ssl_pretrain.py's
build_cpc_batch) + a handful of CPC train_step calls end to end, nothing more.

Run from project root:
    python -m src.conviction_model.pretrain_pilot
"""

import time

import numpy as np
import pyarrow.parquet as pq
import torch

from .config import SSLConfig
from .data import DAILY_FEATURES, MONTHLY_FEATURES, QUARTERLY_FEATURES, WEEKLY_FEATURES, build_frame_cache
from .encoder import EncoderCNN
from .paths import DATASET_PATH
from .ssl_pretrain import CPCPanelStore, build_cpc_batch, sample_cpc_anchor_positions, train_step

PILOT_TICKERS = ("PETR4", "VALE3", "ITUB4", "BBAS3", "ABEV3")
PILOT_STEPS = 20


def load_pilot_panel(tickers=PILOT_TICKERS):
    table = pq.read_table(DATASET_PATH, columns=["ticker", "trade_date"],
                           filters=[("ticker", "in", list(tickers))])
    return table.to_pandas().sort_values(["ticker", "trade_date"]).reset_index(drop=True)


def main() -> None:
    cfg = SSLConfig()
    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    print(f"Pilot tickers: {PILOT_TICKERS}")
    t0 = time.monotonic()
    panel = load_pilot_panel()
    print(f"Panel: {len(panel)} rows, {panel['ticker'].nunique()} tickers "
          f"({time.monotonic() - t0:.2f}s)")

    t0 = time.monotonic()
    frame_cache = build_frame_cache(PILOT_TICKERS)
    print(f"Frame cache built ({time.monotonic() - t0:.2f}s)")

    t0 = time.monotonic()
    store = CPCPanelStore(panel, frame_cache)
    print(f"Panel store precomputed: {len(panel)} positions "
          f"({time.monotonic() - t0:.2f}s, one-time cost)")

    model = EncoderCNN(len(DAILY_FEATURES), len(WEEKLY_FEATURES), len(MONTHLY_FEATURES),
                        len(QUARTERLY_FEATURES), d_model=cfg.d_model, n_heads=cfg.n_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    losses = []
    batch_times, step_times = [], []
    for step in range(PILOT_STEPS):
        t0 = time.monotonic()
        anchors = sample_cpc_anchor_positions(panel, cfg.batch_size, cfg.cpc_horizon, rng=np_rng)
        anchor_batch, positive_batch, negative_batch = build_cpc_batch(
            panel, store, anchors, cfg.cpc_horizon,
            n_same_stock=cfg.n_same_stock_negatives, n_diff_stock=cfg.n_diff_stock_negatives,
            regime_gap_days=cfg.regime_gap_days, rng=np_rng)
        batch_times.append(time.monotonic() - t0)

        t0 = time.monotonic()
        loss = train_step(model, optimizer, anchor_batch, positive_batch, negative_batch,
                           temperature=cfg.temperature)
        step_times.append(time.monotonic() - t0)
        losses.append(loss)
        print(f"  step {step:>3}  loss={loss:.4f}  "
              f"batch={batch_times[-1]*1000:.0f}ms  grad_step={step_times[-1]*1000:.0f}ms")

    print(f"\nLoss: first={losses[0]:.4f}  last={losses[-1]:.4f}  "
          f"min={min(losses):.4f}  finite={all(l == l and l != float('inf') for l in losses)}")
    print(f"Mean batch-assembly time: {sum(batch_times)/len(batch_times)*1000:.0f}ms/step")
    print(f"Mean grad-step time: {sum(step_times)/len(step_times)*1000:.0f}ms/step")


if __name__ == "__main__":
    main()
