"""
run_stage1a.py -- Phase 1, Stage 1A's real training run (docs/conviction_model/CONVICTION_MODEL_PLAN.md):
CPC-only pretraining on a real ticker universe, not the pilot/dry-run (pretrain_pilot.py). Builds
on the same plumbing the pilot rehearsed (CPCPanelStore, build_cpc_batch, train_step) and the
correctness/performance fixes it surfaced.

Default universe: a static snapshot of the top-150-by-volume tickers as of the most recent
quarterly rebalance in top150_universe_membership.parquet -- reused here only to pick "a
manageable, liquid ~150-name subset" per the user's request. The encoder itself has no
architectural reason to be restricted to this set (see Data & universe -- it normally sees all
515 tickers); this is a deliberate scoping choice for the first real run, not the label
universe's point-in-time union-recovers-delisted construction (that's ~360 names across all of
history, not "150").

Checkpoint + log per run, same convention as data_collection/pipeline.py::setup_logging():
timestamped filenames under artifacts/{checkpoints,logs}/conviction_model/, not git-tracked
(CLAUDE.md's existing artifacts/ convention). Checkpoint is saved periodically DURING training
(--checkpoint-every), not just at the end, so a killed/crashed run still leaves a usable
checkpoint on disk.

Run from project root:
    python -m src.conviction_model.run_stage1a
    python -m src.conviction_model.run_stage1a --steps 20000 --log-every 100 --checkpoint-every 500
"""

import argparse
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from .config import SSLConfig
from .data import DAILY_FEATURES, MONTHLY_FEATURES, QUARTERLY_FEATURES, WEEKLY_FEATURES, build_frame_cache
from .encoder import EncoderCNN
from .paths import DATASET_PATH, ROOT, TOP150_MEMBERSHIP_PATH
from .ssl_pretrain import LazyPanelGatherer, build_cpc_batch, sample_cpc_anchor_positions, train_step

CHECKPOINT_DIR = ROOT / "artifacts/checkpoints/conviction_model"
LOG_DIR = ROOT / "artifacts/logs/conviction_model"
DEFAULT_STEPS = 5000  # first-guess budget for this initial run -- arbitrary, adjustable via --steps


def setup_logging(run_id: str) -> logging.Logger:
    """Same convention as data_collection/pipeline.py::setup_logging(): dual
    stdout + file handler, so everything printed during a run is also saved
    for later study, not just visible while the terminal is open."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"stage1a-{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logfile)],
    )
    return logging.getLogger("stage1a")


def top150_snapshot_tickers(membership_path=TOP150_MEMBERSHIP_PATH) -> list:
    """The most recent quarterly rebalance period's ~150 tickers -- a static
    snapshot of "the top 150 tickers", not labels.py's full point-in-time
    union across all of history (that's ~360 names)."""
    membership = pd.read_parquet(membership_path, columns=["ticker", "period_id", "start"])
    latest_period = membership.loc[membership["start"].idxmax(), "period_id"]
    return sorted(membership.loc[membership["period_id"] == latest_period, "ticker"].tolist())


def load_panel(tickers) -> pd.DataFrame:
    table = pq.read_table(DATASET_PATH, columns=["ticker", "trade_date"],
                           filters=[("ticker", "in", list(tickers))])
    return table.to_pandas().sort_values(["ticker", "trade_date"]).reset_index(drop=True)


def _save_checkpoint(path: Path, model, cfg, tickers, losses, step, total_steps) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg),
                "tickers": tickers, "losses": losses, "step": step, "total_steps": total_steps}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=500,
                         help="save a checkpoint every N steps, in addition to the final save")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                         help="defaults to artifacts/checkpoints/conviction_model/stage1a-<run_id>.pt")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = setup_logging(run_id)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else CHECKPOINT_DIR / f"stage1a-{run_id}.pt"

    tickers = top150_snapshot_tickers()
    cfg = SSLConfig()
    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    log.info(f"Universe: {len(tickers)} tickers (top-150 snapshot, most recent rebalance period)")
    t0 = time.monotonic()
    panel = load_panel(tickers)
    log.info(f"Panel: {len(panel)} rows, {panel['ticker'].nunique()} tickers ({time.monotonic() - t0:.1f}s)")

    t0 = time.monotonic()
    frame_cache = build_frame_cache(tickers)
    log.info(f"Frame cache built ({time.monotonic() - t0:.1f}s)")

    # LazyPanelGatherer, not CPCPanelStore: a full precompute at this universe size (150
    # tickers, ~540k positions) is ~9GB resident (measured) -- adjacent windows for the same
    # ticker overlap almost entirely, so per-position storage is fundamentally redundant.
    # This computes each window on demand from the (tiny, ~tens of MB) per-ticker frame
    # cache instead -- slower per step, but bounded, safe memory.
    store = LazyPanelGatherer(panel, frame_cache)

    model = EncoderCNN(len(DAILY_FEATURES), len(WEEKLY_FEATURES), len(MONTHLY_FEATURES),
                        len(QUARTERLY_FEATURES), d_model=cfg.d_model, n_heads=cfg.n_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    losses = []
    t_start = time.monotonic()
    for step in range(args.steps):
        anchors = sample_cpc_anchor_positions(panel, cfg.batch_size, cfg.cpc_horizon, rng=np_rng)
        anchor_batch, positive_batch, negative_batch = build_cpc_batch(
            panel, store, anchors, cfg.cpc_horizon,
            n_same_stock=cfg.n_same_stock_negatives, n_diff_stock=cfg.n_diff_stock_negatives,
            regime_gap_days=cfg.regime_gap_days, rng=np_rng)
        loss = train_step(model, optimizer, anchor_batch, positive_batch, negative_batch,
                           temperature=cfg.temperature)
        losses.append(loss)
        if step % args.log_every == 0 or step == args.steps - 1:
            recent = losses[-args.log_every:]
            log.info(f"step {step:>6}/{args.steps}  loss={loss:.4f}  "
                     f"mean_recent={sum(recent) / len(recent):.4f}  "
                     f"elapsed={time.monotonic() - t_start:.0f}s")
        if step > 0 and (step % args.checkpoint_every == 0 or step == args.steps - 1):
            _save_checkpoint(checkpoint_path, model, cfg, tickers, losses, step, args.steps)
            log.info(f"checkpoint saved: {checkpoint_path} (step {step})")

    if not losses:
        return
    _save_checkpoint(checkpoint_path, model, cfg, tickers, losses, args.steps - 1, args.steps)
    log.info(f"Loss: first={losses[0]:.4f}  last={losses[-1]:.4f}  min={min(losses):.4f}")
    log.info(f"Final checkpoint: {checkpoint_path}")
    log.info(f"Log file: {LOG_DIR / f'stage1a-{run_id}.log'}")


if __name__ == "__main__":
    main()
