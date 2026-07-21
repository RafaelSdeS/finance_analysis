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

Checkpoint-at-peak (mirrors rl_agent/train.py::pretrain()): the trailing checkpoint_holdout_days
of the panel are carved out of training entirely, scored every checkpoint_eval_every steps, and
the best-scoring state is what actually gets restored into the final checkpoint -- not just
whatever the last step happened to land on. See config.py's SSLConfig for the rationale (CLAUDE.md's
measured rl_agent case: same seed, +47% at 100k steps -> -71% at 2M, pure overfitting).

Run from project root:
    python -m src.conviction_model.run_stage1a
    python -m src.conviction_model.run_stage1a --steps 20000 --log-every 100 --checkpoint-every 500
"""

import argparse
import copy
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
from .ssl_pretrain import (
    LazyPanelGatherer, build_cpc_batch, sample_cpc_anchor_positions, score_holdout,
    split_train_holdout, train_step,
)

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


def _save_checkpoint(path: Path, state_dict, cfg, tickers, losses, step, total_steps,
                      best_step=None, best_score=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state_dict, "config": asdict(cfg), "tickers": tickers,
                "losses": losses, "step": step, "total_steps": total_steps,
                "best_step": best_step, "best_score": best_score}, path)


def _holdout_eligible_count(holdout_panel: pd.DataFrame, cpc_horizon: int) -> int:
    """How many holdout rows could serve as a CPC anchor (mirrors
    sample_cpc_anchor_positions' own eligibility rule) -- used only to
    decide up front whether the holdout is big enough for even one eval
    batch, without duplicating the sampling itself."""
    if holdout_panel.empty:
        return 0
    remaining = holdout_panel.groupby("ticker").cumcount(ascending=False).to_numpy()
    return int((remaining >= cpc_horizon).sum())


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

    train_panel, holdout_panel = split_train_holdout(panel, cfg.checkpoint_holdout_days)
    holdout_enabled = _holdout_eligible_count(holdout_panel, cfg.cpc_horizon) >= cfg.batch_size
    if not holdout_enabled:
        log.warning(f"holdout too small for checkpoint_holdout_days={cfg.checkpoint_holdout_days} "
                     "(not enough eligible rows) -- checkpoint-at-peak disabled, training on the full panel")
        train_panel = panel
    else:
        log.info(f"Train: {len(train_panel)} rows. Holdout: {len(holdout_panel)} rows "
                 f"(trailing {cfg.checkpoint_holdout_days} calendar days, never trained on)")

    # LazyPanelGatherer, not CPCPanelStore: a full precompute at this universe size (150
    # tickers, ~540k positions) is ~9GB resident (measured) -- adjacent windows for the same
    # ticker overlap almost entirely, so per-position storage is fundamentally redundant.
    # This computes each window on demand from the (tiny, ~tens of MB) per-ticker frame
    # cache instead -- slower per step, but bounded, safe memory.
    train_store = LazyPanelGatherer(train_panel, frame_cache)
    holdout_store = LazyPanelGatherer(holdout_panel, frame_cache) if holdout_enabled else None

    model = EncoderCNN(len(DAILY_FEATURES), len(WEEKLY_FEATURES), len(MONTHLY_FEATURES),
                        len(QUARTERLY_FEATURES), d_model=cfg.d_model, n_heads=cfg.n_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    losses = []
    best_step, best_score, best_state = None, None, None
    t_start = time.monotonic()
    for step in range(args.steps):
        anchors = sample_cpc_anchor_positions(train_panel, cfg.batch_size, cfg.cpc_horizon, rng=np_rng)
        anchor_batch, positive_batch, negative_batch = build_cpc_batch(
            train_panel, train_store, anchors, cfg.cpc_horizon,
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

        if holdout_enabled and ((step + 1) % cfg.checkpoint_eval_every == 0 or step == args.steps - 1):
            score = score_holdout(model, holdout_panel, holdout_store, cfg, np_rng)
            log.info(f"  holdout eval at step {step}: loss={score:.4f}"
                     + (f" (new best, was {best_score:.4f})" if best_score is not None and score < best_score
                        else " (new best)" if best_score is None else ""))
            if best_score is None or score < best_score:
                best_step, best_score = step, score
                best_state = copy.deepcopy(model.state_dict())

        if step > 0 and (step % args.checkpoint_every == 0 or step == args.steps - 1):
            _save_checkpoint(checkpoint_path, model.state_dict(), cfg, tickers, losses, step, args.steps,
                              best_step, best_score)
            log.info(f"checkpoint saved: {checkpoint_path} (step {step}, latest weights)")

    if not losses:
        return
    final_state = best_state if best_state is not None else model.state_dict()
    _save_checkpoint(checkpoint_path, final_state, cfg, tickers, losses, args.steps - 1, args.steps,
                      best_step, best_score)
    log.info(f"Loss: first={losses[0]:.4f}  last={losses[-1]:.4f}  min={min(losses):.4f}")
    if best_state is not None:
        log.info(f"Final checkpoint holds the BEST-scoring weights: step {best_step}, "
                 f"holdout loss={best_score:.4f} (not the last step's weights)")
    log.info(f"Final checkpoint: {checkpoint_path}")
    log.info(f"Log file: {LOG_DIR / f'stage1a-{run_id}.log'}")


if __name__ == "__main__":
    main()
