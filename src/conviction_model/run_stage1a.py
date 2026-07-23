"""
run_stage1a.py -- Phase 1, Stage 1A's real training run (docs/conviction_model/CONVICTION_MODEL_PLAN.md):
CPC-only pretraining on a real ticker universe, not the pilot/dry-run (pretrain_pilot.py). Builds
on the same plumbing the pilot rehearsed (CPCPanelStore, build_cpc_batch, train_step) and the
correctness/performance fixes it surfaced.

Default universe (fixed 2026-07-22, review finding 1): the point-in-time UNION of every ticker
that ever qualified for top-150 across top150_universe_membership.parquet's full history (~360
names) -- survivorship-safe by construction, same guarantee TOP50_UNIVERSE_VALIDATION.md already
established for the identical point-in-time construction at top-50. The PREVIOUS default (a
static snapshot of whichever ~150 tickers are top-150 as of the MOST RECENT rebalance) trained
the encoder only on today's survivors while still feeding it their full PAST history -- exactly
the survivorship-bias pattern the design's own "the encoder sees all names, unrestricted" intent
(Data & universe) was meant to avoid; any ticker delisted/merged/dropped out before the latest
rebalance was invisible to Stage 1A/1B training entirely. That old behavior is kept ONLY behind
--debug-snapshot for a fast plumbing smoke test, never for a real training run.

Reserved Phase-7 holdout (fixed 2026-07-22, review finding 11): the trailing
--reserved-holdout-years (default 2, matching the plan's Phase 7 "most recent ~1-2 years, run
exactly once") of the dataset are excluded from this run ENTIRELY before anything else happens
-- not just from training, but from checkpoint-at-peak's holdout-eval anchor pool too. Without
this, split_train_holdout's own "trailing checkpoint_holdout_days of the panel" silently reused
the SAME trailing window Phase 7 is supposed to independently validate against, so checkpoint
SELECTION (which state_dict gets kept) was already informed by performance on data Phase 7 is
meant to see fresh. Window tensors themselves were never at risk (window_tensor only reads
history at-or-before `as_of`, causal by construction) -- this only restricts which ANCHOR
positions are eligible to be trained on or scored at all.

Checkpoint + log per run, same convention as data_collection/pipeline.py::setup_logging():
timestamped filenames under artifacts/{checkpoints,logs}/conviction_model/, not git-tracked
(CLAUDE.md's existing artifacts/ convention). Checkpoint is saved periodically DURING training
(--checkpoint-every), not just at the end, so a killed/crashed run still leaves a usable
checkpoint on disk.

Checkpoint-at-peak (mirrors rl_agent/train.py::pretrain()): the trailing checkpoint_holdout_days
of the (already Phase-7-truncated) panel are carved out of training entirely, scored every
checkpoint_eval_every steps, and the best-scoring state is what actually gets restored into the
final checkpoint -- not just whatever the last step happened to land on. See config.py's
SSLConfig for the rationale (CLAUDE.md's measured rl_agent case: same seed, +47% at 100k steps
-> -71% at 2M, pure overfitting).

Run from project root:
    python -m src.conviction_model.run_stage1a
    python -m src.conviction_model.run_stage1a --steps 20000 --log-every 100 --checkpoint-every 500
    python -m src.conviction_model.run_stage1a --debug-snapshot --steps 200  # fast plumbing smoke test only
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
    CachedPanelGatherer, build_cpc_batch, sample_cpc_anchor_positions, score_holdout,
    split_train_holdout, train_step,
)

CHECKPOINT_DIR = ROOT / "artifacts/checkpoints/conviction_model"
LOG_DIR = ROOT / "artifacts/logs/conviction_model"
DEFAULT_STEPS = 5000  # first-guess budget for this initial run -- arbitrary, adjustable via --steps
DEFAULT_RESERVED_HOLDOUT_YEARS = 2.0  # matches the plan's Phase 7 "most recent ~1-2 years"
DEFAULT_PANEL_CACHE_SIZE = 50_000  # positions PER STORE; train+holdout stores are independent,
                                    # so total resident cache =~ 2x this (see BYTES_PER_CACHED_POSITION)
BYTES_PER_CACHED_POSITION = 18_000  # empirically measured via tracemalloc (float32 cache entries,
                                     # ~17.65KB observed; rounded up for a small safety margin) --
                                     # NOT the ~17KB CPCPanelStore docstring figure blindly reused
                                     # (that one already assumes float32; a prior version of this
                                     # constant caused a real OOM by caching float64 arrays at ~2x
                                     # this size before the float32 cast was added -- see git history)


def setup_logging(run_id: str, stage: str = "stage1a") -> logging.Logger:
    """Same convention as data_collection/pipeline.py::setup_logging(): dual
    stdout + file handler, so everything printed during a run is also saved
    for later study, not just visible while the terminal is open. `stage`
    param (default "stage1a") lets run_stage1b.py and later stages reuse this
    exact setup instead of duplicating it, just naming their own log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"{stage}-{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logfile)],
    )
    return logging.getLogger(stage)


def point_in_time_union_tickers(membership_path=TOP150_MEMBERSHIP_PATH) -> list:
    """Every ticker that EVER qualified for top-150 at any point-in-time rebalance across
    the membership file's full history (~360 names) -- survivorship-safe by construction: a
    name that was top-150 in, say, 2015 and later delisted or dropped out is still included,
    so the encoder doesn't train only on "whoever happens to be top-150 today" (review
    finding 1). This is the DEFAULT universe for Stage 1A/1B; top150_snapshot_tickers()
    below is kept only for --debug-snapshot fast smoke tests, never a real training run."""
    membership = pd.read_parquet(membership_path, columns=["ticker"])
    return sorted(membership["ticker"].unique().tolist())


def top150_snapshot_tickers(membership_path=TOP150_MEMBERSHIP_PATH) -> list:
    """DEBUG/SMOKE-TEST ONLY (--debug-snapshot) -- the most recent quarterly rebalance
    period's ~150 tickers, a static CURRENT-DAY snapshot. NOT survivorship-safe: excludes
    every name that was ever top-150 but isn't today, while still feeding the encoder full
    PAST history for whichever names happen to survive to the present -- exactly the
    pattern point_in_time_union_tickers (the real default) exists to avoid. Small/fast,
    fine for a quick plumbing smoke test; do not use for an actual training run."""
    membership = pd.read_parquet(membership_path, columns=["ticker", "period_id", "start"])
    latest_period = membership.loc[membership["start"].idxmax(), "period_id"]
    return sorted(membership.loc[membership["period_id"] == latest_period, "ticker"].tolist())


def load_panel(tickers) -> pd.DataFrame:
    table = pq.read_table(DATASET_PATH, columns=["ticker", "trade_date"],
                           filters=[("ticker", "in", list(tickers))])
    return table.to_pandas().sort_values(["ticker", "trade_date"]).reset_index(drop=True)


def truncate_to_development_window(panel: pd.DataFrame,
                                    reserved_holdout_years: float) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Drops every row dated after `dataset_end - reserved_holdout_years` -- keeps Phase 7's
    final holdout (CONVICTION_MODEL_PLAN.md: "the most recent ~1-2 years, run exactly once")
    completely untouched by every earlier phase, not just by TRAINING but by checkpoint-at-
    peak's holdout-eval anchor eligibility too (review finding 11). Returns
    (truncated_panel, dataset_end, dev_end) so the caller can log the excluded span."""
    dataset_end = panel["trade_date"].max()
    dev_end = dataset_end - pd.DateOffset(years=reserved_holdout_years)
    truncated = panel.loc[panel["trade_date"] <= dev_end].reset_index(drop=True)
    return truncated, dataset_end, dev_end


def _save_checkpoint(path: Path, state_dict, cfg, tickers, losses, step, total_steps,
                      best_step=None, best_score=None, extra: dict | None = None) -> None:
    """`extra`: additional top-level keys merged into the saved dict (e.g. Stage 1C's
    recon_heads_state_dict) -- optional so 1A/1B's calls are unaffected."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": state_dict, "config": asdict(cfg), "tickers": tickers,
               "losses": losses, "step": step, "total_steps": total_steps,
               "best_step": best_step, "best_score": best_score}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


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
    parser.add_argument("--debug-snapshot", action="store_true",
                         help="use the small, CURRENT-DAY top-150 snapshot instead of the "
                              "point-in-time union (~360 names) -- fast plumbing smoke-test "
                              "only, NOT survivorship-safe, never use for a real training run")
    parser.add_argument("--reserved-holdout-years", type=float, default=DEFAULT_RESERVED_HOLDOUT_YEARS,
                         help="years at the end of the dataset reserved for Phase 7's final "
                              "holdout -- excluded from this run entirely (not just training, "
                              "also checkpoint-at-peak's holdout-eval anchor pool)")
    parser.add_argument("--panel-cache-size", type=int, default=DEFAULT_PANEL_CACHE_SIZE,
                         help="max (ticker, date) positions memoized PER STORE -- train and "
                              "holdout each get their own independent cache, so total resident "
                              "memory is ~2x this value at ~18KB/position (e.g. the default "
                              f"{DEFAULT_PANEL_CACHE_SIZE} =~ "
                              f"{2 * DEFAULT_PANEL_CACHE_SIZE * BYTES_PER_CACHED_POSITION / 1e9:.1f}GB "
                              "combined); 0 effectively disables caching (every position "
                              "recomputed every time, lowest memory, slowest per-step)")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = setup_logging(run_id)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else CHECKPOINT_DIR / f"stage1a-{run_id}.pt"

    tickers = top150_snapshot_tickers() if args.debug_snapshot else point_in_time_union_tickers()
    cfg = SSLConfig()
    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    universe_desc = ("DEBUG top-150 snapshot, most recent rebalance period -- NOT survivorship-safe"
                      if args.debug_snapshot else "point-in-time union across full history, survivorship-safe")
    log.info(f"Universe: {len(tickers)} tickers ({universe_desc})")
    t0 = time.monotonic()
    panel = load_panel(tickers)
    log.info(f"Panel (full history): {len(panel)} rows, {panel['ticker'].nunique()} tickers "
             f"({time.monotonic() - t0:.1f}s)")

    panel, dataset_end, dev_end = truncate_to_development_window(panel, args.reserved_holdout_years)
    log.info(f"Reserved Phase-7 holdout: excluding {dev_end.date()} -> {dataset_end.date()} "
             f"({args.reserved_holdout_years}y) from this run entirely. "
             f"Development panel: {len(panel)} rows, ends {dev_end.date()}")

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

    # CachedPanelGatherer, not CPCPanelStore: a full precompute was already ~9GB resident
    # (measured) at the OLD 150-ticker snapshot universe; the ~360-name point-in-time union
    # is larger still -- too large to precompute the whole panel upfront on most machines.
    # CPC/alignment sample WITH REPLACEMENT from the same panel every step, so many
    # positions repeat over a multi-thousand-step run; a BOUNDED LRU cache captures those
    # repeat hits without the unbounded-with-universe-size memory risk (see
    # ssl_pretrain.py::CachedPanelGatherer).
    n_stores = 2 if holdout_enabled else 1  # train_store + holdout_store are each capped
                                             # INDEPENDENTLY -- report the COMBINED estimate,
                                             # not a per-store number (a per-store-only estimate
                                             # previously caused a real OOM: it silently implied
                                             # half the actual peak)
    est_gb = n_stores * args.panel_cache_size * BYTES_PER_CACHED_POSITION / 1e9
    log.info(f"Panel cache: up to {args.panel_cache_size} positions/store x {n_stores} store(s) "
             f"(~{est_gb:.1f}GB estimated COMBINED peak)")
    train_store = CachedPanelGatherer(train_panel, frame_cache, maxsize=args.panel_cache_size)
    holdout_store = CachedPanelGatherer(holdout_panel, frame_cache, maxsize=args.panel_cache_size) \
        if holdout_enabled else None

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
