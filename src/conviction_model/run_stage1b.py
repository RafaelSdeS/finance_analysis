"""
run_stage1b.py -- Phase 1, Stage 1B's real training run (docs/conviction_model/CONVICTION_MODEL_PLAN.md):
CPC + forward cross-modal alignment, warm-started from a Stage 1A checkpoint. Same
panel/frame-cache/holdout plumbing as run_stage1a.py (imported, not duplicated) --
only the model init (warm-start instead of random) and the train/score step
(train_step_stage1b/score_holdout_stage1b instead of the CPC-only versions) differ.

Warm-start, not cold-restart: 1A already learned whatever CPC alone teaches; 1B's
question is "does ADDING alignment improve the diagnostics from here," not "can the
architecture learn from scratch under a different loss" (plan: "Warm-start from 1A;
retrain; rerun diagnostics; compare against 1A"). Universe (review finding 1) is
inherited unchanged from the warm-start checkpoint's own `tickers` field -- whatever
universe 1A trained on, 1B continues on -- so no separate universe fix is needed here.

Reserved Phase-7 holdout (review finding 11): same --reserved-holdout-years truncation
as run_stage1a.py (truncate_to_development_window, reused not duplicated), applied
here too so 1B's checkpoint-at-peak selection doesn't touch Phase 7's window either.

Run from project root:
    python -m src.conviction_model.run_stage1b
    python -m src.conviction_model.run_stage1b --warm-start-checkpoint artifacts/checkpoints/conviction_model/stage1a-20260721-165853.pt
"""

import argparse
import copy
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .config import SSLConfig
from .data import DAILY_FEATURES, MONTHLY_FEATURES, QUARTERLY_FEATURES, WEEKLY_FEATURES, build_frame_cache
from .encoder import EncoderCNN
from .run_stage1a import (
    CHECKPOINT_DIR, DEFAULT_RESERVED_HOLDOUT_YEARS, LOG_DIR, _holdout_eligible_count, _save_checkpoint,
    load_panel, setup_logging, truncate_to_development_window,
)
from .ssl_pretrain import (
    LazyPanelGatherer, build_stage1b_batch, sample_cpc_anchor_positions, score_holdout_stage1b, split_train_holdout,
    train_step_stage1b,
)

DEFAULT_STEPS = 5000  # matches run_stage1a.py's default budget -- same first-guess, adjustable via --steps


def _latest_stage1a_checkpoint() -> Path:
    checkpoints = sorted(CHECKPOINT_DIR.glob("stage1a-*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"no stage1a-*.pt checkpoints found in {CHECKPOINT_DIR} -- "
            "Stage 1B warm-starts from a Stage 1A checkpoint, run run_stage1a.py first")
    return checkpoints[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-start-checkpoint", type=str, default=None,
                         help="defaults to the most recent stage1a-*.pt checkpoint")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=500,
                         help="save a checkpoint every N steps, in addition to the final save")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                         help="defaults to artifacts/checkpoints/conviction_model/stage1b-<run_id>.pt")
    parser.add_argument("--reserved-holdout-years", type=float, default=DEFAULT_RESERVED_HOLDOUT_YEARS,
                         help="years at the end of the dataset reserved for Phase 7's final "
                              "holdout -- excluded from this run entirely (not just training, "
                              "also checkpoint-at-peak's holdout-eval anchor pool)")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = setup_logging(run_id, stage="stage1b")
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else CHECKPOINT_DIR / f"stage1b-{run_id}.pt"
    warm_start_path = Path(args.warm_start_checkpoint) if args.warm_start_checkpoint else _latest_stage1a_checkpoint()

    log.info(f"Warm-start checkpoint: {warm_start_path}")
    ckpt = torch.load(warm_start_path, map_location="cpu")
    # Reuse Stage 1A's exact hyperparameters (d_model/n_heads must match for the
    # warm-started weights to load; batch_size/cpc_horizon/temperature/etc. carry
    # over too so 1B stays a controlled "same setup + one new loss" comparison) --
    # alignment_weight isn't in the old config dict, so it falls back to today's
    # dataclass default (dataclasses.replace-style, no backward-compat shim needed).
    cfg = SSLConfig(**ckpt["config"])
    tickers = ckpt["tickers"]
    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    log.info(f"Universe: {len(tickers)} tickers (inherited from warm-start checkpoint)")
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

    max_horizon = max(cfg.cpc_horizon, cfg.alignment_horizon)
    train_panel, holdout_panel = split_train_holdout(panel, cfg.checkpoint_holdout_days)
    holdout_enabled = _holdout_eligible_count(holdout_panel, max_horizon) >= cfg.batch_size
    if not holdout_enabled:
        log.warning(f"holdout too small for checkpoint_holdout_days={cfg.checkpoint_holdout_days} "
                     "(not enough eligible rows) -- checkpoint-at-peak disabled, training on the full panel")
        train_panel = panel
    else:
        log.info(f"Train: {len(train_panel)} rows. Holdout: {len(holdout_panel)} rows "
                 f"(trailing {cfg.checkpoint_holdout_days} calendar days, never trained on)")

    train_store = LazyPanelGatherer(train_panel, frame_cache)
    holdout_store = LazyPanelGatherer(holdout_panel, frame_cache) if holdout_enabled else None

    model = EncoderCNN(len(DAILY_FEATURES), len(WEEKLY_FEATURES), len(MONTHLY_FEATURES),
                        len(QUARTERLY_FEATURES), d_model=cfg.d_model, n_heads=cfg.n_heads)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    losses = []
    best_step, best_score, best_state = None, None, None
    t_start = time.monotonic()
    for step in range(args.steps):
        anchors = sample_cpc_anchor_positions(train_panel, cfg.batch_size, max_horizon, rng=np_rng)
        anchor_batch, cpc_positive_batch, align_positive_batch, negative_batch = build_stage1b_batch(
            train_panel, train_store, anchors, cfg.cpc_horizon, cfg.alignment_horizon,
            n_same_stock=cfg.n_same_stock_negatives, n_diff_stock=cfg.n_diff_stock_negatives,
            regime_gap_days=cfg.regime_gap_days, rng=np_rng)
        loss = train_step_stage1b(model, optimizer, anchor_batch, cpc_positive_batch, align_positive_batch,
                                   negative_batch, temperature=cfg.temperature, alignment_weight=cfg.alignment_weight)
        losses.append(loss)
        if step % args.log_every == 0 or step == args.steps - 1:
            recent = losses[-args.log_every:]
            mean_total = sum(l["total"] for l in recent) / len(recent)
            mean_cpc = sum(l["cpc"] for l in recent) / len(recent)
            mean_align = sum(l["alignment"] for l in recent) / len(recent)
            log.info(f"step {step:>6}/{args.steps}  total={loss['total']:.4f}  "
                     f"mean_recent(total={mean_total:.4f}, cpc={mean_cpc:.4f}, alignment={mean_align:.4f})  "
                     f"elapsed={time.monotonic() - t_start:.0f}s")

        if holdout_enabled and ((step + 1) % cfg.checkpoint_eval_every == 0 or step == args.steps - 1):
            score = score_holdout_stage1b(model, holdout_panel, holdout_store, cfg, np_rng)
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
    log.info(f"Loss (total): first={losses[0]['total']:.4f}  last={losses[-1]['total']:.4f}  "
             f"min={min(l['total'] for l in losses):.4f}")
    if best_state is not None:
        log.info(f"Final checkpoint holds the BEST-scoring weights: step {best_step}, "
                 f"holdout loss={best_score:.4f} (not the last step's weights)")
    log.info(f"Final checkpoint: {checkpoint_path}")
    log.info(f"Log file: {LOG_DIR / f'stage1b-{run_id}.log'}")


if __name__ == "__main__":
    main()
