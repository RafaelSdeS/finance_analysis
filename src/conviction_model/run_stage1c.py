"""
run_stage1c.py -- Phase 1, Stage 1C's real training run (docs/conviction_model/CONVICTION_MODEL_PLAN.md):
CPC + forward cross-modal alignment + masked reconstruction across branches, warm-started
from a Stage 1B checkpoint. Same panel/frame-cache/holdout plumbing as run_stage1a.py/
run_stage1b.py (imported, not duplicated) -- only the model init (warm-start from 1B,
plus a fresh ReconstructionHeads module) and the train/score step (train_step_stage1c/
score_holdout_stage1c) differ.

Warm-start, not cold-restart: same reasoning as run_stage1b.py -- 1B already learned
whatever CPC+alignment together teach; 1C's question is "does ADDING masked reconstruction
improve the diagnostics from here." Universe is inherited unchanged from the warm-start
checkpoint's own `tickers` field. Reserved Phase-7 holdout: same --reserved-holdout-years
truncation as run_stage1a.py/run_stage1b.py.

ReconstructionHeads (ssl_pretrain.py) is a small extra module (one Linear(d_model,d_model)
per branch) with its own parameters, not part of EncoderCNN -- it has no warm-start source
(1A/1B checkpoints never had reconstruction heads) and is always initialized fresh here.
Its state is saved alongside the encoder's under a separate checkpoint key
(recon_heads_state_dict) so checkpoint-at-peak can restore the exact (encoder, heads) pair
that produced the best holdout score -- Stage 1D, if it warm-starts from 1C, only needs
the encoder's own model_state_dict, same convention this file uses to warm-start from 1B.

Run from project root:
    python -m src.conviction_model.run_stage1c
    python -m src.conviction_model.run_stage1c --warm-start-checkpoint artifacts/checkpoints/conviction_model/stage1b-20260722-213728.pt
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
from .encoder import BRANCHES, EncoderCNN
from .run_stage1a import (
    BYTES_PER_CACHED_POSITION, CHECKPOINT_DIR, DEFAULT_PANEL_CACHE_SIZE, DEFAULT_RESERVED_HOLDOUT_YEARS,
    LOG_DIR, _holdout_eligible_count, _save_checkpoint, load_panel, setup_logging, truncate_to_development_window,
)
from .ssl_pretrain import (
    CachedPanelGatherer, ReconstructionHeads, build_stage1b_batch, sample_cpc_anchor_positions,
    score_holdout_stage1c, split_train_holdout, train_step_stage1c,
)

DEFAULT_STEPS = 5000  # matches run_stage1a.py/run_stage1b.py's default budget -- same first-guess


def _latest_stage1b_checkpoint() -> Path:
    checkpoints = sorted(CHECKPOINT_DIR.glob("stage1b-*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"no stage1b-*.pt checkpoints found in {CHECKPOINT_DIR} -- "
            "Stage 1C warm-starts from a Stage 1B checkpoint, run run_stage1b.py first")
    return checkpoints[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-start-checkpoint", type=str, default=None,
                         help="defaults to the most recent stage1b-*.pt checkpoint")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=500,
                         help="save a checkpoint every N steps, in addition to the final save")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                         help="defaults to artifacts/checkpoints/conviction_model/stage1c-<run_id>.pt")
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
    log = setup_logging(run_id, stage="stage1c")
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else CHECKPOINT_DIR / f"stage1c-{run_id}.pt"
    warm_start_path = Path(args.warm_start_checkpoint) if args.warm_start_checkpoint else _latest_stage1b_checkpoint()

    log.info(f"Warm-start checkpoint: {warm_start_path}")
    ckpt = torch.load(warm_start_path, map_location="cpu")
    # Reuse Stage 1B's exact hyperparameters, same rationale as run_stage1b.py's own
    # warm-start -- reconstruction_weight isn't in the old config dict (1B predates this
    # field), so it falls back to today's dataclass default.
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

    n_stores = 2 if holdout_enabled else 1  # train_store + holdout_store are each capped
                                             # INDEPENDENTLY -- report the COMBINED estimate
    est_gb = n_stores * args.panel_cache_size * BYTES_PER_CACHED_POSITION / 1e9
    log.info(f"Panel cache: up to {args.panel_cache_size} positions/store x {n_stores} store(s) "
             f"(~{est_gb:.1f}GB estimated COMBINED peak)")
    train_store = CachedPanelGatherer(train_panel, frame_cache, maxsize=args.panel_cache_size)
    holdout_store = CachedPanelGatherer(holdout_panel, frame_cache, maxsize=args.panel_cache_size) \
        if holdout_enabled else None

    model = EncoderCNN(len(DAILY_FEATURES), len(WEEKLY_FEATURES), len(MONTHLY_FEATURES),
                        len(QUARTERLY_FEATURES), d_model=cfg.d_model, n_heads=cfg.n_heads)
    model.load_state_dict(ckpt["model_state_dict"])
    recon_heads = ReconstructionHeads(cfg.d_model)  # no warm-start source -- always fresh
    optimizer = torch.optim.Adam(list(model.parameters()) + list(recon_heads.parameters()), lr=cfg.learning_rate)

    losses = []
    best_step, best_score, best_state, best_recon_state = None, None, None, None
    t_start = time.monotonic()
    for step in range(args.steps):
        anchors = sample_cpc_anchor_positions(train_panel, cfg.batch_size, max_horizon, rng=np_rng)
        anchor_batch, cpc_positive_batch, align_positive_batch, negative_batch = build_stage1b_batch(
            train_panel, train_store, anchors, cfg.cpc_horizon, cfg.alignment_horizon,
            n_same_stock=cfg.n_same_stock_negatives, n_diff_stock=cfg.n_diff_stock_negatives,
            regime_gap_days=cfg.regime_gap_days, rng=np_rng)
        masked_branch = BRANCHES[int(np_rng.integers(len(BRANCHES)))]
        loss = train_step_stage1c(model, recon_heads, optimizer, anchor_batch, cpc_positive_batch,
                                   align_positive_batch, negative_batch, masked_branch,
                                   temperature=cfg.temperature, alignment_weight=cfg.alignment_weight,
                                   reconstruction_weight=cfg.reconstruction_weight)
        losses.append(loss)
        if step % args.log_every == 0 or step == args.steps - 1:
            recent = losses[-args.log_every:]
            mean_total = sum(l["total"] for l in recent) / len(recent)
            mean_cpc = sum(l["cpc"] for l in recent) / len(recent)
            mean_align = sum(l["alignment"] for l in recent) / len(recent)
            mean_recon = sum(l["reconstruction"] for l in recent) / len(recent)
            log.info(f"step {step:>6}/{args.steps}  total={loss['total']:.4f}  "
                     f"mean_recent(total={mean_total:.4f}, cpc={mean_cpc:.4f}, alignment={mean_align:.4f}, "
                     f"reconstruction={mean_recon:.4f})  elapsed={time.monotonic() - t_start:.0f}s")

        if holdout_enabled and ((step + 1) % cfg.checkpoint_eval_every == 0 or step == args.steps - 1):
            score = score_holdout_stage1c(model, recon_heads, holdout_panel, holdout_store, cfg, np_rng)
            log.info(f"  holdout eval at step {step}: loss={score:.4f}"
                     + (f" (new best, was {best_score:.4f})" if best_score is not None and score < best_score
                        else " (new best)" if best_score is None else ""))
            if best_score is None or score < best_score:
                best_step, best_score = step, score
                best_state = copy.deepcopy(model.state_dict())
                best_recon_state = copy.deepcopy(recon_heads.state_dict())

        if step > 0 and (step % args.checkpoint_every == 0 or step == args.steps - 1):
            _save_checkpoint(checkpoint_path, model.state_dict(), cfg, tickers, losses, step, args.steps,
                              best_step, best_score, extra={"recon_heads_state_dict": recon_heads.state_dict()})
            log.info(f"checkpoint saved: {checkpoint_path} (step {step}, latest weights)")

    if not losses:
        return
    final_state = best_state if best_state is not None else model.state_dict()
    final_recon_state = best_recon_state if best_recon_state is not None else recon_heads.state_dict()
    _save_checkpoint(checkpoint_path, final_state, cfg, tickers, losses, args.steps - 1, args.steps,
                      best_step, best_score, extra={"recon_heads_state_dict": final_recon_state})
    log.info(f"Loss (total): first={losses[0]['total']:.4f}  last={losses[-1]['total']:.4f}  "
             f"min={min(l['total'] for l in losses):.4f}")
    if best_state is not None:
        log.info(f"Final checkpoint holds the BEST-scoring weights: step {best_step}, "
                 f"holdout loss={best_score:.4f} (not the last step's weights)")
    log.info(f"Final checkpoint: {checkpoint_path}")
    log.info(f"Log file: {LOG_DIR / f'stage1c-{run_id}.log'}")


if __name__ == "__main__":
    main()
