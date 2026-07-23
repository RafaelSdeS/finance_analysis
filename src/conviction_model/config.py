"""
config.py -- Phase 1 (docs/conviction_model/CONVICTION_MODEL_PLAN.md): frozen-dataclass
pretraining config, JSON round-trippable, mirrors rl_agent/config.py's convention.

Stage 1A (CPC) + Stage 1B (forward cross-modal alignment) + Stage 1C (masked
reconstruction) fields exist. The valuation-probe field (Stage 1D) gets added when that
loss is written, not speculatively now -- see Module layout's ssl_pretrain.py row.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SSLConfig:
    cpc_horizon: int = 21          # trading days ahead the CPC positive is drawn from -- matches the primary short label horizon (Labels)
    n_same_stock_negatives: int = 4
    n_diff_stock_negatives: int = 4
    regime_gap_days: int = 252
    temperature: float = 0.1
    batch_size: int = 64
    learning_rate: float = 1e-3
    d_model: int = 64
    n_heads: int = 4
    seed: int = 0
    # Checkpoint-at-peak (mirrors rl_agent/config.py's checkpoint_holdout_days/
    # checkpoint_eval_every -- CLAUDE.md: caught a real case where the same seed went
    # from +47% return at 100k steps to -71% at 2M, pure overfitting past a peak that a
    # fixed step count alone can't detect). ~1 calendar year held out of CPC training
    # entirely; scored every 250 steps here vs. rl_agent's 5000 since this run's default
    # step budget (5000) is far smaller -- both arbitrary, adjustable.
    checkpoint_holdout_days: int = 365
    checkpoint_eval_every: int = 250
    # Stage 1B: forward cross-modal alignment, combined with CPC as a weighted sum
    # (Module layout: "weighted sum" -- loss = cpc + alignment_weight * alignment).
    # 1.0 = equal footing with CPC to start; arbitrary, adjustable like every other
    # loss weight in this plan (e.g. Phase 2's per-horizon regressor weights).
    alignment_weight: float = 1.0
    # Deliberately NOT cpc_horizon (21 trading days, ~1 month). The fundamentals
    # branch only updates on a filing cadence roughly 3x longer (~63 trading days/
    # quarter, CLAUDE.md's documented ~45-90 day filing lag) -- at cpc_horizon, most
    # (t, t+k) pairs span no real fundamentals transition at all, so the alignment
    # loss was mostly testing "stay consistent with the still-current quarter," not
    # real forward prediction (first real Stage 1B run, 2026-07-21: alignment train
    # loss collapsed 0.30->0.05 in 50 steps while holdout never improved past step
    # ~1500 -- a shortcut, not learning). 63 trading days = one fiscal quarter, so a
    # (t, t+alignment_horizon) pair is far more likely to actually straddle a filing.
    alignment_horizon: int = 63
    # Stage 1C: masked reconstruction across branches, combined with CPC + alignment as a
    # weighted sum (Module layout: "weighted sum" -- loss = cpc + alignment_weight*alignment
    # + reconstruction_weight*reconstruction). 1.0 = equal footing with the other two losses
    # to start, matching alignment_weight's own default; arbitrary, adjustable. The plan
    # calls this the "weakest of the three" losses for the encoder's actual goal ("What the
    # latent representation is for") -- kept as a regularizer, not expected to drive
    # diagnostic gains on its own.
    reconstruction_weight: float = 1.0

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path) -> "SSLConfig":
        return cls(**json.loads(Path(path).read_text()))
