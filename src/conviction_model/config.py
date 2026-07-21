"""
config.py -- Phase 1 (docs/conviction_model/CONVICTION_MODEL_PLAN.md): frozen-dataclass
pretraining config, JSON round-trippable, mirrors rl_agent/config.py's convention.

Only Stage 1A (CPC) fields exist yet. Masked-reconstruction / forward-cross-modal-
alignment / valuation-probe fields (Stages 1B-1D) get added when those losses are
written, not speculatively now -- see Module layout's ssl_pretrain.py row.
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

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path) -> "SSLConfig":
        return cls(**json.loads(Path(path).read_text()))
