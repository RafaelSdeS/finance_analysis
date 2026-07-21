"""
encoder.py -- Phase 1 (docs/conviction_model/CONVICTION_MODEL_PLAN.md): the
multi-resolution encoder. 4 branch sub-networks (dilated Conv1d along the
time axis, one branch per resolution) each pool to a d_model-wide token; a
cross-attention layer then lets the 4 tokens attend to each other and
returns 4 UPDATED, still-separate sub-embeddings -- not one pooled vector
(Architecture section) -- so a downstream tree model can report per-branch
feature importance and a branch can be ablated without retraining fusion.

House style follows src/rl_agent/networks.py::EIIECNN: plain nn.Module,
functional F.relu calls, constructor args threaded straight from data.py's
per-branch feature counts, no Sequential/factory/config-class indirection.
Uses nn.Conv1d rather than EIIECNN's kernel-height-1 Conv2d -- EIIECNN's
trick keeps N assets independent inside one batched multi-asset conv; this
encoder scores one ticker at a time (no asset axis inside a branch's
tensor), so a plain 1D temporal conv is the direct analog, not a copy of
EIIECNN's shape.
"""

import torch
import torch.nn.functional as F
from torch import nn

BRANCHES = ("daily", "weekly", "monthly", "fundamentals")


class _BranchCNN(nn.Module):
    """Dilated Conv1d stack over one branch's [B, n_features, window] window
    tensor -> a single pooled d_model token.
    ponytail: adaptive avg pool instead of a width-matched final conv
    (EIIECNN's approach) -- one pooling op works for any branch window length
    without a per-branch kernel-width calc; revisit if a branch needs
    positional structure the pool destroys."""

    def __init__(self, n_features: int, d_model: int = 64, hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1)
        return self.proj(h)


class EncoderCNN(nn.Module):
    """forward(daily, weekly, monthly, fundamentals) -> dict of 4
    [B, d_model] sub-embeddings, one per branch (BRANCHES order), after a
    cross-attention update. Each arg is that branch's
    [B, n_features_branch, window_branch] tensor (data.py's window_tensor(),
    batched)."""

    def __init__(self, n_features_daily: int, n_features_weekly: int,
                 n_features_monthly: int, n_features_fundamentals: int,
                 d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.branch_daily = _BranchCNN(n_features_daily, d_model)
        self.branch_weekly = _BranchCNN(n_features_weekly, d_model)
        self.branch_monthly = _BranchCNN(n_features_monthly, d_model)
        self.branch_fundamentals = _BranchCNN(n_features_fundamentals, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def branch_tokens(self, daily: torch.Tensor, weekly: torch.Tensor,
                       monthly: torch.Tensor, fundamentals: torch.Tensor) -> torch.Tensor:
        """Pre-attention tokens, [B, 4, d_model] in BRANCHES order -- exposed
        separately so tests/diagnostics can compare against the post-attention
        result without duplicating the branch-CNN calls."""
        return torch.stack([
            self.branch_daily(daily), self.branch_weekly(weekly),
            self.branch_monthly(monthly), self.branch_fundamentals(fundamentals),
        ], dim=1)

    def forward(self, daily: torch.Tensor, weekly: torch.Tensor,
                monthly: torch.Tensor, fundamentals: torch.Tensor) -> dict:
        tokens = self.branch_tokens(daily, weekly, monthly, fundamentals)
        updated, _ = self.cross_attn(tokens, tokens, tokens)
        return {name: updated[:, i, :] for i, name in enumerate(BRANCHES)}
