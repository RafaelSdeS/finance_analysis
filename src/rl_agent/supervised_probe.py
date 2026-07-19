"""
supervised_probe.py — M3 experiment: supervised daily ranking probe.

Tests "is there extractable cross-sectional signal in price+technical features?"
without RL noise or concentration attractor.

Architecture: same conv trunk as EIIE, per-asset scores (no softmax/portfolio mechanics).
Loss: listwise cross-entropy over active slots' forward k-day returns.
Metric: daily IC (active-only Spearman) on train & val vs. permutation null.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedRankingProbe(nn.Module):
    """Per-asset scoring network: conv trunk → per-asset logits (no softmax).

    Labels are each asset's forward k-day log-return; loss is listwise
    cross-entropy (classify which assets had highest realized returns).
    """

    def __init__(self, window: int, conv1_out_channels: int = 2,
                 conv2_out_channels: int = 20, n_features: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_features, conv1_out_channels, kernel_size=(1, 3))
        conv2_width = window - 2
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels,
                              kernel_size=(1, conv2_width))
        # ponytail: no w_prev here, and no cash column — just per-asset scores
        self.conv3 = nn.Conv2d(conv2_out_channels, 1, kernel_size=(1, 1))

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: [B, f, m, n] price tensor (m = n_assets, n = lookback window)
            mask: [B, m] bool mask (True = real asset, False = padding)

        Returns:
            scores: [B, m] per-asset logits (not softmaxed — will be handled by
                    loss function which applies masked softmax over the active set)
        """
        h = F.relu(self.conv1(X))                    # [B, c1, m, n-2]
        h = F.relu(self.conv2(h))                    # [B, c2, m, 1]
        scores = self.conv3(h).squeeze(1).squeeze(-1)  # [B, m]
        # Mask inactive slots (they won't participate in the loss)
        scores = scores.masked_fill(~mask, float("-inf"))
        return scores


def listwise_ranking_loss(scores: torch.Tensor, labels: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    """Listwise cross-entropy loss over active assets' ranking.

    For each day (batch element), labels are the forward k-day log-returns
    of each asset. Loss treats this as a classification problem: "which
    assets had the highest realized returns?" The ranking (softmax over
    log-returns) is the ground truth, and the network's scores should
    predict that ranking.

    Args:
        scores: [B, m] network predictions
        labels: [B, m] forward k-day log-returns
        mask: [B, m] bool, True = active/holdable asset

    Returns:
        loss: scalar, averaged over batch
    """
    # Cross-entropy: predict the ranking of returns from scores
    # ponytail: could use other ranking losses (ListNet, LambdaRank), but
    # cross-entropy is the simplest and most direct
    batch_size = scores.shape[0]
    loss = 0.0

    for i in range(batch_size):
        active_mask = mask[i]
        active_scores = scores[i, active_mask]      # [n_active]
        active_labels = labels[i, active_mask]      # [n_active]

        if active_scores.numel() == 0:
            # No active assets this day (shouldn't happen)
            continue

        # Softmax over scores; labels should also be softmaxed (both represent
        # a ranking distribution, loss is KL)
        score_probs = F.softmax(active_scores, dim=0)
        label_probs = F.softmax(active_labels, dim=0)

        # KL divergence: D(label_probs || score_probs)
        # equivalent to cross-entropy with label_probs as the target distribution
        loss += F.kl_div(
            F.log_softmax(active_scores, dim=0),
            label_probs,
            reduction='batchmean'
        )

    return loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=scores.device)


def compute_daily_ic(scores: torch.Tensor, returns: torch.Tensor,
                     mask: torch.Tensor) -> float:
    """Compute daily IC (Spearman correlation) between scores and realized returns.

    Args:
        scores: [B, m] network scores (or raw predictions)
        returns: [B, m] realized forward returns
        mask: [B, m] bool, True = active/holdable

    Returns:
        mean_ic: float, mean daily Spearman over the batch
    """
    from scipy.stats import spearmanr

    ics = []
    for i in range(scores.shape[0]):
        active_mask = mask[i]
        if not active_mask.any():
            continue

        active_scores = scores[i, active_mask].detach().cpu().numpy()
        active_returns = returns[i, active_mask].detach().cpu().numpy()

        # Spearman requires at least 2 points and no all-NaN
        if len(active_scores) >= 2 and not (np.isnan(active_returns).all()):
            rho, _ = spearmanr(active_scores, active_returns)
            if not np.isnan(rho):
                ics.append(rho)

    return float(np.mean(ics)) if ics else 0.0


# imports for IC computation
import numpy as np
