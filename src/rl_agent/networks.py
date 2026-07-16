"""
networks.py — the EIIE CNN encoder (paper Fig. 2), docs/EIIE_AGENT_PLAN.md
"EIIE network" section.

Any future encoder (RNN/LSTM, the paper's other two instantiations, or a
feature-branch encoder for fundamentals/macro) just needs to match this
forward signature to plug into train.py unchanged:

    forward(X: [B, f, m, n], w_prev: [B, m], mask: [B, m]) -> w: [B, m+1]

X: the price tensor (eq. 18); w_prev: previous period's weight on each of
TODAY's m active slots (PVM.read_slots' output with the cash column
stripped); mask: which of the m slots are real vs. padding; w: softmaxed
portfolio weights, column 0 = cash.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EIIECNN(nn.Module):
    """Fig. 2's fully convolutional EIIE: a chain of kernel-height-1
    convolutions (isolating each asset's row -- the "Identical Independent
    Evaluators") ending in a 1x1 conv down to one score per asset, with the
    previous portfolio vector inserted as an extra feature map just before
    that scoring layer so the network can weigh transaction cost."""

    def __init__(self, window: int, conv1_out_channels: int = 2,
                 conv2_out_channels: int = 20, n_features: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_features, conv1_out_channels, kernel_size=(1, 3))
        conv2_width = window - 2  # collapses the remaining time dimension entirely
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=(1, conv2_width))
        self.conv3 = nn.Conv2d(conv2_out_channels + 1, 1, kernel_size=(1, 1))  # +1: the w_prev feature map
        self.cash_bias = nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor, w_prev: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(X))                          # [B, c1, m, n-2]
        h = F.relu(self.conv2(h))                           # [B, c2, m, 1]
        w_prev_map = w_prev.unsqueeze(1).unsqueeze(-1)        # [B, 1, m, 1]
        h = torch.cat([h, w_prev_map], dim=1)                   # [B, c2+1, m, 1]
        scores = self.conv3(h).squeeze(1).squeeze(-1)             # [B, m]
        scores = scores.masked_fill(~mask, float("-inf"))            # padding slots never win the softmax
        cash_score = self.cash_bias.expand(scores.shape[0], 1)
        logits = torch.cat([cash_score, scores], dim=1)                # [B, m+1]
        return F.softmax(logits, dim=1)
