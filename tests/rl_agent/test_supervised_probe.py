"""
test_supervised_probe.py — tests for M3's supervised ranking probe.
"""

import numpy as np
import torch
import pytest

from src.rl_agent.supervised_probe import (
    SupervisedRankingProbe, listwise_ranking_loss, compute_daily_ic
)


@pytest.fixture
def probe():
    """Create a small supervised probe for testing."""
    return SupervisedRankingProbe(window=50, conv1_out_channels=2,
                                  conv2_out_channels=4, n_features=3)


@pytest.fixture
def synthetic_batch():
    """Create a synthetic price batch."""
    batch_size = 4
    n_assets = 10
    lookback = 50
    n_features = 3

    # [B, f, m, n]: random prices
    X = torch.randn(batch_size, n_features, n_assets, lookback)
    # [B, m]: all assets real (no padding)
    mask = torch.ones(batch_size, n_assets, dtype=torch.bool)
    # [B, m]: synthetic forward returns
    returns = torch.randn(batch_size, n_assets) * 0.1
    return X, mask, returns


def test_forward_shape(probe, synthetic_batch):
    """Output shape should be [B, m]."""
    X, mask, _ = synthetic_batch
    scores = probe(X, mask)
    assert scores.shape == (X.shape[0], X.shape[2])


def test_masking_inactive_slots(probe):
    """Inactive slots should have -inf logits."""
    batch_size = 2
    n_assets = 5
    lookback = 50
    n_features = 3

    X = torch.randn(batch_size, n_features, n_assets, lookback)
    mask = torch.tensor([[True, True, False, False, True],
                         [True, False, True, False, False]], dtype=torch.bool)

    scores = probe(X, mask)

    # Inactive slots should be -inf
    assert torch.isinf(scores[0, 2]) and scores[0, 2] < 0
    assert torch.isinf(scores[0, 3]) and scores[0, 3] < 0
    assert torch.isinf(scores[1, 1]) and scores[1, 1] < 0

    # Active slots should be finite
    assert torch.isfinite(scores[0, 0])
    assert torch.isfinite(scores[0, 1])


def test_loss_finite(probe, synthetic_batch):
    """Loss should be finite and non-zero."""
    X, mask, returns = synthetic_batch
    scores = probe(X, mask)
    loss = listwise_ranking_loss(scores, returns, mask)
    assert torch.isfinite(loss)
    assert loss > 0


def test_loss_with_padding(probe):
    """Loss should handle some assets being inactive."""
    batch_size = 2
    n_assets = 10
    lookback = 50
    n_features = 3

    X = torch.randn(batch_size, n_features, n_assets, lookback)
    mask = torch.tensor([[True] * 5 + [False] * 5,
                         [True] * 7 + [False] * 3], dtype=torch.bool)
    returns = torch.randn(batch_size, n_assets) * 0.1

    scores = probe(X, mask)
    loss = listwise_ranking_loss(scores, returns, mask)
    assert torch.isfinite(loss)


def test_ic_computation(probe, synthetic_batch):
    """IC should be a float between -1 and 1."""
    X, mask, returns = synthetic_batch
    scores = probe(X, mask)
    ic = compute_daily_ic(scores, returns, mask)
    assert isinstance(ic, float)
    assert -1.0 <= ic <= 1.0


def test_ic_perfect_prediction():
    """IC should be 1 when scores perfectly rank returns."""
    batch_size = 2
    n_assets = 5

    # Create perfect correlation: scores = returns (in expected order)
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                          [5.0, 4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
    returns = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                           [5.0, 4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
    mask = torch.ones(batch_size, n_assets, dtype=torch.bool)

    ic = compute_daily_ic(scores, returns, mask)
    # Should be near 1.0 (perfect Spearman)
    assert ic > 0.9


def test_ic_opposite_prediction():
    """IC should be near -1 when scores perfectly reverse returns."""
    batch_size = 2
    n_assets = 5

    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                          [5.0, 4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
    returns = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0],
                           [1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)
    mask = torch.ones(batch_size, n_assets, dtype=torch.bool)

    ic = compute_daily_ic(scores, returns, mask)
    # Should be near -1.0 (perfect negative Spearman)
    assert ic < -0.9


def test_ic_random_is_near_zero():
    """IC should be near 0 for random scores vs. random returns."""
    batch_size = 8
    n_assets = 20
    np.random.seed(0)

    scores = torch.randn(batch_size, n_assets)
    returns = torch.randn(batch_size, n_assets)
    mask = torch.ones(batch_size, n_assets, dtype=torch.bool)

    ic = compute_daily_ic(scores, returns, mask)
    # Should be close to zero (random correlation)
    assert abs(ic) < 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
