"""
test_supervised_experiment.py — tests for M3 supervised experiment.

Key invariants:
- Label window ends at t, forward returns are [t+1, t+k] — zero lookahead
- Masks propagate correctly through batching
- Train/val split is respected
"""

import numpy as np
import torch
import pytest

from src.rl_agent.supervised_experiment import (
    compute_forward_returns,
    create_train_val_loaders,
)


def test_forward_returns_shape():
    """Forward returns should be [T, n_global], with NaNs for out-of-window."""
    T, n_global = 100, 50
    k = 5

    # Mock close prices: [T, n_global]
    prices = np.ones((T, n_global)) * np.arange(1, n_global + 1)  # constant per asset

    # Manually create a mock panel-like object
    class MockPanel:
        def __init__(self, prices):
            self.prices = prices

    panel = MockPanel(prices)
    fwd_ret = compute_forward_returns(panel, k)

    assert fwd_ret.shape == (T, n_global)
    # First T-k rows should have finite returns (0 for constant prices)
    assert np.allclose(fwd_ret[: T - k], 0.0)
    # Last k rows should be NaN (no k-day window available)
    assert np.all(np.isnan(fwd_ret[T - k :]))


def test_forward_returns_zero_for_constant_prices():
    """Forward log-return should be 0 for constant prices."""
    T, n_global = 50, 10
    k = 3
    prices = np.ones((T, n_global))  # All prices = 1

    class MockPanel:
        def __init__(self, prices):
            self.prices = prices

    panel = MockPanel(prices)
    fwd_ret = compute_forward_returns(panel, k)

    # All finite rows should be ~0
    finite_rows = ~np.isnan(fwd_ret)
    assert np.allclose(fwd_ret[finite_rows], 0.0, atol=1e-10)


def test_forward_returns_positive_growth():
    """Forward log-return should be positive for price growth."""
    T, n_global = 50, 10
    k = 5
    prices = np.ones((T, n_global))
    # Asset 0 grows 2× over k days
    prices[: T - k, 0] = 1.0
    prices[k : T, 0] = 2.0

    class MockPanel:
        def __init__(self, prices):
            self.prices = prices

    panel = MockPanel(prices)
    fwd_ret = compute_forward_returns(panel, k)

    # First T-k rows, asset 0 should have log(2) ≈ 0.693 return
    assert np.allclose(fwd_ret[: T - k, 0], np.log(2.0), atol=1e-6)


def test_forward_returns_nan_propagation():
    """NaN in prices should propagate to forward returns."""
    T, n_global = 50, 10
    k = 3
    prices = np.ones((T, n_global))
    # Inject NaN at t=10 for asset 0
    prices[10, 0] = np.nan

    class MockPanel:
        def __init__(self, prices):
            self.prices = prices

    panel = MockPanel(prices)
    fwd_ret = compute_forward_returns(panel, k)

    # Any row that includes t=10 in its forward window should have NaN for asset 0
    assert np.isnan(fwd_ret[10, 0])  # t=10 is the start
    # Also t=9,8,7 have forward windows that include t=10
    for t in range(10 - k, 11):
        if t >= 0 and t < T - k:
            assert np.isnan(fwd_ret[t, 0])


def test_train_val_split_no_overlap():
    """Train and val loaders should not overlap."""
    T, n_global, m = 100, 50, 50
    k_list = [1, 5, 21]

    # Mock panel
    class MockPanel:
        def __init__(self, prices):
            self.prices = prices
            self.valid = np.ones((prices.shape[0], m), dtype=bool)

        def window_tensor(self, t):
            # Return dummy tensor [f, m, window]
            return torch.randn(11, m, 50)

    prices = np.random.randn(T, n_global) * 0.01 + 1.0
    panel = MockPanel(prices)

    fwd_returns = {k: np.random.randn(T, n_global) * 0.1 for k in k_list}
    train_end = 70
    val_end = 90

    loaders = list(create_train_val_loaders(panel, fwd_returns, train_end, val_end))
    train_loader = dict(loaders)["train"]
    val_loader = dict(loaders)["val"]

    train_indices = set()
    for X, mask, returns_t, t in train_loader:
        train_indices.add(t.item() if torch.is_tensor(t) else t)

    val_indices = set()
    for X, mask, returns_t, t in val_loader:
        val_indices.add(t.item() if torch.is_tensor(t) else t)

    # No overlap
    assert len(train_indices & val_indices) == 0
    # Train should be [0, train_end]
    assert all(idx <= train_end for idx in train_indices)
    # Val should be [train_end+1, val_end]
    assert all(idx > train_end for idx in val_indices)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
