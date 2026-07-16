"""
Test: networks.py's EIIECNN -- output shape, simplex constraint, and slot
masking (docs/EIIE_AGENT_PLAN.md Phase 6). Synthetic data only.

Run from project root:
    python tests/rl_agent/test_networks.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.rl_agent.networks import EIIECNN  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402


def test_forward_shape_and_simplex(passed, failed):
    torch.manual_seed(0)
    model = EIIECNN(window=10, n_features=3)
    B, m = 4, 6
    X = torch.randn(B, 3, m, 10)
    w_prev = torch.rand(B, m)
    mask = torch.ones(B, m, dtype=torch.bool)

    w = model(X, w_prev, mask)
    ok = w.shape == (B, m + 1)
    print_check("EIIECNN: output shape is [B, m+1]", ok, str(w.shape))
    passed, failed = passed + ok, failed + (not ok)

    ok = torch.allclose(w.sum(dim=1), torch.ones(B), atol=1e-5)
    print_check("EIIECNN: output sums to 1 per batch row (simplex)", ok, str(w.sum(dim=1).tolist()))
    passed, failed = passed + ok, failed + (not ok)

    ok = bool((w >= 0).all())
    print_check("EIIECNN: output is non-negative", ok)
    passed, failed = passed + ok, failed + (not ok)

    ok = bool(torch.isfinite(w).all())
    print_check("EIIECNN: output is finite (no NaN/Inf)", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_masking(passed, failed):
    torch.manual_seed(1)
    model = EIIECNN(window=10, n_features=3)
    B, m = 3, 5
    X = torch.randn(B, 3, m, 10)
    w_prev = torch.rand(B, m)
    mask = torch.ones(B, m, dtype=torch.bool)
    mask[:, -2:] = False  # last 2 slots are padding

    w = model(X, w_prev, mask)
    masked_weights = w[:, 1:][:, -2:]  # cash is column 0, so slot i is column i+1
    ok = torch.allclose(masked_weights, torch.zeros_like(masked_weights), atol=1e-6)
    print_check("EIIECNN: masked/padding slots get exactly 0 weight", ok, str(masked_weights.tolist()))
    passed, failed = passed + ok, failed + (not ok)

    ok = torch.allclose(w.sum(dim=1), torch.ones(B), atol=1e-5)
    print_check("EIIECNN: masking still leaves a valid simplex (mass on cash + real slots)", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_gradient_flows(passed, failed):
    torch.manual_seed(2)
    model = EIIECNN(window=8, n_features=3)
    X = torch.randn(2, 3, 4, 8)
    w_prev = torch.rand(2, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)

    w = model(X, w_prev, mask)
    w.sum().backward()
    grads_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    print_check("EIIECNN: gradients are finite after backward()", grads_finite)
    passed, failed = passed + grads_finite, failed + (not grads_finite)

    has_grad = all(p.grad is not None for p in model.parameters())
    print_check("EIIECNN: every parameter (incl. cash_bias) receives a gradient", has_grad)
    passed, failed = passed + has_grad, failed + (not has_grad)
    return passed, failed


def main():
    print_header("test_networks")
    passed = failed = 0

    passed, failed = test_forward_shape_and_simplex(passed, failed)
    passed, failed = test_masking(passed, failed)
    passed, failed = test_gradient_flows(passed, failed)

    print_section_end(passed, failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
