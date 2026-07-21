"""
Test: conviction_model/encoder.py's EncoderCNN -- forward-pass shapes,
finite gradients, CPU determinism, and that cross-attention actually updates
each branch token (not a no-op). Synthetic tensors only, no dependency on
data/raw or data/processed.

Run from project root:
    python tests/conviction_model/test_encoder.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.conviction_model.encoder import BRANCHES, EncoderCNN  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402

B = 4
N_DAILY, N_WEEKLY, N_MONTHLY, N_FUND = 11, 11, 14, 19
W_DAILY, W_WEEKLY, W_MONTHLY, W_FUND = 60, 104, 120, 40


def _synthetic_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    daily = torch.randn(B, N_DAILY, W_DAILY, generator=g)
    weekly = torch.randn(B, N_WEEKLY, W_WEEKLY, generator=g)
    monthly = torch.randn(B, N_MONTHLY, W_MONTHLY, generator=g)
    fundamentals = torch.randn(B, N_FUND, W_FUND, generator=g)
    return daily, weekly, monthly, fundamentals


def _build_model(seed=0):
    torch.manual_seed(seed)
    return EncoderCNN(N_DAILY, N_WEEKLY, N_MONTHLY, N_FUND, d_model=16, n_heads=4)


def test_forward_shapes_four_separate_embeddings(passed, failed):
    model = _build_model()
    out = model(*_synthetic_inputs())
    ok = (set(out.keys()) == set(BRANCHES)
          and all(out[name].shape == (B, model.d_model) for name in BRANCHES))
    print_check("EncoderCNN.forward: returns 4 separately-keyed [B, d_model] sub-embeddings",
                ok, f"keys={list(out.keys())}, shapes={[tuple(v.shape) for v in out.values()]}")
    return passed + ok, failed + (not ok)


def test_gradients_are_finite(passed, failed):
    model = _build_model()
    out = model(*_synthetic_inputs())
    loss = sum(v.sum() for v in out.values())
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    ok = all(g is not None and torch.isfinite(g).all() for g in grads)
    print_check("EncoderCNN: every parameter has a finite gradient after backward()",
                ok, f"n_params_checked={len(grads)}")
    return passed + ok, failed + (not ok)


def test_deterministic_on_cpu(passed, failed):
    model = _build_model()
    model.eval()
    inputs = _synthetic_inputs()
    with torch.no_grad():
        out1 = model(*inputs)
        out2 = model(*inputs)
    ok = all(torch.equal(out1[name], out2[name]) for name in BRANCHES)
    print_check("EncoderCNN: two forward passes on the same input/weights match exactly (CPU)", ok)
    return passed + ok, failed + (not ok)


def test_cross_attention_actually_updates_tokens(passed, failed):
    model = _build_model()
    model.eval()
    inputs = _synthetic_inputs()
    with torch.no_grad():
        pre = model.branch_tokens(*inputs)          # [B, 4, d_model], before attention
        post_dict = model(*inputs)
        post = torch.stack([post_dict[name] for name in BRANCHES], dim=1)
    ok = not torch.allclose(pre, post, atol=1e-6)
    print_check("EncoderCNN: cross-attention changes branch tokens relative to pre-attention values "
                "(not an identity fallback)", ok)
    return passed + ok, failed + (not ok)


def main() -> int:
    print_header("conviction_model/encoder.py")
    passed = failed = 0
    for test_fn in [
        test_forward_shapes_four_separate_embeddings,
        test_gradients_are_finite,
        test_deterministic_on_cpu,
        test_cross_attention_actually_updates_tokens,
    ]:
        passed, failed = test_fn(passed, failed)
    print_section_end(passed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
