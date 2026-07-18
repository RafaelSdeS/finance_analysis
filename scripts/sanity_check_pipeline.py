"""Pre-flight sanity check (EIIE_IMPROVEMENT_PLAN.md, Stage 0 pre-flight).

One real-data batch through the current 11-channel baseline network:
forward pass, backward pass, and assertions that (a) inputs contain no
NaN/inf, (b) conv1/conv2 activations are finite, (c) the output weight
vector is finite and non-degenerate (valid simplex, not uniform-collapsed,
not one-hot at init), (d) gradients flow back to conv1. Read-only: no
training step, no artifacts written.

Run from project root:  python scripts/sanity_check_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.rl_agent.config import ExperimentConfig
from src.rl_agent.data import load_price_panel
from src.rl_agent.networks import EIIECNN

CONFIG = "configs/eiie_features_frozen.json"  # the 11-channel baseline


def main():
    cfg = ExperimentConfig.from_json(CONFIG)
    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    print(f"config={CONFIG} features={len(cfg.data.features)} device={device}")

    print("loading real price panel...")
    panel = load_price_panel(cfg.data, n_slots=cfg.model.n_assets)
    B = cfg.train.batch_size
    t_idx = np.arange(panel.start_idx + 100, panel.start_idx + 100 + B)
    print(f"batch: {B} consecutive days starting {panel.dates[t_idx[0]].date()}")

    # --- inputs ---
    X_np = panel.window_tensor_batch(t_idx, cfg.data.features)
    assert np.isfinite(X_np).all(), "NaN/inf in input tensor X"
    y_next = panel.price_relative_batch(t_idx + 1)  # earned by w_t (eq. 22 pairing)
    assert np.isfinite(y_next).all() and (y_next > 0).all(), "bad price relatives"
    print(f"[OK] inputs finite: X {X_np.shape}, y_next {y_next.shape}")

    # --- forward ---
    model = EIIECNN(cfg.data.window, cfg.model.conv1_out_channels,
                    cfg.model.conv2_out_channels, len(cfg.data.features)).to(device)
    acts = {}
    model.conv1.register_forward_hook(lambda m, i, o: acts.__setitem__("conv1", o.detach()))
    model.conv2.register_forward_hook(lambda m, i, o: acts.__setitem__("conv2", o.detach()))

    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    mask = torch.tensor(panel.valid[t_idx], dtype=torch.bool, device=device)
    w_prev = torch.full((B, cfg.model.n_assets), 1.0 / (cfg.model.n_assets + 1), device=device)
    w = model(X, w_prev, mask)

    for name, a in acts.items():
        assert torch.isfinite(a).all(), f"non-finite activations in {name}"
        print(f"[OK] {name} activations finite, shape {tuple(a.shape)}")
    # raw pre-softmax logits carry -inf on masked slots BY DESIGN, so degeneracy
    # is asserted on the softmax output instead
    assert torch.isfinite(w).all(), "non-finite output weights"
    assert torch.allclose(w.sum(dim=1), torch.ones(B, device=device), atol=1e-5), "weights don't sum to 1"
    active = w[:, 1:][mask]
    assert active.std() > 0, "degenerate output: identical weight on every active slot"
    assert w.max() < 0.999, "degenerate output: near-one-hot at init"
    print(f"[OK] output non-degenerate: sum=1, active-slot std={active.std():.2e}, max w={w.max():.4f}")

    # --- backward ---
    gidx = panel.slot_gidx[t_idx]  # all valid in-window, safe gather
    y_slots = np.concatenate([y_next[:, :1], np.take_along_axis(y_next, gidx, axis=1)], axis=1)
    growth = (w * torch.tensor(y_slots, dtype=torch.float32, device=device)).sum(dim=1)
    loss = -torch.log(torch.clamp(growth, min=1e-12)).mean()
    loss.backward()

    g = model.conv1.weight.grad
    assert g is not None, "no gradient reached conv1"
    assert torch.isfinite(g).all(), "non-finite gradient at conv1"
    assert g.norm() > 0, "zero gradient at conv1 (dead path)"
    print(f"[OK] backward: loss={loss.item():.6f}, conv1 grad norm={g.norm():.3e}")
    print("\nPRE-FLIGHT SANITY: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
