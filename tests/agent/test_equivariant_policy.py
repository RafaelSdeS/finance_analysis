"""
Unit tests for policy.py's permutation-equivariant architecture (synthetic, no data files).

DIAGNOSIS_PLAN.md Phase 2.1 / roadmap M6.1: the flat 1224-dim MLP (SB3's default
"MlpPolicy") must learn ticker-SLOT-specific weights — what it learns about slot 3 doesn't
transfer to slot 17, and reordering the universe breaks it. These tests first demonstrate
that broken inductive bias on a generic dense network (the problem), then verify
EquivariantMlpExtractor's actor path is EXACTLY equivariant and its critic path is EXACTLY
invariant under any ticker permutation (the solution), and that the architecture is
actually trainable (gradients flow to a sensible per-ticker scoring function).
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.policy import EquivariantMlpExtractor

N_TICKERS = 12
N_FEATURES = 5
OBS_DIM = N_TICKERS * N_FEATURES + 2 * N_TICKERS


def _make_obs(seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    feat = rng.normal(size=(N_TICKERS, N_FEATURES)).astype(np.float32)
    mask = np.ones(N_TICKERS, dtype=np.float32)
    mask[-2:] = 0.0  # a couple of inactive slots, matching real usage
    prev_w = rng.uniform(size=N_TICKERS).astype(np.float32)
    prev_w *= mask
    prev_w /= max(prev_w.sum(), 1e-6)
    obs = np.concatenate([feat.ravel(), mask, prev_w])
    return torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), feat, mask, prev_w


def _permute_obs(feat: np.ndarray, mask: np.ndarray, prev_w: np.ndarray, perm: np.ndarray) -> torch.Tensor:
    obs = np.concatenate([feat[perm].ravel(), mask[perm], prev_w[perm]])
    return torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)


def main() -> None:
    torch.manual_seed(0)
    obs, feat, mask, prev_w = _make_obs(seed=42)
    rng = np.random.default_rng(1)
    perm = rng.permutation(N_TICKERS)
    obs_perm = _permute_obs(feat, mask, prev_w, perm)

    print("Test 1 (problem): a generic flat MLP does NOT preserve equivariance under a "
          "ticker-slot permutation")
    flat_mlp = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU(), nn.Linear(64, N_TICKERS))
    with torch.no_grad():
        out = flat_mlp(obs).squeeze(0).numpy()
        out_perm = flat_mlp(obs_perm).squeeze(0).numpy()
    expected_if_equivariant = out[perm]
    max_err = np.abs(out_perm - expected_if_equivariant).max()
    assert max_err > 0.1, (
        f"expected a generic dense MLP to badly violate equivariance, got max error {max_err} "
        f"(suspiciously small — check the test setup)"
    )
    print(f"  ✓ flat MLP violates equivariance: max|actual_perm - expected_perm| = {max_err:.4f} "
          f"(the broken inductive bias this architecture exists to fix)")

    print("\nTest 2 (solution): EquivariantMlpExtractor's actor path is EXACTLY equivariant")
    extractor = EquivariantMlpExtractor(N_TICKERS, N_FEATURES, hidden_dim=32, encoder_dim=16)
    extractor.eval()
    with torch.no_grad():
        latent_pi = extractor.forward_actor(obs).squeeze(0).numpy()
        latent_pi_perm = extractor.forward_actor(obs_perm).squeeze(0).numpy()
    max_err_equiv = np.abs(latent_pi_perm - latent_pi[perm]).max()
    assert max_err_equiv < 1e-5, (
        f"expected EXACT equivariance (permute(logits) == logits(permute(obs))), got max error {max_err_equiv}"
    )
    print(f"  ✓ forward_actor(permute(obs)) == permute(forward_actor(obs)): max error {max_err_equiv:.2e}")

    print("\nTest 3 (solution): EquivariantMlpExtractor's critic path is EXACTLY invariant")
    with torch.no_grad():
        latent_vf = extractor.forward_critic(obs).squeeze(0).numpy()
        latent_vf_perm = extractor.forward_critic(obs_perm).squeeze(0).numpy()
    max_err_inv = np.abs(latent_vf_perm - latent_vf).max()
    assert max_err_inv < 1e-5, f"expected EXACT invariance (same value regardless of order), got max error {max_err_inv}"
    print(f"  ✓ forward_critic(permute(obs)) == forward_critic(obs): max error {max_err_inv:.2e}")

    print("\nTest 4 (solution): masked-out (inactive) slots don't influence the context vector")
    # Corrupting an INACTIVE slot's features must not change any ACTIVE slot's output --
    # the masked mean/std context must genuinely ignore it, not just down-weight it.
    feat_corrupt = feat.copy()
    feat_corrupt[-1] = 999.0  # slot -1 is inactive (mask[-2:] = 0)
    obs_corrupt = torch.as_tensor(
        np.concatenate([feat_corrupt.ravel(), mask, prev_w]), dtype=torch.float32
    ).unsqueeze(0)
    with torch.no_grad():
        latent_pi_corrupt = extractor.forward_actor(obs_corrupt).squeeze(0).numpy()
    active = mask.astype(bool)
    max_err_mask = np.abs(latent_pi_corrupt[active] - latent_pi[active]).max()
    assert max_err_mask < 1e-5, (
        f"corrupting an INACTIVE slot changed an ACTIVE slot's logit by {max_err_mask} -- "
        f"the context vector is leaking inactive-slot data"
    )
    print(f"  ✓ corrupting an inactive slot's features doesn't change any active slot's output "
          f"(max error {max_err_mask:.2e})")

    print("\nTest 5: the architecture is actually trainable (gradients flow, learns a "
          "sensible per-ticker scoring function)")
    torch.manual_seed(7)
    train_extractor = EquivariantMlpExtractor(N_TICKERS, N_FEATURES, hidden_dim=32, encoder_dim=16)
    optimizer = torch.optim.Adam(train_extractor.policy_net.parameters(), lr=1e-2)

    rng2 = np.random.default_rng(2)
    n_samples = 200
    feats = rng2.normal(size=(n_samples, N_TICKERS, N_FEATURES)).astype(np.float32)
    masks = np.ones((n_samples, N_TICKERS), dtype=np.float32)
    prev_ws = np.zeros((n_samples, N_TICKERS), dtype=np.float32)
    # Teacher: target logit = feature[0] for every ticker (a simple, learnable per-ticker rule)
    targets = feats[:, :, 0].copy()

    obs_batch = torch.as_tensor(
        np.concatenate([feats.reshape(n_samples, -1), masks, prev_ws], axis=1), dtype=torch.float32
    )
    target_batch = torch.as_tensor(targets, dtype=torch.float32)

    losses = []
    for _ in range(300):
        pred = train_extractor.forward_actor(obs_batch)
        loss = ((pred - target_batch) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"  loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0] * 0.1, f"expected loss to drop by >90%, got {losses[0]:.4f} -> {losses[-1]:.4f}"
    print("  ✓ gradients flow and the shared-encoder architecture learns the target per-ticker rule")

    print("\n✓ All equivariant_policy tests passed")


if __name__ == "__main__":
    main()
