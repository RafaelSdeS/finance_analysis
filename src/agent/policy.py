"""
Permutation-equivariant policy (DIAGNOSIS_PLAN.md Phase 2.1 / roadmap M6.1).

The flat 1224-dim MLP (SB3's stock "MlpPolicy") must learn ticker-slot-specific weights:
what it learns about slot 3's features doesn't transfer to slot 17, and any universe
reordering breaks it. Cross-sectional ranking — the actual task — is permutation-
equivariant by nature: "how good is this stock" shouldn't depend on which column it lands
in. This module applies a SHARED per-ticker encoder to every ticker slot (parameter tying,
not just more capacity), with a cross-sectional context vector (masked mean/std across
active names) broadcast to every ticker so the shared encoder can reason about a stock's
features RELATIVE to that day's market, not just in absolute terms.

`EquivariantMlpExtractor` mimics SB3's `MlpExtractor` interface (`.policy_net`,
`.value_net`, `.forward_actor`, `.forward_critic`, `.latent_dim_pi`, `.latent_dim_vf`) so
`bc_pretrain.py`'s existing BC warm-start code works unmodified against it.
`EquivariantActorCriticPolicy` wires it into SB3's `ActorCriticPolicy`, additionally
replacing the default `action_net` (a full `Linear(n_tickers, n_tickers)`, which would
silently break equivariance by mixing every ticker's latent into every output logit) with
`nn.Identity()` — the extractor's per-ticker actor head outputs the logits directly.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

EPS = 1e-6


class SharedTickerEncoder(nn.Module):
    """Permutation-equivariant feature transform: the SAME small MLP applied to every
    ticker slot, each slot's input augmented with a cross-sectional context vector
    (masked mean/std over ACTIVE tickers that day) so the shared weights can express
    relative, not just absolute, judgments.

    Equivariance holds exactly: permuting ticker slots in the input permutes the output
    identically, because (a) the per-slot MLP is literally the same weight matrix applied
    independently per slot (a batched matmul, no cross-slot mixing), and (b) the context
    vector is a masked mean/std — a symmetric (order-independent) function of the other
    slots — so it takes the same value regardless of slot order.
    """

    def __init__(self, n_features: int, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        # +2 for this slot's own (mask, prev_weight); the context vector is
        # mean+std of n_features, i.e. 2*n_features, appended per slot.
        in_dim = n_features + 2 + 2 * n_features
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim), nn.ReLU(),
        )

    def forward(self, feat: torch.Tensor, mask: torch.Tensor, prev_w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat:   [batch, n_tickers, n_features]
            mask:   [batch, n_tickers]  (1.0 active / 0.0 inactive)
            prev_w: [batch, n_tickers]
        Returns:
            [batch, n_tickers, out_dim] per-ticker encoded representation.
        """
        n_active = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [batch, 1]
        masked_feat = feat * mask.unsqueeze(-1)
        ctx_mean = masked_feat.sum(dim=1) / n_active  # [batch, n_features]
        ctx_var = (((feat - ctx_mean.unsqueeze(1)) ** 2) * mask.unsqueeze(-1)).sum(dim=1) / n_active
        ctx_std = torch.sqrt(ctx_var.clamp(min=EPS))  # [batch, n_features]

        n_tickers = feat.shape[1]
        context = torch.cat([ctx_mean, ctx_std], dim=-1).unsqueeze(1).expand(-1, n_tickers, -1)  # [batch, n_tickers, 2*n_features]
        per_ticker_in = torch.cat([feat, mask.unsqueeze(-1), prev_w.unsqueeze(-1), context], dim=-1)
        return self.net(per_ticker_in)  # shared Linear layers broadcast over the [batch, n_tickers] leading dims


class EquivariantMlpExtractor(nn.Module):
    """Drop-in replacement for SB3's MlpExtractor, exposing the same attribute names
    bc_pretrain.py's BC warm-start already reads/optimizes (`policy_net`, `value_net`,
    `forward_actor`, `forward_critic`)."""

    def __init__(self, n_tickers: int, n_features: int, hidden_dim: int = 64, encoder_dim: int = 32):
        super().__init__()
        self.n_tickers = n_tickers
        self.n_features = n_features

        # Separate actor/critic encoders (no shared trunk) — matches SB3's own default
        # net_arch=[256,256] behavior (a plain list = fully separate policy/value stacks,
        # not a shared-then-split architecture), so bc_pretrain.py's independent
        # policy_net/value_net optimization stays correct unmodified.
        self.policy_net = nn.ModuleDict({
            "encoder": SharedTickerEncoder(n_features, hidden_dim, encoder_dim),
            "head": nn.Linear(encoder_dim, 1),  # per-ticker scalar -> directly the action logit
        })
        self.value_net = nn.ModuleDict({
            "encoder": SharedTickerEncoder(n_features, hidden_dim, encoder_dim),
            "pool_head": nn.Sequential(nn.Linear(encoder_dim, hidden_dim), nn.ReLU()),
        })

        self.latent_dim_pi = n_tickers
        self.latent_dim_vf = hidden_dim

    def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Undo PortfolioEnv._obs()'s flatten: concatenate([features.ravel(), mask, prev_weights])."""
        n = self.n_tickers
        feat_flat_dim = n * self.n_features
        feat = obs[:, :feat_flat_dim].reshape(-1, n, self.n_features)
        mask = obs[:, feat_flat_dim: feat_flat_dim + n]
        prev_w = obs[:, feat_flat_dim + n: feat_flat_dim + 2 * n]
        return feat, mask, prev_w

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        feat, mask, prev_w = self._split_obs(obs)
        encoded = self.policy_net["encoder"](feat, mask, prev_w)          # [batch, n_tickers, encoder_dim]
        logits = self.policy_net["head"](encoded).squeeze(-1)             # [batch, n_tickers]
        return logits

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        feat, mask, prev_w = self._split_obs(obs)
        encoded = self.value_net["encoder"](feat, mask, prev_w)           # [batch, n_tickers, encoder_dim]
        n_active = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / n_active     # [batch, encoder_dim] -- permutation-invariant
        return self.value_net["pool_head"](pooled)                        # [batch, hidden_dim]

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(obs), self.forward_critic(obs)


class EquivariantActorCriticPolicy(ActorCriticPolicy):
    """SB3 ActorCriticPolicy wired to EquivariantMlpExtractor. Requires n_tickers and
    n_features via policy_kwargs (obs_dim = n_tickers*n_features + 2*n_tickers must match
    the env's actual observation space, checked at construction)."""

    def __init__(self, *args, n_tickers: int, n_features: int,
                 encoder_hidden_dim: int = 64, encoder_dim: int = 32, **kwargs):
        self._n_tickers = n_tickers
        self._n_features = n_features
        self._encoder_hidden_dim = encoder_hidden_dim
        self._encoder_dim = encoder_dim
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        expected_obs_dim = self._n_tickers * self._n_features + 2 * self._n_tickers
        if self.features_dim != expected_obs_dim:
            raise ValueError(
                f"EquivariantActorCriticPolicy: features_dim={self.features_dim} but "
                f"n_tickers={self._n_tickers} * n_features={self._n_features} + "
                f"2*n_tickers = {expected_obs_dim} -- obs layout mismatch."
            )
        self.mlp_extractor = EquivariantMlpExtractor(
            self._n_tickers, self._n_features, self._encoder_hidden_dim, self._encoder_dim,
        )

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # Replace SB3's default action_net (a full Linear(n_tickers, n_tickers), which
        # would mix every ticker's latent into every output logit and silently destroy
        # equivariance) with a pure pass-through: the extractor's per-ticker actor head
        # already outputs the logits directly (latent_dim_pi == action_dim == n_tickers).
        self.action_net = nn.Identity()
