"""
sanity.py — automated invariant checks run before any real training
(docs/EIIE_AGENT_PLAN.md "Sanity checks"). These check STRUCTURAL
invariants of the actual wired-up model/PVM/panel (determinism, simplex
weights, positive/finite portfolio values, finite gradients, baselines
running cleanly) -- never whether the agent behaves any particular way. A
dominant-asset toy market is a useful diagnostic but deliberately not
tested here as a pass/fail gate: real markets don't guarantee an agent
should fully concentrate on one asset.
"""

import random
from dataclasses import dataclass, field

import numpy as np
import torch

from .baselines import run_baseline
from .config import ExperimentConfig
from .data import PricePanel
from .networks import EIIECNN
from .pvm import PortfolioVectorMemory
from .train import train_step


@dataclass
class SanityReport:
    checks: dict = field(default_factory=dict)  # name -> (ok, detail)

    @property
    def passed(self) -> bool:
        return all(ok for ok, _ in self.checks.values())

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks[name] = (bool(ok), detail)

    def __str__(self) -> str:
        lines = [f"SanityReport ({'PASSED' if self.passed else 'FAILED'}):"]
        for name, (ok, detail) in self.checks.items():
            glyph = "OK  " if ok else "FAIL"
            lines.append(f"  [{glyph}] {name}" + (f" -- {detail}" if detail else ""))
        return "\n".join(lines)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_model(cfg: ExperimentConfig, device: str = "cpu") -> EIIECNN:
    return EIIECNN(cfg.data.window, cfg.model.conv1_out_channels,
                    cfg.model.conv2_out_channels, len(cfg.data.features)).to(device)


def _short_train_run(cfg: ExperimentConfig, panel: PricePanel, t0: int, n_steps: int, device: str):
    seed_everything(cfg.train.seed)
    model = _build_model(cfg, device)
    pvm = PortfolioVectorMemory(len(panel.dates), panel.n_global, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.l2)
    losses = []
    for i in range(n_steps):
        t_idx = np.arange(t0 + i, t0 + i + cfg.train.batch_size)
        losses.append(train_step(model, pvm, panel, optimizer, t_idx, cfg.data.features,
                                  cfg.costs.c_sell, cfg.costs.c_buy, cfg.costs.train_mu_iters,
                                  cfg.train.grad_clip_norm, device))
    return losses, model, pvm, optimizer


def run_sanity_checks(cfg: ExperimentConfig, panel: PricePanel, device: str = "cpu",
                       n_steps: int = 3) -> SanityReport:
    """Run every gate. Called by experiment.py before pretraining starts;
    a failing report should stop the run, not just be logged."""
    report = SanityReport()
    min_valid_t = panel.window - 1
    # +1 so t0-1 is also a valid window_tensor index (PVM reads t-1's slot layout too)
    t0 = max(panel.start_idx, min_valid_t + 1)

    if t0 + cfg.train.batch_size + n_steps > panel.end_idx + 1:
        report.add("enough_history_for_checks", False,
                    f"t0={t0}, batch_size={cfg.train.batch_size}, n_steps={n_steps}, "
                    f"end_idx={panel.end_idx} -- window too small to run sanity checks")
        return report

    # --- deterministic seeding: two identically-seeded short runs must match exactly ---
    losses_a, model_a, _, _ = _short_train_run(cfg, panel, t0, n_steps, device)
    losses_b, _, _, _ = _short_train_run(cfg, panel, t0, n_steps, device)
    report.add("deterministic_seeding", np.allclose(losses_a, losses_b),
               f"run A={losses_a}, run B={losses_b}")

    # --- weights on the simplex, finite, respecting the mask ---
    with torch.no_grad():
        X = torch.tensor(panel.window_tensor(t0, cfg.data.features)[None], dtype=torch.float32, device=device)
        w_prev = torch.zeros(1, cfg.model.n_assets, device=device)
        mask = torch.tensor(panel.valid[t0][None], device=device)
        w = model_a(X, w_prev, mask)
    w_np = w.cpu().numpy()[0]
    report.add("weights_on_simplex", bool(np.all(w_np >= -1e-6)) and bool(np.isclose(w_np.sum(), 1.0, atol=1e-4)),
               f"sum={w_np.sum():.6f}, min={w_np.min():.6f}")
    report.add("weights_finite", bool(np.all(np.isfinite(w_np))), "")

    # --- finite gradients + finite loss on a fresh model's first batch ---
    losses_c, model_c, pvm_c, _ = _short_train_run(cfg, panel, t0, 1, device)
    grads_finite = all(torch.isfinite(p.grad).all().item() for p in model_c.parameters() if p.grad is not None)
    report.add("finite_gradients_first_batch", grads_finite, "")
    report.add("finite_loss", bool(np.isfinite(losses_c[0])), f"loss={losses_c[0]}")

    # --- PVM stays finite and no weight leaks outside [0, 1] after training ---
    pvm_slice = pvm_c.buffer.cpu().numpy()
    report.add("pvm_finite", bool(np.all(np.isfinite(pvm_slice))), "")

    # --- baselines run cleanly on the real panel (end-to-end wiring smoke check) ---
    try:
        bl = run_baseline("ucrp", panel, cfg.costs.c_sell, cfg.costs.c_buy,
                           start_idx=t0, end_idx=min(t0 + 30, panel.end_idx))
        ok = bool(np.all(bl.portfolio_value > 0)) and bool(np.all(np.isfinite(bl.portfolio_value)))
        detail = f"final_pv={bl.portfolio_value[-1]:.6f}"
    except Exception as e:  # noqa: BLE001 -- a sanity gate must report the failure, not crash the caller
        ok, detail = False, f"{type(e).__name__}: {e}"
    report.add("baselines_run_cleanly", ok, detail)

    # --- zero transaction cost vs. real cost: forced trading must cost strictly more at c>0 ---
    from .environment import run_backtest

    def _flip_flop(t, w_prev, w_drift, panel):
        w = np.zeros(panel.n_global)
        active = panel.slot_gidx[t][panel.valid[t]]
        if t % 2 == 0:
            w[0] = 0.5
            w[active[0]] = 0.5
        else:
            w[0] = 0.5
            w[active[-1]] = 0.5
        return w

    zero_cost = run_backtest(panel, _flip_flop, 0.0, 0.0, t0, min(t0 + 10, panel.end_idx))
    real_cost = run_backtest(panel, _flip_flop, cfg.costs.c_sell, cfg.costs.c_buy, t0, min(t0 + 10, panel.end_idx))
    report.add("zero_cost_no_drag", bool(np.allclose(zero_cost.mu, 1.0)),
               f"mu range=[{zero_cost.mu.min():.6f}, {zero_cost.mu.max():.6f}]")
    report.add("real_cost_reduces_value", real_cost.portfolio_value[-1] < zero_cost.portfolio_value[-1],
               f"zero_cost_final={zero_cost.portfolio_value[-1]:.6f}, "
               f"real_cost_final={real_cost.portfolio_value[-1]:.6f}")

    return report
