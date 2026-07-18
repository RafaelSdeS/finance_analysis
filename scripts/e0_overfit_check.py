"""E0 — overfit sanity check (EIIE_IMPROVEMENT_PLAN.md, Stage 0).

Can the current architecture + optimizer memorize a tiny (~126-day) train
window? Costs and entropy are 0 in the config, so the objective is exactly
per-day log-return, whose optimum is all-in the next day's best asset --
a crisp memorization target. w_{t-1} is zeroed at the network boundary
(ZeroWPrev below) so PVM path-dependence can't interfere: with costs 0 its
only remaining role is noise for this target.

Pass:  in-sample k=1 Spearman >= 0.3 AND day-varying weights (argmax switches).
Fail:  Spearman ~ 0, or a static concentrated position.

Run from project root:  python scripts/e0_overfit_check.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.rl_agent.baselines import run_baseline
from src.rl_agent.config import ExperimentConfig
from src.rl_agent.data import CASH_GIDX, load_price_panel
from src.rl_agent.diagnostics import ranking_quality
from src.rl_agent.environment import run_backtest
from src.rl_agent.metrics import total_return
from src.rl_agent.networks import EIIECNN
from src.rl_agent.paths import ROOT
from src.rl_agent.pvm import PortfolioVectorMemory
from src.rl_agent.sanity import seed_everything
from src.rl_agent.train import agent_forward, pretrain

CONFIG = "configs/eiie_overfit_check.json"
SPEARMAN_PASS = 0.3


class ZeroWPrev(torch.nn.Module):
    """E0 only: zero the w_{t-1} input feature map before the real forward.
    Script-local wrapper -- production train/inference paths untouched."""

    def __init__(self, inner: EIIECNN):
        super().__init__()
        self.inner = inner

    def forward(self, X, w_prev, mask):
        return self.inner(X, torch.zeros_like(w_prev), mask)


def make_saturation_probe(panel, features, device, every: int = 500):
    """Saturation check: did the policy freeze early into a corner (softmax
    entropy -> ~0, gradient vanishes) well before pretrain_steps, the same
    mechanism as the cash-attractor bug just landing on a different corner
    (e.g. one-hot momentum-chasing)? Every `every` steps, runs an EXTRA
    forward pass (no grad, w_prev=0 to match ZeroWPrev) on a FIXED reference
    day -- never the real training batch -- so probing never touches the PVM
    or the actual training trajectory. Logs (step, max non-cash weight,
    softmax entropy). Returns (on_step callback, history list to read after
    pretrain() returns)."""
    t_ref = panel.start_idx + (panel.end_idx - panel.start_idx) // 2
    X_ref = torch.tensor(panel.window_tensor(t_ref, features), dtype=torch.float32,
                          device=device).unsqueeze(0)
    mask_ref = torch.tensor(panel.valid[t_ref], dtype=torch.bool, device=device).unsqueeze(0)
    w_prev_ref = torch.zeros(1, mask_ref.shape[1], device=device)
    history = []

    def probe(step, loss, model):
        if step % every != 0:
            return
        was_training = model.training
        model.eval()
        with torch.no_grad():
            w = model(X_ref, w_prev_ref, mask_ref)
            max_w = float(w[0, 1:].max())
            entropy = float(-(w * w.clamp_min(1e-12).log()).sum())
        if was_training:
            model.train()
        history.append({"step": step, "loss": loss, "max_weight": max_w, "entropy": entropy})

    return probe, history


def main():
    cfg = ExperimentConfig.from_json(CONFIG)
    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    seed_everything(cfg.train.seed)

    panel = load_price_panel(cfg.data, n_slots=cfg.model.n_assets)
    n_days = panel.end_idx - panel.start_idx + 1
    print(f"E0 window: {panel.dates[panel.start_idx].date()} .. {panel.dates[panel.end_idx].date()} "
          f"({n_days} trading days), seed={cfg.train.seed}, device={device}")

    # DataConfig.cash_mode only selects the risk-free baseline for Sharpe/Sortino
    # reporting (experiment.py) -- it does NOT change what cash actually earns in
    # price_relative (data.py always reads the real CDI series). E0's first run
    # (2015 H1, CDI-accruing cash + entropy=0) collapsed to 100% cash immediately --
    # the exact cash-attractor bug entropy_beta exists to prevent, confounding the
    # memorization read. Overriding to zero-return cash removes that asymmetric pull
    # so the crisp log-return objective has no "safe harbor" biasing it away from
    # actually trying to memorize. Script-local; no production code changed.
    panel.cdi_factor = np.ones_like(panel.cdi_factor)
    print("cash override: cdi_factor forced to 1.0 (zero return) for this diagnostic only")

    model = ZeroWPrev(EIIECNN(cfg.data.window, cfg.model.conv1_out_channels,
                              cfg.model.conv2_out_channels, len(cfg.data.features))).to(device)
    pvm = PortfolioVectorMemory(len(panel.dates), panel.n_global, slot_gidx=panel.slot_gidx,
                                valid=panel.valid, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.l2)

    probe, sat_history = make_saturation_probe(panel, cfg.data.features, device)
    losses, _, _ = pretrain(model, pvm, panel, optimizer, cfg,
                            train_end_idx=panel.end_idx, device=device, on_step=probe)
    assert np.all(np.isfinite(losses)), "non-finite training loss"

    max_entropy = float(np.log(cfg.model.n_assets + 1))  # uniform-over-(assets+cash) ceiling
    print("\n=== saturation probe (fixed reference day, w_prev=0) ===")
    print(f"{'step':>8} {'loss':>12} {'max_weight':>10} {'entropy':>10}")
    checkpoints = [0, len(sat_history) // 10, len(sat_history) // 2, -1] if len(sat_history) > 3 else range(len(sat_history))
    seen = set()
    for i in checkpoints:
        if i in seen or not sat_history:
            continue
        seen.add(i)
        h = sat_history[i]
        print(f"{h['step']:>8} {h['loss']:>12.6f} {h['max_weight']:>10.4f} {h['entropy']:>10.4f}")
    early_i = max(1, len(sat_history) // 10)
    if len(sat_history) > early_i:
        early_entropy = sat_history[early_i]["entropy"]
        final_entropy = sat_history[-1]["entropy"]
        frozen = (early_entropy < 0.05 * max_entropy) and (abs(final_entropy - early_entropy) < 0.02 * max_entropy)
        print(f"entropy at step {sat_history[early_i]['step']} (10%): {early_entropy:.4f} vs "
              f"final (step {sat_history[-1]['step']}): {final_entropy:.4f}  (ceiling {max_entropy:.4f})")
        print(f"SATURATED EARLY: {frozen}" + (" -- froze into a corner before 10% of training and never moved again"
                                               if frozen else " -- entropy kept changing after the early checkpoint"))

    # frozen in-sample evaluation over the SAME tiny window it trained on
    model.eval()

    def weight_fn(t, w_prev_np, w_drift_np, p):
        return agent_forward(model, pvm, panel, t, cfg.data.features, device)

    result = run_backtest(panel, weight_fn, cfg.costs.c_sell, cfg.costs.c_buy,
                          panel.start_idx, panel.end_idx, cfg.costs.backtest_mu_tol)

    tickers = np.array(["cash"] + list(panel.asset_index.tickers))
    quality = ranking_quality(result.weights, result.dates.values.astype("datetime64[D]"),
                              tickers, panel, k_list=(1,))
    rho = quality[1]["mean_spearman"]

    w = result.weights
    non_cash = np.delete(w, CASH_GIDX, axis=1)
    argmax_ids = non_cash.argmax(axis=1)
    switches = int((argmax_ids[1:] != argmax_ids[:-1]).sum())
    cash_mean = float(w[:, CASH_GIDX].mean())
    top1_mean = float(non_cash.max(axis=1).mean())

    agent_ret = total_return(result)
    ucrp_ret = total_return(run_baseline("ucrp", panel, 0.0, 0.0,
                                         start_idx=panel.start_idx, end_idx=panel.end_idx))
    best_ret = total_return(run_baseline("best_stock", panel, 0.0, 0.0,
                                         start_idx=panel.start_idx, end_idx=panel.end_idx))

    passed = rho >= SPEARMAN_PASS and switches > 0
    verdict = "PASS" if passed else "FAIL"

    print("\n=== E0 overfit check ===")
    print(f"in-sample k=1 mean Spearman : {rho:.4f}   (pass bar: >= {SPEARMAN_PASS})")
    print(f"top-10 hit rate             : {quality[1]['mean_top10_hit_rate']:.3f}")
    print(f"argmax switches             : {switches} over {len(argmax_ids) - 1} day-pairs")
    print(f"mean cash fraction          : {cash_mean:.3f}")
    print(f"mean top-1 asset weight     : {top1_mean:.3f}")
    print(f"train-window return: agent {agent_ret:+.2%} | UCRP {ucrp_ret:+.2%} | "
          f"best-stock (buy-hold) {best_ret:+.2%}")
    print(f"final train loss (mean last 100): {np.mean(losses[-100:]):.6f}")
    print(f"\nVERDICT: {verdict}")

    out_dir = ROOT / cfg.experiment.out_dir / \
        f"{cfg.experiment.name}_{datetime.now():%Y%m%dT%H%M%S%f}_{os.getpid()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")
    np.savez_compressed(out_dir / "weights.npz", weights=w.astype(np.float32),
                        dates=result.dates.values.astype("datetime64[D]"), tickers=tickers)
    (out_dir / "e0_result.json").write_text(json.dumps({
        "verdict": verdict, "spearman_k1": rho, "ranking_quality_k1": quality[1],
        "argmax_switches": switches, "mean_cash_fraction": cash_mean,
        "mean_top1_weight": top1_mean, "agent_return": agent_ret,
        "ucrp_return": ucrp_ret, "best_stock_return": best_ret,
        "final_loss_mean100": float(np.mean(losses[-100:])),
        "zero_w_prev": True, "n_days": n_days,
        "saturation_probe": sat_history,
    }, indent=2))
    print(f"artifacts: {out_dir}")


if __name__ == "__main__":
    main()
