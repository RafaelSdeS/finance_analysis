"""
train.py — Online Stochastic Batch Learning (paper Sec. 5.3),
docs/EIIE_AGENT_PLAN.md "PVM, OSBL training, split protocol" section.

Key design point: within a mini-batch of n_b CONSECUTIVE periods, the PVM
supplies w_{t-1} only as the network's INPUT (a detached constant), so all
n_b forward passes run in parallel -- the paper's stated benefit of the
PVM over a truly sequential rollout. The LOSS, however, is the batch-window
slice of eq. 21's whole-period objective R = mean ln(mu_t * y_{t+1} . w_t)
with EVERY w a differentiable network output (eq. 23): each period's
mu_t is computed against the previous row's OUTPUT, not its PVM read.

Concretely:
    w_{t-1}^in = PVM.read_slots(t-1, ...)     <- detached, network input only
    w_t        = model(X_t, w_{t-1}^in, mask)    <- parallel over the batch
    r_t        = ln(mu_t(drift(y_t, w_{t-1}), w_t) * y_{t+1} . w_t)
                 with w_{t-1} = the batch's own output row t-1
    loss       = -mean(r_t over the batch)
    PVM.write(t, w_t.detach())                   <- persists for a future read

Two pairings here are load-bearing, both bugs that shipped once:
- y_{t+1} . w_t (eq. 22): w_t earns the NEXT period's price relatives.
  Pairing y_t . w_{t-1} instead (correct backtest bookkeeping, wrong loss)
  makes the return term a constant -- the only gradient left is mu's cost
  penalty, whose optimum is "never trade", and the agent collapses to the
  PVM's all-cash init.
- w_{t-1} in mu_t from the batch's own outputs: gives w_t the NEXT term's
  unwind-cost gradient (mu_{t+1} depends on w_t). Reading it detached from
  the PVM cuts the only cross-period link, so nothing rewards holding a
  position and the agent flip-flops single-name bets every few days.
"""

from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from .config import ExperimentConfig
from .data import CASH_GIDX, PricePanel
from .environment import drift_weights_torch, run_backtest, solve_mu_torch, BacktestResult
from .networks import EIIECNN
from .pvm import PortfolioVectorMemory, scatter_to_global_row


def sample_batch_starts(t_now: int, n_b: int, beta: float, n_batches: int,
                         min_t: int, rng: np.random.Generator) -> np.ndarray:
    """eq. 26: P(t_b) ∝ beta*(1-beta)^(t_now-t_b-n_b), favoring more recent
    windows, truncated + renormalized to the valid range [min_t, t_now-n_b]."""
    k_max = t_now - n_b - min_t
    if k_max < 0:
        raise ValueError(f"not enough history for a batch: t_now={t_now}, n_b={n_b}, min_t={min_t}")
    k = np.arange(k_max + 1)
    weights = beta * (1.0 - beta) ** k
    weights = weights / weights.sum()
    chosen_k = rng.choice(k, size=n_batches, p=weights)
    return t_now - n_b - chosen_k


def _batch_tensors(panel: PricePanel, t_idx: np.ndarray, features, device):
    X = panel.window_tensor_batch(t_idx, features)
    y = panel.price_relative_batch(t_idx)
    y_next = panel.price_relative_batch(t_idx + 1)  # earned by w_t; callers keep t_idx+1 <= t_now
    return (
        torch.tensor(X, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.float32, device=device),
        torch.tensor(y_next, dtype=torch.float32, device=device),
        torch.tensor(panel.slot_gidx[t_idx], dtype=torch.long, device=device),
        torch.tensor(panel.valid[t_idx], dtype=torch.bool, device=device),
        torch.tensor(t_idx - 1, dtype=torch.long, device=device),
        torch.tensor(t_idx, dtype=torch.long, device=device),
    )


def train_step(model: EIIECNN, pvm: PortfolioVectorMemory, panel: PricePanel,
               optimizer: torch.optim.Optimizer, t_idx: np.ndarray, features,
               c_sell: float, c_buy: float, mu_iters: int, grad_clip_norm: float,
               device: str = "cpu") -> float:
    """One OSBL gradient step over a batch of CONSECUTIVE period indices
    (contiguity is load-bearing: the loss chains each period's w_{t-1} to the
    previous row's output -- see module docstring)."""
    assert len(t_idx) == 1 or (np.diff(t_idx) == 1).all(), "train_step requires consecutive periods"
    X, y, y_next, slot_gidx, valid, prev_rows, curr_rows = _batch_tensors(panel, t_idx, features, device)

    w_prev_slots = pvm.read_slots(prev_rows, slot_gidx, valid)  # [B, m+1], detached (buffer holds no grad)
    w = model(X, w_prev_slots[:, 1:], valid)                     # [B, m+1], differentiable

    n_global = pvm.n_global
    w_prev_global = pvm.read_global(prev_rows)                     # [B, n_global], detached
    w_target_global = scatter_to_global_row(slot_gidx, w, n_global)[:, :n_global]  # differentiable

    # Within the contiguous batch, w_{t-1} in the LOSS is the batch's own
    # differentiable output for t-1 (paper eq. 23: every w is pi_theta), so
    # each w_t also receives mu_{t+1}'s unwind-cost gradient from the next
    # term -- the cross-period signal that rewards holding. Only the first
    # row falls back to the detached PVM read (no earlier output exists).
    w_prev_loss = torch.cat([w_prev_global[:1], w_target_global[:-1]], dim=0)

    growth = (y_next * w_target_global).sum(dim=1)                   # y_{t+1} . w_t (eq. 22) -- differentiable, see module docstring
    w_drift_global = drift_weights_torch(y, w_prev_loss)               # eq. 7
    mu = solve_mu_torch(w_drift_global, w_target_global, c_sell, c_buy, k=mu_iters)

    reward = torch.log(torch.clamp(mu * growth, min=1e-12))  # loss-stability clamp before log
    loss = -reward.mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()

    pvm.write(curr_rows, slot_gidx, w.detach())
    return float(loss.item())


def pretrain(model: EIIECNN, pvm: PortfolioVectorMemory, panel: PricePanel,
             optimizer: torch.optim.Optimizer, cfg: ExperimentConfig,
             train_end_idx: int, device: str = "cpu") -> list:
    """OSBL pretraining over the train split: `pretrain_steps` mini-batches,
    each of n_b consecutive periods, sampled with eq. 26's recency bias
    relative to train_end_idx. Returns the per-step loss history (the
    "training reward curve" is -loss)."""
    # panel.start_idx, not panel.window - 1: sampling must stay inside the
    # 2011-2026 experiment window, where every period is guaranteed exactly
    # n_slots active members (docs/EIIE_AGENT_PLAN.md) -- pre-window periods
    # have fewer members and isolated data-quality zero-price rows.
    min_t = panel.start_idx
    rng = np.random.default_rng(cfg.train.seed)
    losses = []
    pbar = tqdm(range(cfg.train.pretrain_steps), desc="pretrain", unit="step")
    for _ in pbar:
        t_b = int(sample_batch_starts(train_end_idx, cfg.train.batch_size, cfg.train.beta, 1, min_t, rng)[0])
        t_idx = np.arange(t_b, t_b + cfg.train.batch_size)
        loss = train_step(model, pvm, panel, optimizer, t_idx, cfg.data.features,
                           cfg.costs.c_sell, cfg.costs.c_buy, cfg.costs.train_mu_iters,
                           cfg.train.grad_clip_norm, device)
        losses.append(loss)
        pbar.set_postfix(loss=f"{loss:.6f}")
    return losses


def agent_forward(model: EIIECNN, pvm: PortfolioVectorMemory, panel: PricePanel,
                   t: int, features, device: str = "cpu") -> np.ndarray:
    """One inference-only forward pass at period t: read w_{t-1} from the
    PVM in slot space, run the network, write w_t back, return it in GLOBAL
    space (ready for environment.run_backtest's weight_fn contract)."""
    X = torch.tensor(panel.window_tensor(t, features)[None], dtype=torch.float32, device=device)
    slot_gidx = torch.tensor(panel.slot_gidx[t][None], dtype=torch.long, device=device)
    valid = torch.tensor(panel.valid[t][None], dtype=torch.bool, device=device)
    prev_row = torch.tensor([t - 1], dtype=torch.long, device=device)
    curr_row = torch.tensor([t], dtype=torch.long, device=device)

    with torch.no_grad():
        w_prev = pvm.read_slots(prev_row, slot_gidx, valid)
        w = model(X, w_prev[:, 1:], valid)
    pvm.write(curr_row, slot_gidx, w)

    w_slots = w[0].cpu().numpy()
    mask = panel.valid[t]
    active_gidx = panel.slot_gidx[t][mask]
    w_global = np.zeros(panel.n_global)
    w_global[CASH_GIDX] = w_slots[0]
    w_global[active_gidx] = w_slots[1:][mask]
    return w_global


def run_online_backtest(model: EIIECNN, pvm: PortfolioVectorMemory, panel: PricePanel,
                         optimizer: torch.optim.Optimizer, cfg: ExperimentConfig,
                         start_idx: Optional[int] = None, end_idx: Optional[int] = None,
                         device: str = "cpu") -> BacktestResult:
    """The paper's OSBL online backtest (Sec. 5.3): at each period the agent
    acts first (inference only), then -- having now seen that period's price
    move -- trains `rolling_steps` additional OSBL updates sampling from
    everything seen so far, before moving to the next period. Reuses
    environment.run_backtest for all the bookkeeping via its on_step hook,
    so the agent gets exactly the same cost/reward treatment as every
    baseline."""
    start_idx = panel.start_idx if start_idx is None else start_idx
    end_idx = panel.end_idx if end_idx is None else end_idx
    min_t = panel.start_idx  # see pretrain()'s comment: stay inside the guaranteed-50-members window
    rng = np.random.default_rng(cfg.train.seed)

    def agent_weight_fn(t, w_prev_np, w_drift_np, panel):
        return agent_forward(model, pvm, panel, t, cfg.data.features, device)

    pbar = tqdm(total=end_idx - start_idx + 1, desc="online backtest", unit="day")

    def after_step(t):
        loss = None
        for _ in range(cfg.train.rolling_steps):
            t_b = int(sample_batch_starts(t, cfg.train.batch_size, cfg.train.beta, 1, min_t, rng)[0])
            t_idx = np.arange(t_b, t_b + cfg.train.batch_size)
            loss = train_step(model, pvm, panel, optimizer, t_idx, cfg.data.features,
                       cfg.costs.c_sell, cfg.costs.c_buy, cfg.costs.train_mu_iters,
                       cfg.train.grad_clip_norm, device)
        pbar.update(1)
        if loss is not None:
            pbar.set_postfix(date=str(panel.dates[t].date()), loss=f"{loss:.6f}")

    result = run_backtest(panel, agent_weight_fn, cfg.costs.c_sell, cfg.costs.c_buy,
                           start_idx, end_idx, cfg.costs.backtest_mu_tol, on_step=after_step)
    pbar.close()
    return result


def save_checkpoint(path, model: EIIECNN, optimizer: torch.optim.Optimizer,
                     pvm: PortfolioVectorMemory, step: int, extra: Optional[dict] = None) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "pvm_buffer": pvm.buffer,
        "step": step,
        "extra": extra or {},
    }, path)


def load_checkpoint(path, model: EIIECNN, optimizer: torch.optim.Optimizer,
                     pvm: PortfolioVectorMemory, map_location: str = "cpu") -> tuple:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    pvm.buffer = ckpt["pvm_buffer"].to(pvm.buffer.device)
    return ckpt["step"], ckpt.get("extra", {})
