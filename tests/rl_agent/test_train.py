"""
Test: train.py's OSBL batch sampler, gradient step, pretraining loop,
online-backtest driver, and checkpointing (docs/EIIE_AGENT_PLAN.md Phase
6). Everything here runs on a tiny synthetic market and a scaled-down
config (few periods, small batch, few steps) -- fast unit/integration
tests of the code, not a real training run.

Run from project root:
    python tests/rl_agent/test_train.py
"""

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.rl_agent.config import ExperimentConfig  # noqa: E402
from src.rl_agent.data import GlobalAssetIndex, PricePanel  # noqa: E402
from src.rl_agent.networks import EIIECNN  # noqa: E402
from src.rl_agent.pvm import PortfolioVectorMemory  # noqa: E402
from src.rl_agent.train import (  # noqa: E402
    _batch_tensors,
    _score_holdout,
    agent_forward,
    entropy_schedule,
    load_checkpoint,
    pretrain,
    run_online_backtest,
    sample_batch_starts,
    save_checkpoint,
    saturation_probe,
    train_step,
)
from test_utils import print_check, print_header, print_section_end  # noqa: E402

WINDOW = 5
N_SLOTS = 4
T = 30


def _tiny_cfg():
    d = ExperimentConfig().to_dict()
    d["data"]["window"] = WINDOW
    d["model"]["n_assets"] = N_SLOTS
    d["train"]["batch_size"] = 3
    d["train"]["pretrain_steps"] = 5
    d["train"]["rolling_steps"] = 2
    d["train"]["beta"] = 0.1
    d["train"]["seed"] = 0
    return ExperimentConfig.from_dict(d)


def _tiny_panel(seed=0):
    rng = np.random.default_rng(seed)
    tickers = tuple(f"T{i}" for i in range(N_SLOTS))
    asset_index = GlobalAssetIndex(tickers=tickers, ticker_to_gidx={t: i + 1 for i, t in enumerate(tickers)})
    dates = pd.bdate_range("2020-01-01", periods=T)
    log_r = rng.normal(0.0002, 0.01, size=(T, N_SLOTS))
    prices = 10.0 * np.exp(np.cumsum(log_r, axis=0))
    close = np.column_stack([np.ones(T), prices])
    return PricePanel(
        asset_index=asset_index, dates=dates, close=close, high=close.copy(), low=close.copy(),
        cdi_factor=np.full(T, 1.0003),
        slot_gidx=np.array([[1, 2, 3, 4]] * T), valid=np.array([[True] * N_SLOTS] * T),
        window=WINDOW, start_idx=10, end_idx=T - 1,
    )


def _tiny_model_pvm(cfg, panel):
    torch.manual_seed(cfg.train.seed)
    model = EIIECNN(cfg.data.window, cfg.model.conv1_out_channels, cfg.model.conv2_out_channels,
                     len(cfg.data.features))
    pvm = PortfolioVectorMemory(len(panel.dates), panel.n_global)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.l2)
    return model, pvm, optimizer


def test_sample_batch_starts(passed, failed):
    rng = np.random.default_rng(0)
    starts = sample_batch_starts(t_now=20, n_b=3, beta=0.2, n_batches=100, min_t=5, rng=rng)
    ok = bool(np.all(starts >= 5)) and bool(np.all(starts <= 20 - 3))
    print_check("sample_batch_starts: all draws within [min_t, t_now-n_b]", ok,
                f"range=[{starts.min()}, {starts.max()}]")
    passed, failed = passed + ok, failed + (not ok)

    rng_a, rng_b = np.random.default_rng(42), np.random.default_rng(42)
    a = sample_batch_starts(20, 3, 0.2, 10, 5, rng_a)
    b = sample_batch_starts(20, 3, 0.2, 10, 5, rng_b)
    ok = np.array_equal(a, b)
    print_check("sample_batch_starts: deterministic given the same seeded Generator", ok)
    passed, failed = passed + ok, failed + (not ok)

    try:
        sample_batch_starts(t_now=6, n_b=3, beta=0.2, n_batches=1, min_t=5, rng=rng)
        ok = False
    except ValueError:
        ok = True
    print_check("sample_batch_starts: raises when there isn't enough history", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_store_matches_direct_path(passed, failed):
    """S1 equivalence (TRAINING_SPEEDUP_PLAN.md): the cached-store
    _batch_tensors must be bit-identical to tensors built directly from the
    panel's numpy batch methods -- the one check that fails if the store's
    indexing/offset logic is ever wrong."""
    panel = _tiny_panel()
    features = ("close", "high", "low")
    rng = np.random.default_rng(1)
    ok = True
    for _ in range(3):
        t_b = int(rng.integers(WINDOW - 1, T - 4))
        t_idx = np.arange(t_b, t_b + 3)
        X, y, y_next, slot_gidx, valid, prev_rows, curr_rows = _batch_tensors(panel, t_idx, features, "cpu")
        ok &= torch.equal(X, torch.tensor(panel.window_tensor_batch(t_idx, features), dtype=torch.float32))
        ok &= torch.equal(y, torch.tensor(panel.price_relative_batch(t_idx), dtype=torch.float32))
        ok &= torch.equal(y_next, torch.tensor(panel.price_relative_batch(t_idx + 1), dtype=torch.float32))
        ok &= torch.equal(slot_gidx, torch.tensor(panel.slot_gidx[t_idx], dtype=torch.long))
        ok &= torch.equal(valid, torch.tensor(panel.valid[t_idx], dtype=torch.bool))
        ok &= torch.equal(prev_rows, torch.tensor(t_idx - 1, dtype=torch.long))
        ok &= torch.equal(curr_rows, torch.tensor(t_idx, dtype=torch.long))
    ok = bool(ok)
    print_check("feature store: _batch_tensors bit-identical to the direct numpy path", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_train_step_updates_model_and_pvm(passed, failed):
    cfg = _tiny_cfg()
    panel = _tiny_panel()
    model, pvm, optimizer = _tiny_model_pvm(cfg, panel)
    params_before = [p.clone() for p in model.parameters()]
    pvm_row_before = pvm.buffer[11].clone()

    t_idx = np.arange(10, 13)
    loss = train_step(model, pvm, panel, optimizer, t_idx, cfg.data.features,
                       cfg.costs.c_sell, cfg.costs.c_buy, cfg.costs.train_mu_iters,
                       cfg.train.grad_clip_norm)

    ok = np.isfinite(loss)
    print_check("train_step: returns a finite loss", ok, f"loss={loss}")
    passed, failed = passed + ok, failed + (not ok)

    changed = any(not torch.allclose(a, b) for a, b in zip(params_before, model.parameters()))
    print_check("train_step: model parameters change after one gradient step", changed)
    passed, failed = passed + changed, failed + (not changed)

    pvm_changed = not torch.allclose(pvm_row_before, pvm.buffer[11])
    print_check("train_step: PVM row for the trained period is overwritten", pvm_changed)
    passed, failed = passed + pvm_changed, failed + (not pvm_changed)
    return passed, failed


def test_pretrain(passed, failed):
    cfg = _tiny_cfg()
    panel = _tiny_panel()
    model, pvm, optimizer = _tiny_model_pvm(cfg, panel)

    # tiny panel is far smaller than the default checkpoint_holdout_days -- pretrain must
    # fall back to no checkpointing rather than crash on an impossible holdout carve.
    losses, best_step, best_score = pretrain(model, pvm, panel, optimizer, cfg, train_end_idx=panel.end_idx)
    ok = len(losses) == cfg.train.pretrain_steps
    print_check("pretrain: runs exactly pretrain_steps gradient steps", ok, f"got {len(losses)}")
    passed, failed = passed + ok, failed + (not ok)

    ok = all(np.isfinite(l) for l in losses)
    print_check("pretrain: every step's loss is finite (no divergence in a few steps)", ok, str(losses))
    passed, failed = passed + ok, failed + (not ok)

    ok = best_step is None and best_score is None
    print_check("pretrain: too-small panel for the holdout falls back to no checkpointing", ok,
                f"best_step={best_step}, best_score={best_score}")
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_pretrain_saturation_log(passed, failed):
    cfg = _tiny_cfg()
    panel = _tiny_panel()
    model, pvm, optimizer = _tiny_model_pvm(cfg, panel)
    external_calls = []

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "saturation_probe.json"
        pretrain(model, pvm, panel, optimizer, cfg, train_end_idx=panel.end_idx,
                 on_step=lambda step, loss, model: external_calls.append(step),
                 saturation_log_path=log_path, saturation_every=1)

        ok = log_path.exists()
        print_check("pretrain: saturation_log_path writes a file when given", ok)
        passed, failed = passed + ok, failed + (not ok)

        history = json.loads(log_path.read_text())
        ok = len(history) == cfg.train.pretrain_steps
        print_check("pretrain: saturation history has one entry per step at saturation_every=1", ok,
                    f"got {len(history)}, expected {cfg.train.pretrain_steps}")
        passed, failed = passed + ok, failed + (not ok)

        ok = all({"step", "loss", "max_weight", "entropy"} <= set(h) for h in history)
        print_check("pretrain: each saturation entry has step/loss/max_weight/entropy", ok, str(history[:1]))
        passed, failed = passed + ok, failed + (not ok)

        ok = len(external_calls) == cfg.train.pretrain_steps
        print_check("pretrain: an externally-provided on_step still runs every step (composes, doesn't replace)",
                    ok, f"got {len(external_calls)} calls")
        passed, failed = passed + ok, failed + (not ok)

    # No saturation_log_path given: no file-writing side effect, on_step still fires.
    model2, pvm2, optimizer2 = _tiny_model_pvm(cfg, panel)
    calls2 = []
    pretrain(model2, pvm2, panel, optimizer2, cfg, train_end_idx=panel.end_idx,
             on_step=lambda step, loss, model: calls2.append(step))
    ok = len(calls2) == cfg.train.pretrain_steps
    print_check("pretrain: on_step alone (no saturation_log_path) still works exactly as before", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_saturation_probe_reference_day_fixed(passed, failed):
    panel = _tiny_panel()
    cfg = _tiny_cfg()
    model, pvm, _ = _tiny_model_pvm(cfg, panel)
    model.train()
    probe, history = saturation_probe(panel, cfg.data.features, device="cpu", every=1)

    probe(0, 0.123, model)
    probe(1, 0.456, model)
    ok = len(history) == 2 and history[0]["loss"] == 0.123 and history[1]["loss"] == 0.456
    print_check("saturation_probe: logs one entry per call at every=1, preserving the passed-in loss", ok,
                str(history))
    passed, failed = passed + ok, failed + (not ok)

    ok = model.training  # probe must restore training mode after its internal eval()
    print_check("saturation_probe: restores the model's training mode after probing", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_entropy_schedule(passed, failed):
    ok = entropy_schedule(0, 100, 1e-4, 1e-5, 0.1) == 1e-4
    print_check("entropy_schedule: starts at entropy_beta_start", ok)
    passed, failed = passed + ok, failed + (not ok)

    ok = np.isclose(entropy_schedule(10, 100, 1e-4, 1e-5, 0.1), 1e-5)
    print_check("entropy_schedule: reaches entropy_beta_end exactly at anneal_frac*total_steps", ok,
                f"got {entropy_schedule(10, 100, 1e-4, 1e-5, 0.1)}")
    passed, failed = passed + ok, failed + (not ok)

    ok = np.isclose(entropy_schedule(50, 100, 1e-4, 1e-5, 0.1), 1e-5)
    print_check("entropy_schedule: stays flat at entropy_beta_end past anneal_frac*total_steps", ok)
    passed, failed = passed + ok, failed + (not ok)

    mid = entropy_schedule(5, 100, 1e-4, 1e-5, 0.1)
    ok = 1e-5 < mid < 1e-4
    print_check("entropy_schedule: strictly between start and end mid-anneal", ok, f"got {mid}")
    passed, failed = passed + ok, failed + (not ok)

    ok = entropy_schedule(0, 100, 1e-5, 1e-5, 0.1) == 1e-5
    print_check("entropy_schedule: start==end reproduces the old flat-beta behavior", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def _wide_tiny_panel(seed=0, t=120, n_slots=N_SLOTS):
    """Like _tiny_panel but long enough to carve a checkpoint holdout out of its tail."""
    rng = np.random.default_rng(seed)
    tickers = tuple(f"T{i}" for i in range(n_slots))
    asset_index = GlobalAssetIndex(tickers=tickers, ticker_to_gidx={tk: i + 1 for i, tk in enumerate(tickers)})
    dates = pd.bdate_range("2020-01-01", periods=t)
    log_r = rng.normal(0.0002, 0.01, size=(t, n_slots))
    prices = 10.0 * np.exp(np.cumsum(log_r, axis=0))
    close = np.column_stack([np.ones(t), prices])
    return PricePanel(
        asset_index=asset_index, dates=dates, close=close, high=close.copy(), low=close.copy(),
        cdi_factor=np.full(t, 1.0003),
        slot_gidx=np.array([[1, 2, 3, 4]] * t), valid=np.array([[True] * n_slots] * t),
        window=WINDOW, start_idx=10, end_idx=t - 1,
    )


def test_pretrain_checkpoint_at_peak(passed, failed):
    """checkpoint-at-peak: a holdout that fits inside the panel must actually
    engage, and the restored best checkpoint must reproduce its recorded score."""
    d = ExperimentConfig().to_dict()
    d["data"]["window"] = WINDOW
    d["model"]["n_assets"] = N_SLOTS
    d["train"]["batch_size"] = 3
    d["train"]["pretrain_steps"] = 8
    d["train"]["rolling_steps"] = 2
    d["train"]["beta"] = 0.1
    d["train"]["seed"] = 0
    d["train"]["checkpoint_holdout_days"] = 15
    d["train"]["checkpoint_eval_every"] = 2
    cfg = ExperimentConfig.from_dict(d)
    panel = _wide_tiny_panel()
    model, pvm, optimizer = _tiny_model_pvm(cfg, panel)

    losses, best_step, best_score = pretrain(model, pvm, panel, optimizer, cfg, train_end_idx=panel.end_idx)

    ok = best_step is not None and 0 <= best_step < cfg.train.pretrain_steps
    print_check("pretrain: a holdout that fits the panel records a best_step", ok,
                f"best_step={best_step}")
    passed, failed = passed + ok, failed + (not ok)

    fit_end_idx = panel.end_idx - cfg.train.checkpoint_holdout_days
    replayed_score = _score_holdout(model, pvm, panel, cfg, fit_end_idx, panel.end_idx, "cpu")
    ok = best_score is not None and np.isclose(replayed_score, best_score, atol=1e-9)
    print_check("pretrain: restoring the best checkpoint reproduces its recorded holdout score", ok,
                f"best_score={best_score}, replayed={replayed_score}")
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_agent_forward(passed, failed):
    cfg = _tiny_cfg()
    panel = _tiny_panel()
    model, pvm, _ = _tiny_model_pvm(cfg, panel)

    w = agent_forward(model, pvm, panel, t=15, features=cfg.data.features)
    ok = w.shape == (panel.n_global,)
    print_check("agent_forward: returns a full global-space vector", ok, str(w.shape))
    passed, failed = passed + ok, failed + (not ok)

    ok = np.isclose(w.sum(), 1.0, atol=1e-5) and bool(np.all(w >= -1e-6))
    print_check("agent_forward: global vector sums to 1 and is non-negative", ok, f"sum={w.sum():.6f}")
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_run_online_backtest(passed, failed):
    cfg = _tiny_cfg()
    panel = _tiny_panel()
    model, pvm, optimizer = _tiny_model_pvm(cfg, panel)
    params_before = [p.clone() for p in model.parameters()]

    result = run_online_backtest(model, pvm, panel, optimizer, cfg, start_idx=20, end_idx=24)

    ok = len(result.dates) == 5 and len(result.portfolio_value) == 6
    print_check("run_online_backtest: correct array lengths for a 5-day window", ok)
    passed, failed = passed + ok, failed + (not ok)

    ok = bool(np.all(result.portfolio_value > 0)) and bool(np.all(np.isfinite(result.portfolio_value)))
    print_check("run_online_backtest: portfolio value stays positive and finite", ok,
                str(result.portfolio_value))
    passed, failed = passed + ok, failed + (not ok)

    changed = any(not torch.allclose(a, b) for a, b in zip(params_before, model.parameters()))
    print_check("run_online_backtest: OSBL rolling updates actually change model parameters", changed)
    passed, failed = passed + changed, failed + (not changed)
    return passed, failed


def test_checkpoint_roundtrip(passed, failed):
    cfg = _tiny_cfg()
    panel = _tiny_panel()
    model, pvm, optimizer = _tiny_model_pvm(cfg, panel)
    train_step(model, pvm, panel, optimizer, np.arange(10, 13), cfg.data.features,
               cfg.costs.c_sell, cfg.costs.c_buy, cfg.costs.train_mu_iters, cfg.train.grad_clip_norm)
    model_state_before = copy.deepcopy(model.state_dict())
    pvm_buffer_before = pvm.buffer.clone()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_checkpoint(path, model, optimizer, pvm, step=1)

        model2, pvm2, optimizer2 = _tiny_model_pvm(cfg, panel)  # fresh, different init
        step, extra = load_checkpoint(path, model2, optimizer2, pvm2)

        ok = step == 1
        print_check("checkpoint: step number round-trips", ok)
        passed, failed = passed + ok, failed + (not ok)

        ok = all(torch.allclose(model_state_before[k], model2.state_dict()[k]) for k in model_state_before)
        print_check("checkpoint: model weights round-trip exactly", ok)
        passed, failed = passed + ok, failed + (not ok)

        ok = torch.allclose(pvm_buffer_before, pvm2.buffer)
        print_check("checkpoint: PVM buffer round-trips exactly", ok)
        passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def main():
    print_header("test_train")
    passed = failed = 0

    passed, failed = test_sample_batch_starts(passed, failed)
    passed, failed = test_store_matches_direct_path(passed, failed)
    passed, failed = test_train_step_updates_model_and_pvm(passed, failed)
    passed, failed = test_pretrain(passed, failed)
    passed, failed = test_pretrain_saturation_log(passed, failed)
    passed, failed = test_saturation_probe_reference_day_fixed(passed, failed)
    passed, failed = test_entropy_schedule(passed, failed)
    passed, failed = test_pretrain_checkpoint_at_peak(passed, failed)
    passed, failed = test_agent_forward(passed, failed)
    passed, failed = test_run_online_backtest(passed, failed)
    passed, failed = test_checkpoint_roundtrip(passed, failed)

    print_section_end(passed, failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
