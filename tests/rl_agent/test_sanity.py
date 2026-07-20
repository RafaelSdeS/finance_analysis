"""
Test: sanity.py's pre-training invariant gate (docs/eiie_agent/EIIE_AGENT_PLAN.md
Phase 6). Synthetic data only.

Run from project root:
    python tests/rl_agent/test_sanity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.rl_agent.config import ExperimentConfig  # noqa: E402
from src.rl_agent.data import GlobalAssetIndex, PricePanel  # noqa: E402
from src.rl_agent.sanity import run_sanity_checks  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402

WINDOW = 5
N_SLOTS = 4
T = 30


def _tiny_cfg():
    d = ExperimentConfig().to_dict()
    d["data"]["window"] = WINDOW
    d["model"]["n_assets"] = N_SLOTS
    d["train"]["batch_size"] = 3
    d["train"]["seed"] = 0
    return ExperimentConfig.from_dict(d)


def _tiny_panel(seed=0, T=T, start_idx=10, end_idx=None):
    end_idx = (T - 1) if end_idx is None else end_idx
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
        window=WINDOW, start_idx=start_idx, end_idx=end_idx,
    )


def test_sanity_passes_on_a_healthy_setup(passed, failed):
    cfg = _tiny_cfg()
    panel = _tiny_panel()
    report = run_sanity_checks(cfg, panel, n_steps=3)

    print(str(report))
    ok = report.passed
    print_check("run_sanity_checks: every gate passes on a healthy synthetic setup", ok)
    passed, failed = passed + ok, failed + (not ok)

    expected_checks = {
        "deterministic_seeding", "weights_on_simplex", "weights_finite",
        "finite_gradients_first_batch", "finite_loss", "pvm_finite",
        "baselines_run_cleanly", "zero_cost_no_drag", "real_cost_reduces_value",
    }
    ok = expected_checks.issubset(report.checks.keys())
    print_check("run_sanity_checks: all expected gates ran", ok, str(sorted(report.checks.keys())))
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_sanity_flags_insufficient_history(passed, failed):
    cfg = _tiny_cfg()
    tiny_panel = _tiny_panel(T=15, start_idx=12, end_idx=14)  # far too little room for batch_size + n_steps
    report = run_sanity_checks(cfg, tiny_panel, n_steps=3)

    ok = not report.passed
    print_check("run_sanity_checks: reports failure when there isn't enough history", ok)
    passed, failed = passed + ok, failed + (not ok)

    ok = "enough_history_for_checks" in report.checks and not report.checks["enough_history_for_checks"][0]
    print_check("run_sanity_checks: names the specific failing gate", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def main():
    print_header("test_sanity")
    passed = failed = 0

    passed, failed = test_sanity_passes_on_a_healthy_setup(passed, failed)
    passed, failed = test_sanity_flags_insufficient_history(passed, failed)

    print_section_end(passed, failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
