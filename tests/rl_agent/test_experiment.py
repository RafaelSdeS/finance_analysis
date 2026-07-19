"""
Test: experiment.py's CLI orchestrator -- --dry-run, a full synthetic run,
the validation checklist, and the sanity-failure path
(docs/EIIE_AGENT_PLAN.md Phase 7). Runs entirely on a tiny, dependency-
injected synthetic panel (never the real 15-year dataset, never a real
training run) via run_experiment(..., panel=...).

Run from project root:
    python tests/rl_agent/test_experiment.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.rl_agent.config import ExperimentConfig  # noqa: E402
from src.rl_agent.data import GlobalAssetIndex, PricePanel  # noqa: E402
from src.rl_agent.experiment import compute_window_split, run_experiment  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402

WINDOW = 5
N_SLOTS = 4
T = 60


def _tiny_cfg(out_dir: str):
    d = ExperimentConfig().to_dict()
    d["data"]["window"] = WINDOW
    d["model"]["n_assets"] = N_SLOTS
    d["train"]["batch_size"] = 3
    d["train"]["pretrain_steps"] = 3
    d["train"]["rolling_steps"] = 1
    d["train"]["beta"] = 0.1
    d["train"]["seed"] = 0
    d["eval"]["bootstrap_n"] = 20
    d["eval"]["baselines"] = ["ubah", "ucrp", "constant_cash", "bova11"]
    d["experiment"]["out_dir"] = out_dir
    d["experiment"]["name"] = "test_run"
    return ExperimentConfig.from_dict(d)


def _tiny_panel(seed=0, start_idx=10, end_idx=T - 1):
    rng = np.random.default_rng(seed)
    tickers = tuple(f"T{i}" for i in range(N_SLOTS))
    asset_index = GlobalAssetIndex(tickers=tickers, ticker_to_gidx={t: i + 1 for i, t in enumerate(tickers)})
    dates = pd.bdate_range("2020-01-01", periods=T)
    log_r = rng.normal(0.0002, 0.01, size=(T, N_SLOTS))
    prices = 10.0 * np.exp(np.cumsum(log_r, axis=0))
    close = np.column_stack([np.ones(T), prices])
    bova11 = 100.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.008, size=T)))
    return PricePanel(
        asset_index=asset_index, dates=dates, close=close, high=close.copy(), low=close.copy(),
        cdi_factor=np.full(T, 1.0003),
        slot_gidx=np.array([[1, 2, 3, 4]] * T), valid=np.array([[True] * N_SLOTS] * T),
        window=WINDOW, start_idx=start_idx, end_idx=end_idx,
        bova11_close=bova11,
    )


def test_compute_window_split(passed, failed):
    panel = _tiny_panel()
    train_end_idx, val_end_idx = compute_window_split(panel)
    n = panel.end_idx - panel.start_idx + 1
    ok = train_end_idx == panel.start_idx + int(n * 0.7) - 1
    print_check("compute_window_split: train_end_idx matches hand-computed formula", ok,
                f"got {train_end_idx}")
    passed, failed = passed + ok, failed + (not ok)

    ok = panel.start_idx <= train_end_idx < val_end_idx <= panel.end_idx
    print_check("compute_window_split: cutoffs are ordered and within the panel's window", ok,
                f"[{panel.start_idx}, {train_end_idx}, {val_end_idx}, {panel.end_idx}]")
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_dry_run(passed, failed):
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tiny_cfg(tmp)
        panel = _tiny_panel()
        out_dir = run_experiment(cfg, dry_run=True, panel=panel)

        ok = (out_dir / "config.json").exists() and (out_dir / "sanity_report.txt").exists()
        print_check("dry-run: config and sanity report saved", ok)
        passed, failed = passed + ok, failed + (not ok)

        ok = not (out_dir / "model.pt").exists() and not (out_dir / "report.html").exists()
        print_check("dry-run: stops before training -- no checkpoint or report produced", ok)
        passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_full_run_val_split(passed, failed):
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tiny_cfg(tmp)
        panel = _tiny_panel()
        out_dir = run_experiment(cfg, dry_run=False, eval_split="val", panel=panel)

        expected_files = ["config.json", "run_manifest.json", "sanity_report.txt",
                           "model.pt", "report.html", "metrics_summary.json", "report.json",
                           "train_metrics.json", "saturation_probe.json"]
        missing = [f for f in expected_files if not (out_dir / f).exists()]
        ok = len(missing) == 0
        print_check("full run: every required artifact is produced", ok, f"missing={missing}")
        passed, failed = passed + ok, failed + (not ok)

        report = json.loads((out_dir / "report.json").read_text())
        ok = report["valid"] and all(report["checklist"].values())
        print_check("full run: validation checklist is entirely true", ok, str(report["checklist"]))
        passed, failed = passed + ok, failed + (not ok)

        train_metrics = json.loads((out_dir / "train_metrics.json").read_text())
        ok = (np.isfinite(train_metrics["total_return"]) and train_metrics["n_days"] > 0
              and set(train_metrics["ranking_quality"].keys()) == {"1", "5", "21"})
        print_check("full run: train_metrics.json has a finite return and ranking_quality at k=1/5/21",
                    ok, str(train_metrics))
        passed, failed = passed + ok, failed + (not ok)

        # The train-window diagnostic snapshots/restores the PVM buffer around itself
        # (see experiment.py) -- if that restore were missing or buggy, the subsequent
        # online backtest's portfolio value would very likely go non-finite, or the
        # checklist above would already have failed. Explicit second look specifically
        # at the seam that snapshot/restore is designed to protect.
        summary_check = json.loads((out_dir / "metrics_summary.json").read_text())
        ok = np.isfinite(summary_check["agent"]["total_return"])
        print_check("full run: agent's val-window return is finite (PVM seam wasn't perturbed by the "
                    "train-window diagnostic)", ok, str(summary_check["agent"]["total_return"]))
        passed, failed = passed + ok, failed + (not ok)

        summary = json.loads((out_dir / "metrics_summary.json").read_text())
        ok = "agent" in summary and set(summary["baselines"].keys()) == set(cfg.eval.baselines)
        print_check("full run: metrics_summary.json has the agent plus every configured baseline",
                    ok, str(list(summary["baselines"].keys())))
        passed, failed = passed + ok, failed + (not ok)

        manifest = json.loads((out_dir / "run_manifest.json").read_text())
        ok = manifest["seed"] == cfg.train.seed and manifest["eval_split"] == "val"
        print_check("full run: run_manifest.json logs seed and eval_split", ok)
        passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_full_run_test_split(passed, failed):
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tiny_cfg(tmp)
        panel = _tiny_panel()
        out_dir = run_experiment(cfg, dry_run=False, eval_split="test", panel=panel)
        manifest = json.loads((out_dir / "run_manifest.json").read_text())
        val_end_idx = manifest["split"]["val_end_idx"]

        report_json = json.loads((out_dir / "report.json").read_text())
        ok = report_json["valid"]
        print_check("eval_split='test': full run completes and validates cleanly", ok)
        passed, failed = passed + ok, failed + (not ok)

        summary = json.loads((out_dir / "metrics_summary.json").read_text())
        ok = summary["agent"]["final_apv"] > 0 and np.isfinite(summary["agent"]["final_apv"])
        print_check("eval_split='test': agent backtests over [val_end+1, panel.end] and produces a valid APV",
                    ok, f"val_end_idx={val_end_idx}, final_apv={summary['agent']['final_apv']}")
        passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_sanity_failure_path(passed, failed):
    """A panel with far too little history should fail sanity and raise,
    writing an invalid report.json rather than silently proceeding."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tiny_cfg(tmp)
        tiny_panel = _tiny_panel(start_idx=12, end_idx=14)  # nowhere near enough room for batch_size+n_steps

        raised = False
        try:
            run_experiment(cfg, dry_run=False, panel=tiny_panel)
        except RuntimeError:
            raised = True
        print_check("insufficient history: run_experiment raises RuntimeError instead of proceeding", raised)
        passed, failed = passed + raised, failed + (not raised)

        out_dirs = list((ROOT / tmp).glob(f"{cfg.experiment.name}_*"))
        ok = len(out_dirs) == 1
        if ok:
            report = json.loads((out_dirs[0] / "report.json").read_text())
            ok = not report["valid"] and not report["checklist"]["sanity_passed"]
        print_check("insufficient history: report.json records valid=False, sanity_passed=False", ok)
        passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def main():
    print_header("test_experiment")
    passed = failed = 0

    passed, failed = test_compute_window_split(passed, failed)
    passed, failed = test_dry_run(passed, failed)
    passed, failed = test_full_run_val_split(passed, failed)
    passed, failed = test_full_run_test_split(passed, failed)
    passed, failed = test_sanity_failure_path(passed, failed)

    print_section_end(passed, failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
