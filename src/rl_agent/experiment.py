"""
experiment.py — CLI orchestrator (docs/EIIE_AGENT_PLAN.md "Implementation
phases", "Reproducibility", "Experiment Validation Checklist"). Ties every
module together into one reproducible run:

    seed everything -> load data -> recompute a window-scoped split
    -> sanity checks (stop if failed) -> pretrain -> online backtest
    (val or test) -> every baseline -> metrics -> HTML report
    -> validation checklist -> everything saved to experiments/{run_id}/

--dry-run stops after sanity checks -- verifies data/config/wiring without
committing to a real (and, per this project's working rules, explicitly
approved) training run.
"""

import argparse
from dataclasses import replace
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .baselines import run_baseline
from .config import ExperimentConfig
from .data import PricePanel, load_price_panel
from .metrics import simple_returns, summarize
from .networks import EIIECNN
from .paths import ROOT
from .plots import write_report
from .pvm import PortfolioVectorMemory
from .sanity import check_policy_not_saturated, run_sanity_checks, seed_everything
from .train import pretrain, run_online_backtest, save_checkpoint


def compute_window_split(panel: PricePanel, train_frac: float = 0.7, val_frac: float = 0.15) -> tuple:
    """Recompute train/val/test cutoffs WITHIN this experiment's date window
    (2011-2026) -- the repo's split_config.json was computed over the full
    2000-2026 dataset and doesn't fit here (docs/EIIE_AGENT_PLAN.md "PVM,
    OSBL training, split protocol")."""
    n = panel.end_idx - panel.start_idx + 1
    train_end_idx = panel.start_idx + int(n * train_frac) - 1
    val_end_idx = panel.start_idx + int(n * (train_frac + val_frac)) - 1
    return train_end_idx, val_end_idx


def _git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT)
        return result.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _dataset_fingerprint() -> dict:
    manifest_path = ROOT / "data/processed/ml_dataset.manifest.json"
    if not manifest_path.exists():
        return {"available": False}
    manifest = json.loads(manifest_path.read_text())
    return {
        "available": True,
        "git_commit": manifest.get("git_commit"),
        "rows": manifest.get("rows"),
        "date_min": manifest.get("date_min"),
        "date_max": manifest.get("date_max"),
    }


def run_experiment(cfg: ExperimentConfig, dry_run: bool = False, eval_split: str = "val",
                    panel: Optional[PricePanel] = None) -> Path:
    """eval_split: 'val' for hyperparameter-selection runs (pretrain on
    train, backtest on val) or 'test' for the final run (pretrain on
    train+val, backtest on test) -- docs/EIIE_AGENT_PLAN.md's "Training /
    evaluation protocol". `panel` can be injected (e.g. by tests) instead
    of loading the real dataset."""
    if eval_split not in ("val", "test"):
        raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split!r}")

    seed_everything(cfg.train.seed)
    if cfg.train.device == "cpu":
        # Measured: this network's tensors are tiny (50x50), so torch's default
        # thread count (one per core) spends more time on inter-thread sync than
        # the ops themselves save -- 2 threads measured ~27% faster end-to-end
        # than 14 on the dev machine. Doesn't affect results, only wall-clock.
        torch.set_num_threads(2)

    # %f: parallel sweep launches (sweep.py, S5) in the same second must not collide
    out_dir = ROOT / cfg.experiment.out_dir / f"{cfg.experiment.name}_{datetime.now():%Y%m%dT%H%M%S%f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")

    panel = load_price_panel(cfg.data, n_slots=cfg.model.n_assets) if panel is None else panel
    train_end_idx, val_end_idx = compute_window_split(panel)

    checklist = {
        "sanity_passed": False, "no_numerical_instability": False,
        "config_saved": True, "seed_logged": True, "dataset_version_logged": False,
        "model_version_logged": False, "metrics_generated": False,
        "baseline_comparison_generated": False, "visualizations_generated": False,
        "policy_not_saturated": False,
    }

    dataset_info = _dataset_fingerprint()
    checklist["dataset_version_logged"] = dataset_info.get("available", False)
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "git_commit": _git_commit(),
        "seed": cfg.train.seed,
        "eval_split": eval_split,
        "dataset": dataset_info,
        "versions": {"torch": torch.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "python": platform.python_version()},
        "asset_index": {"n_global": panel.n_global, "tickers": list(panel.asset_index.tickers)},
        "split": {
            "train_end_idx": train_end_idx, "val_end_idx": val_end_idx,
            "train_end_date": str(panel.dates[train_end_idx].date()),
            "val_end_date": str(panel.dates[val_end_idx].date()),
        },
    }, indent=2))

    model = EIIECNN(cfg.data.window, cfg.model.conv1_out_channels, cfg.model.conv2_out_channels,
                     len(cfg.data.features)).to(cfg.train.device)
    # S3: train through the compiled wrapper (shares parameters with `model`);
    # checkpoints keep saving the eager module so state_dict keys stay prefix-free.
    train_model = torch.compile(model, mode="reduce-overhead") if cfg.train.compile else model
    pvm = PortfolioVectorMemory(len(panel.dates), panel.n_global, slot_gidx=panel.slot_gidx,
                                valid=panel.valid, device=cfg.train.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.l2)

    sanity_report = run_sanity_checks(cfg, panel, device=cfg.train.device)
    (out_dir / "sanity_report.txt").write_text(str(sanity_report))
    checklist["sanity_passed"] = sanity_report.passed
    print(sanity_report)

    if not sanity_report.passed:
        (out_dir / "report.json").write_text(json.dumps({"checklist": checklist, "valid": False}, indent=2))
        raise RuntimeError(f"Sanity checks failed -- see {out_dir / 'sanity_report.txt'}")

    if dry_run:
        print(f"--dry-run: stopping after sanity checks. Artifacts in {out_dir}")
        return out_dir

    if eval_split == "test":
        pretrain_end_idx, backtest_start_idx, backtest_end_idx = val_end_idx, val_end_idx + 1, panel.end_idx
    else:
        pretrain_end_idx, backtest_start_idx, backtest_end_idx = train_end_idx, train_end_idx + 1, val_end_idx

    train_losses, best_step, best_holdout_score = pretrain(
        train_model, pvm, panel, optimizer, cfg, train_end_idx=pretrain_end_idx, device=cfg.train.device)
    checklist["no_numerical_instability"] = bool(np.all(np.isfinite(train_losses)))

    # Post-pretrain: did the policy collapse into a saturated (gradient-dead) corner?
    # Loud, because a collapsed policy makes every downstream number meaningless -- the
    # backtest still "runs", it just reports the frozen all-cash portfolio as if it were
    # a decision. Not fatal: the report is still written so the failure is inspectable.
    pol_ok, pol_msg = check_policy_not_saturated(model, panel, cfg, pretrain_end_idx + 1,
                                                  device=cfg.train.device)
    checklist["policy_not_saturated"] = pol_ok
    print(f"[{'OK  ' if pol_ok else 'WARN'}] policy_not_saturated -- {pol_msg}")

    agent_result = run_online_backtest(train_model, pvm, panel, optimizer, cfg,
                                        start_idx=backtest_start_idx, end_idx=backtest_end_idx,
                                        device=cfg.train.device)
    checklist["no_numerical_instability"] &= bool(np.all(np.isfinite(agent_result.portfolio_value)))

    save_checkpoint(out_dir / "model.pt", model, optimizer, pvm, step=len(train_losses),
                     extra={"eval_split": eval_split})
    checklist["model_version_logged"] = True

    if cfg.data.cash_mode == "cdi":
        risk_free = panel.cdi_factor[backtest_start_idx:backtest_end_idx + 1] - 1.0
    else:
        risk_free = np.zeros(backtest_end_idx - backtest_start_idx + 1)

    benchmark_result = run_baseline("bova11", panel, cfg.costs.c_sell, cfg.costs.c_buy,
                                     start_idx=backtest_start_idx, end_idx=backtest_end_idx)
    benchmark_returns = simple_returns(benchmark_result.log_returns)

    agent_summary = summarize(agent_result, risk_free, benchmark_returns, cfg.eval.var_level,
                               cfg.eval.bootstrap_n, cfg.eval.bootstrap_block, seed=cfg.train.seed)
    checklist["metrics_generated"] = True

    baseline_results, baseline_summaries = {}, {}
    for name in cfg.eval.baselines:
        result = run_baseline(name, panel, cfg.costs.c_sell, cfg.costs.c_buy, seed=cfg.train.seed,
                               start_idx=backtest_start_idx, end_idx=backtest_end_idx)
        baseline_results[name] = result
        baseline_summaries[name] = summarize(result, risk_free, benchmark_returns, cfg.eval.var_level,
                                              cfg.eval.bootstrap_n, cfg.eval.bootstrap_block, seed=cfg.train.seed)
    checklist["baseline_comparison_generated"] = True

    write_report(out_dir / "report.html", agent_result, agent_summary, baseline_results, baseline_summaries,
                 train_losses, panel.asset_index, eval_log_returns=agent_result.log_returns)
    checklist["visualizations_generated"] = True

    (out_dir / "metrics_summary.json").write_text(json.dumps({
        "agent": asdict(agent_summary),
        "baselines": {name: asdict(s) for name, s in baseline_summaries.items()},
    }, indent=2, default=str))

    (out_dir / "report.json").write_text(json.dumps({
        "checklist": checklist, "valid": all(checklist.values()),
        "pretrain_checkpoint": {"best_step": best_step, "best_holdout_return": best_holdout_score},
    }, indent=2))
    if not all(checklist.values()):
        failed = [k for k, v in checklist.items() if not v]
        print(f"WARNING: experiment validation checklist incomplete: {failed}")
    print(f"Experiment complete. Artifacts in {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Run an EIIE portfolio-management agent experiment")
    parser.add_argument("--config", required=True, help="path to an ExperimentConfig JSON")
    parser.add_argument("--dry-run", action="store_true", help="stop after sanity checks, no training")
    parser.add_argument("--eval-split", choices=["val", "test"], default="val",
                         help="'val' for hyperparameter runs, 'test' for the final run")
    parser.add_argument("--seed", type=int, help="override random seed (for seed ensemble)")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_json(args.config)
    if args.seed is not None:
        cfg = replace(cfg, train=replace(cfg.train, seed=args.seed))
    run_experiment(cfg, dry_run=args.dry_run, eval_split=args.eval_split)


if __name__ == "__main__":
    main()
