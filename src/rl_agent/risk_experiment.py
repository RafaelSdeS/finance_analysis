"""
risk_experiment.py — CLI orchestrator for the risk/diversification mandate
(RISK_MANDATE_PLAN.md, RISK_MANDATE_IMPL_PLAN.md). Clone of experiment.py's
post-training half: seed -> load data -> recompute a window-scoped split ->
every risk policy + every baseline -> metrics -> HTML report. No pretrain,
no PVM, no network -- risk policies are pure numpy/scipy/sklearn weight_fn's,
identical in kind to baselines.py's.

--dry-run stops after loading the panel and reporting per-rebalance
eligibility stats, without running any backtest.
"""

import argparse
import json
import os
import platform
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .baselines import run_baseline
from .config import ExperimentConfig
from .data import PricePanel, load_price_panel
from .environment import run_backtest
from .experiment import _dataset_fingerprint, _git_commit, compute_window_split
from .metrics import simple_returns, summarize
from .paths import ROOT
from .plots import write_report
from .risk_portfolios import _active_gidx, eligible_mask, make_risk_weight_fn


def _eligibility_report(panel: PricePanel, cfg: ExperimentConfig, start_idx: int, end_idx: int) -> dict:
    """--dry-run diagnostic: at each scheduled rebalance date, how many of
    today's active names actually clear min_history_frac coverage."""
    counts = []
    for t in range(start_idx, end_idx + 1, cfg.risk.rebalance_every):
        active = _active_gidx(t, panel)
        if t - cfg.risk.lookback + 1 < 1:
            continue
        counts.append(int(eligible_mask(panel, t, cfg.risk.lookback, active, cfg.risk.min_history_frac).sum()))
    counts = np.array(counts) if counts else np.array([0])
    return {"n_rebalances": len(counts), "min_eligible": int(counts.min()),
            "max_eligible": int(counts.max()), "mean_eligible": float(counts.mean())}


def run_risk_experiment(cfg: ExperimentConfig, dry_run: bool = False, eval_split: str = "val",
                         panel: Optional[PricePanel] = None) -> Path:
    """eval_split: 'val' (backtest on val split) or 'test' (backtest on
    test split) -- same window-scoped split as experiment.py."""
    if eval_split not in ("val", "test"):
        raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split!r}")

    out_dir = ROOT / cfg.experiment.out_dir / f"{cfg.experiment.name}_{datetime.now():%Y%m%dT%H%M%S%f}_{os.getpid()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")

    panel = load_price_panel(cfg.data, n_slots=cfg.model.n_assets) if panel is None else panel
    train_end_idx, val_end_idx = compute_window_split(panel)
    backtest_start_idx, backtest_end_idx = (
        (val_end_idx + 1, panel.end_idx) if eval_split == "test" else (train_end_idx + 1, val_end_idx)
    )

    (out_dir / "run_manifest.json").write_text(json.dumps({
        "git_commit": _git_commit(), "seed": cfg.train.seed, "eval_split": eval_split,
        "dataset": _dataset_fingerprint(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "python": platform.python_version()},
        "asset_index": {"n_global": panel.n_global, "tickers": list(panel.asset_index.tickers)},
        "split": {"backtest_start_date": str(panel.dates[backtest_start_idx].date()),
                  "backtest_end_date": str(panel.dates[backtest_end_idx].date())},
    }, indent=2))

    eligibility = _eligibility_report(panel, cfg, backtest_start_idx, backtest_end_idx)
    (out_dir / "eligibility_report.json").write_text(json.dumps(eligibility, indent=2))

    if dry_run:
        print(f"--dry-run: stopping after data load. Eligibility: {eligibility}. Artifacts in {out_dir}")
        return out_dir

    if cfg.data.cash_mode == "cdi":
        risk_free = panel.cdi_factor[backtest_start_idx:backtest_end_idx + 1] - 1.0
    else:
        risk_free = np.zeros(backtest_end_idx - backtest_start_idx + 1)

    benchmark_result = run_baseline("bova11", panel, cfg.costs.c_sell, cfg.costs.c_buy,
                                     start_idx=backtest_start_idx, end_idx=backtest_end_idx)
    benchmark_returns = simple_returns(benchmark_result.log_returns)

    def _run(name: str):
        if name in cfg.risk.policies:
            weight_fn = make_risk_weight_fn(name, cfg.risk, panel, backtest_start_idx)
            result = run_backtest(panel, weight_fn, cfg.costs.c_sell, cfg.costs.c_buy,
                                   backtest_start_idx, backtest_end_idx, cfg.costs.backtest_mu_tol)
        else:
            result = run_baseline(name, panel, cfg.costs.c_sell, cfg.costs.c_buy, seed=cfg.train.seed,
                                   start_idx=backtest_start_idx, end_idx=backtest_end_idx)
        summary = summarize(result, risk_free, benchmark_returns, cfg.eval.var_level,
                             cfg.eval.bootstrap_n, cfg.eval.bootstrap_block, seed=cfg.train.seed)
        return result, summary

    focal_name = cfg.risk.policies[0]
    focal_result, focal_summary = _run(focal_name)

    other_results, other_summaries = {}, {}
    for name in list(cfg.risk.policies[1:]) + list(cfg.eval.baselines):
        other_results[name], other_summaries[name] = _run(name)

    write_report(out_dir / "report.html", focal_result, focal_summary, other_results, other_summaries,
                 train_losses=[], asset_index=panel.asset_index, eval_log_returns=focal_result.log_returns,
                 agent_name=focal_name, title="Risk Mandate -- Experiment Report")

    (out_dir / "metrics_summary.json").write_text(json.dumps({
        "focal_policy": focal_name,
        "focal": asdict(focal_summary),
        "others": {name: asdict(s) for name, s in other_summaries.items()},
    }, indent=2, default=str))

    print(f"Risk-mandate experiment complete. Artifacts in {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Run a risk/diversification-mandate backtest")
    parser.add_argument("--config", required=True, help="path to an ExperimentConfig JSON")
    parser.add_argument("--dry-run", action="store_true", help="data load + eligibility report only")
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_json(args.config)
    run_risk_experiment(cfg, dry_run=args.dry_run, eval_split=args.eval_split)


if __name__ == "__main__":
    main()
