#!/usr/bin/env python
"""
R3 disjoint test-split confirmation + cost-rate stress for the risk mandate
(RISK_MANDATE_IMPL_PLAN.md R3, results in R3_FINDINGS.md). Reuses one loaded
panel across three eval_split="test" backtest configs:
  (a) R1's original default (lookback=126, rebal=21)
  (b) R2's selected candidate (lookback=63, rebal=1) -- picked on val, confirmed
      here on test, never re-picked after seeing test numbers
  (c) (b) again with transaction costs doubled (cost-rate stress)

Usage:
    python scripts/risk_mandate_r3_test_split.py > /tmp/r3_results.csv
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl_agent.baselines import run_baseline  # noqa: E402
from src.rl_agent.config import ExperimentConfig  # noqa: E402
from src.rl_agent.data import load_price_panel  # noqa: E402
from src.rl_agent.environment import run_backtest  # noqa: E402
from src.rl_agent.experiment import compute_window_split  # noqa: E402
from src.rl_agent.metrics import simple_returns, summarize  # noqa: E402
from src.rl_agent.risk_portfolios import RISK_POLICY_NAMES, make_risk_weight_fn  # noqa: E402

FIELDS = ["sharpe", "calmar", "max_drawdown", "annualized_return", "annualized_turnover",
          "transaction_cost_drag"]


def run_grid_cell(panel, start_idx, end_idx, cfg, label):
    risk_free = panel.cdi_factor[start_idx:end_idx + 1] - 1.0
    benchmark = run_baseline("bova11", panel, cfg.costs.c_sell, cfg.costs.c_buy,
                              start_idx=start_idx, end_idx=end_idx)
    benchmark_returns = simple_returns(benchmark.log_returns)

    def _summ(result):
        return summarize(result, risk_free, benchmark_returns, cfg.eval.var_level,
                          cfg.eval.bootstrap_n, cfg.eval.bootstrap_block, seed=cfg.train.seed)

    rows = []
    for name in ("ucrp", "ubah"):
        r = run_baseline(name, panel, cfg.costs.c_sell, cfg.costs.c_buy, seed=cfg.train.seed,
                          start_idx=start_idx, end_idx=end_idx)
        rows.append((label, name, _summ(r)))
    rows.append((label, "bova11", _summ(benchmark)))

    for policy in RISK_POLICY_NAMES:
        weight_fn = make_risk_weight_fn(policy, cfg.risk, panel, start_idx)
        result = run_backtest(panel, weight_fn, cfg.costs.c_sell, cfg.costs.c_buy,
                               start_idx, end_idx, cfg.costs.backtest_mu_tol)
        rows.append((label, policy, _summ(result)))
    return rows


def main():
    base_cfg = ExperimentConfig.from_json(ROOT / "configs" / "risk_mandate.json")
    panel = load_price_panel(base_cfg.data, n_slots=base_cfg.model.n_assets)
    _, val_end_idx = compute_window_split(panel)
    start_idx, end_idx = val_end_idx + 1, panel.end_idx

    print(f"test split: {panel.dates[start_idx].date()} -> {panel.dates[end_idx].date()} "
          f"({end_idx - start_idx + 1} days)", file=sys.stderr)

    configs = {
        "default_lb126_rb21": base_cfg,
        "r2_selected_lb63_rb1": replace(base_cfg, risk=replace(base_cfg.risk, lookback=63, rebalance_every=1)),
        "r2_selected_2x_costs": replace(
            base_cfg,
            risk=replace(base_cfg.risk, lookback=63, rebalance_every=1),
            costs=replace(base_cfg.costs, c_sell=0.0006, c_buy=0.0006)),
    }

    all_rows = []
    for label, cfg in configs.items():
        all_rows.extend(run_grid_cell(panel, start_idx, end_idx, cfg, label))
        print(f"done: {label}", file=sys.stderr)

    print("config,policy," + ",".join(FIELDS))
    for label, name, s in all_rows:
        vals = ",".join(f"{getattr(s, f):.4f}" if np.isfinite(getattr(s, f)) else "nan" for f in FIELDS)
        print(f"{label},{name},{vals}")


if __name__ == "__main__":
    main()
