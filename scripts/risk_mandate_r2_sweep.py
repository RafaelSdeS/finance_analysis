#!/usr/bin/env python
"""
R2 sensitivity grid for the risk/diversification mandate (RISK_MANDATE_IMPL_PLAN.md
R2, results in R2_FINDINGS.md): lookback x rebalance_every, val split, all 4 risk
policies + UCRP/UBAH reference rows. Panel/split/benchmark loaded once and reused
across the whole grid instead of paying risk_experiment.py's per-run I/O 9x.

Usage:
    python scripts/risk_mandate_r2_sweep.py > /tmp/r2_results.csv
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

LOOKBACKS = (63, 126, 252)
REBALS = (1, 5, 21)  # daily / weekly / monthly
FIELDS = ["sharpe", "calmar", "max_drawdown", "annualized_return", "annualized_turnover",
          "transaction_cost_drag"]


def main():
    cfg = ExperimentConfig.from_json(ROOT / "configs" / "risk_mandate.json")
    panel = load_price_panel(cfg.data, n_slots=cfg.model.n_assets)
    train_end_idx, val_end_idx = compute_window_split(panel)
    start_idx, end_idx = train_end_idx + 1, val_end_idx

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
        rows.append((name, None, None, _summ(r)))

    for lookback in LOOKBACKS:
        for rebal in REBALS:
            risk_cfg = replace(cfg.risk, lookback=lookback, rebalance_every=rebal)
            for policy in RISK_POLICY_NAMES:
                weight_fn = make_risk_weight_fn(policy, risk_cfg, panel, start_idx)
                result = run_backtest(panel, weight_fn, cfg.costs.c_sell, cfg.costs.c_buy,
                                       start_idx, end_idx, cfg.costs.backtest_mu_tol)
                rows.append((policy, lookback, rebal, _summ(result)))
            print(f"done: lookback={lookback} rebal={rebal}", file=sys.stderr)

    print("policy,lookback,rebalance_every," + ",".join(FIELDS))
    for name, lb, rb, s in rows:
        vals = ",".join(f"{getattr(s, f):.4f}" if np.isfinite(getattr(s, f)) else "nan" for f in FIELDS)
        print(f"{name},{lb if lb else ''},{rb if rb else ''},{vals}")


if __name__ == "__main__":
    main()
