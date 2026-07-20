"""
Test: plots.py's self-contained HTML report generation
(docs/eiie_agent/EIIE_AGENT_PLAN.md Phase 7). Synthetic data only -- checks the report
is produced, is self-contained, and includes every required chart, not the
pixel-level rendering of any individual chart.

Run from project root:
    python tests/rl_agent/test_plots.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.rl_agent import metrics as m  # noqa: E402
from src.rl_agent.data import GlobalAssetIndex  # noqa: E402
from src.rl_agent.environment import BacktestResult  # noqa: E402
from src.rl_agent.plots import write_report  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402


def _synthetic_result(seed, T=60, n_global=5):
    rng = np.random.default_rng(seed)
    log_r = np.log(1 + rng.normal(0.0003, 0.01, size=T))
    pv = np.concatenate([[1.0], np.exp(np.cumsum(log_r))])
    mu = np.clip(1 - np.abs(rng.normal(0.0005, 0.0003, size=T)), 0.9, 1.0)
    turnover = np.abs(rng.normal(0.05, 0.02, size=T))
    weights = rng.dirichlet(np.ones(n_global), size=T)
    return BacktestResult(
        dates=pd.bdate_range("2020-01-01", periods=T),
        portfolio_value=pv, log_returns=log_r, mu=mu, turnover=turnover,
        cost=1.0 - mu, weights=weights,
    )


def test_write_report_smoke(passed, failed):
    asset_index = GlobalAssetIndex(tickers=("AAA", "BBB", "CCC", "DDD"),
                                    ticker_to_gidx={"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4})
    agent_result = _synthetic_result(seed=0)
    baseline_results = {"ucrp": _synthetic_result(seed=1), "ubah": _synthetic_result(seed=2)}

    rf = np.zeros(60)
    bench = np.random.default_rng(3).normal(0.0002, 0.008, size=60)
    agent_summary = m.summarize(agent_result, rf, bench, bootstrap_n=20, seed=0)
    baseline_summaries = {name: m.summarize(r, rf, bench, bootstrap_n=20, seed=0)
                           for name, r in baseline_results.items()}
    train_losses = list(np.random.default_rng(4).normal(-0.001, 0.0005, size=50))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.html"
        write_report(path, agent_result, agent_summary, baseline_results, baseline_summaries,
                     train_losses, asset_index, eval_log_returns=agent_result.log_returns)

        ok = path.exists() and path.stat().st_size > 10_000
        print_check("write_report: produces a non-trivial HTML file", ok,
                    f"size={path.stat().st_size if path.exists() else 0} bytes")
        passed, failed = passed + ok, failed + (not ok)

        html = path.read_text()
        ok = html.count("Plotly.newPlot") >= 6
        print_check("write_report: all 6 charts are embedded in one file", ok,
                    f"found {html.count('Plotly.newPlot')} plot calls")
        passed, failed = passed + ok, failed + (not ok)

        ok = html.count("var Plotly") <= 1
        print_check("write_report: plotly.js library is embedded exactly once (self-contained)", ok)
        passed, failed = passed + ok, failed + (not ok)

        for expected_title in ["Portfolio Value", "Reward Curve", "Allocation Evolution",
                                "Transaction Costs", "Position-Size Distribution", "Performance Metrics"]:
            ok = expected_title in html
            print_check(f"write_report: includes the '{expected_title}' chart", ok)
            passed, failed = passed + ok, failed + (not ok)

        ok = "AAA" in html or "BBB" in html
        print_check("write_report: allocation chart references real ticker names", ok)
        passed, failed = passed + ok, failed + (not ok)

        ok = "ucrp" in html and "ubah" in html
        print_check("write_report: both baselines appear in the report", ok)
        passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def main():
    print_header("test_plots")
    passed = failed = 0

    passed, failed = test_write_report_smoke(passed, failed)

    print_section_end(passed, failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
