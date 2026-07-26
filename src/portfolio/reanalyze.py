"""
reanalyze.py -- metrics-only re-analysis of a saved backtest run (plan
V.0a): recompute deflated Sharpe on the active-vs-EW / excess-CDI series,
sliced to any date cutoff, without paying for a fresh walk-forward
retrain. Built for a concrete V.1a follow-up: the 2026-07-26 top_n sweep
showed active-return-vs-EW flips sign pre/post-2011 depending on universe
size, so the full-sample DSR (which blends both eras) may be understating
-- or overstating -- what the post-2011 era (the one the model actually
has fundamentals for) looks like on its own.

Run: python -m src.portfolio.reanalyze [--run PATH] [--since 2011-01-31] [--n-trials 16]
(default --run: the most recently saved run)
"""

import argparse
from pathlib import Path

import pandas as pd

from src.portfolio import artifacts
from src.portfolio.metrics import deflated_sharpe_ratio


def dsr_by_era(series: pd.Series, cutoff: pd.Timestamp, n_trials: int) -> dict:
    """{"full": {...}, "since_cutoff": {...}}, each an ann_mean/PSR@1/DSR@n_trials
    dict (or None if too few observations in that slice) -- the pure,
    testable piece; main() just prints this."""
    out = {}
    for tag, s in [("full", series), ("since_cutoff", series[series.index >= cutoff])]:
        if len(s) < 2:
            out[tag] = None
            continue
        out[tag] = {
            "n": len(s),
            "ann_mean": s.mean() * 252,
            "psr_1": deflated_sharpe_ratio(s, n_trials=1),
            "dsr_n": deflated_sharpe_ratio(s, n_trials=n_trials),
        }
    return out


def main(run_path: str = None, since: str = "2011-01-31", n_trials: int = 16):
    path = Path(run_path) if run_path else artifacts.latest_run()
    run = artifacts.load_run(path)
    print(f"Re-analyzing {path}\nconfig: {run['config']}")

    alpha_returns = run["alpha_curve"].pct_change().dropna()
    eq_returns = run["eq_curve"].pct_change().dropna()
    cdi_series = run["cdi_series"]
    active = (alpha_returns - eq_returns).dropna()
    excess_cdi = (alpha_returns - cdi_series.reindex(alpha_returns.index).ffill() / 100).dropna()

    cutoff = pd.Timestamp(since)
    for label, series in [("active-return-vs-EW", active), ("excess-CDI", excess_cdi)]:
        print(f"\n=== {label} ===")
        table = dsr_by_era(series, cutoff, n_trials)
        for tag, stats in [("full sample", table["full"]), (f"since {cutoff.date()}", table["since_cutoff"])]:
            if stats is None:
                print(f"  {tag}: too few observations")
                continue
            print(f"  {tag:<20} n={stats['n']:<5} ann.mean={stats['ann_mean']:>7.2%}  "
                  f"PSR@1={stats['psr_1']:.3f}  DSR@{n_trials}={stats['dsr_n']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None, help="run directory (default: most recent)")
    parser.add_argument("--since", type=str, default="2011-01-31")
    parser.add_argument("--n-trials", type=int, default=16)
    args = parser.parse_args()
    main(run_path=args.run, since=args.since, n_trials=args.n_trials)
