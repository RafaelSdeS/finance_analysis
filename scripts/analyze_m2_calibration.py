#!/usr/bin/env python
"""
Analyze M2 entropy calibration results.

Extracts effective_n, return, Sharpe from all calibration runs and recommends
which entropy_beta_end to use for the full SEP.
"""

import json
from pathlib import Path
from collections import defaultdict


def main():
    exp_dir = Path("experiments")
    runs_by_beta = defaultdict(list)

    # Scan for M2 calibration results
    for run_dir in sorted(exp_dir.glob("eiie_m2_calib_*")):
        # Extract beta from directory name
        parts = run_dir.name.split("_")
        if len(parts) < 5:
            continue

        beta_str = parts[3]  # e.g., "1e-3"
        metrics_file = run_dir / "metrics_summary.json"

        if not metrics_file.exists():
            continue

        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"  Error reading {metrics_file}: {e}")
            continue

        agent = metrics.get("agent", {})
        runs_by_beta[beta_str].append({
            "run": run_dir.name,
            "effective_n": agent.get("effective_n"),
            "total_return": agent.get("total_return"),
            "sharpe_vs_cdi": agent.get("sharpe_vs_cdi"),
        })

    if not runs_by_beta:
        print("No M2 calibration results found.")
        return

    print("\nM2 Entropy Calibration Summary")
    print("=" * 80)
    print("Target: effective_n ≈ 5–15 (diversified but trained)\n")

    for beta in sorted(runs_by_beta.keys()):
        runs = runs_by_beta[beta]
        print(f"entropy_beta_end = {beta}")
        for run in runs:
            eff_n = run["effective_n"]
            ret = run["total_return"]
            sharpe = run["sharpe_vs_cdi"]

            eff_n_str = f"{eff_n:.2f}" if eff_n else "N/A"
            ret_str = f"{ret:.1%}" if ret else "N/A"
            sharpe_str = f"{sharpe:.3f}" if sharpe else "N/A"

            in_target = "✓" if eff_n and 5 <= eff_n <= 15 else " "
            print(f"  {in_target} eff_n={eff_n_str:>8s}  return={ret_str:>10s}  sharpe={sharpe_str:>8s}")
        print()

    print("\nRecommendation:")
    # Find beta(s) in the target range
    candidates = []
    for beta in sorted(runs_by_beta.keys()):
        runs = runs_by_beta[beta]
        for run in runs:
            if run["effective_n"] and 5 <= run["effective_n"] <= 15:
                candidates.append((beta, run["effective_n"], run["sharpe_vs_cdi"]))

    if candidates:
        best = max(candidates, key=lambda x: x[2])  # Sort by Sharpe
        print(f"  Use entropy_beta_end = {best[0]} (eff_n={best[1]:.2f}, Sharpe={best[2]:.3f})")
        print(f"\n  Next: Run full SEP with this beta:")
        print(f"    python -m src.rl_agent.sweep --config configs/eiie_m2_beta_{best[0]}.json --seeds 1-8 -j 1")
    else:
        print("  No beta in target range 5–15. Options:")
        print("  1. Try intermediate values (e.g., 5e-3 between 3e-3 and 1e-2)")
        print("  2. Use the closest value and check results")


if __name__ == "__main__":
    main()
