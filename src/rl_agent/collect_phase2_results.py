"""
collect_phase2_results.py — Extract Phase 2 seed-ensemble results and update EIIE_DIAGNOSIS_PLAN.md

Usage:
  python src/rl_agent/collect_phase2_results.py
"""

import json
from pathlib import Path
from collections import defaultdict


def extract_seed_results(run_dirs):
    """Extract metrics from run directories, keyed by seed."""
    results_by_seed = defaultdict(dict)

    for run_dir in sorted(run_dirs):
        metrics_file = run_dir / "metrics_summary.json"
        config_file = run_dir / "config.json"

        if not metrics_file.exists():
            continue

        # Extract seed from config
        with open(config_file) as f:
            cfg = json.load(f)
        seed = cfg.get("train", {}).get("seed", "unknown")

        # Extract metrics
        with open(metrics_file) as f:
            metrics = json.load(f)

        agent = metrics.get("agent", {})
        baselines = metrics.get("baselines", {})

        results_by_seed[seed] = {
            "total_return": agent.get("total_return"),
            "sharpe": agent.get("sharpe"),
            "vs_cdi": agent.get("total_return", 0) - baselines.get("constant_cash", {}).get("total_return", 0),
            "vs_bova11": agent.get("total_return", 0) - baselines.get("bova11", {}).get("total_return", 0),
            "turnover": agent.get("mean_daily_turnover"),
            "run_dir": run_dir.name,
        }

    return results_by_seed


def format_table(results_by_seed):
    """Generate markdown table for EIIE_DIAGNOSIS_PLAN.md."""
    if not results_by_seed:
        return "No Phase 2 results yet.\n"

    lines = ["| Seed | Agent Return | vs CDI | vs BOVA11 | Sharpe | Mean Turnover | Run Dir |"]
    lines.append("|------|---|---|---|---|---|---|")

    # Sort by seed
    for seed in sorted(results_by_seed.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
        r = results_by_seed[seed]
        lines.append(
            f"| {seed} | {r['total_return']:>10.2%} | {r['vs_cdi']:>8.2%} | {r['vs_bova11']:>9.2%} | "
            f"{r['sharpe']:>8.2f} | {r['turnover']:>13.6f} | {r['run_dir']} |"
        )

    # Compute medians and stats
    if results_by_seed:
        returns = [r["total_return"] for r in results_by_seed.values() if r["total_return"] is not None]
        vs_cdi = [r["vs_cdi"] for r in results_by_seed.values() if r["vs_cdi"] is not None]
        sharpes = [r["sharpe"] for r in results_by_seed.values() if r["sharpe"] is not None]

        if returns:
            returns.sort()
            median_return = returns[len(returns) // 2]
            median_vs_cdi = sorted(vs_cdi)[len(vs_cdi) // 2] if vs_cdi else 0
            median_sharpe = sorted(sharpes)[len(sharpes) // 2] if sharpes else 0

            lines.append("|---|---|---|---|---|---|---|")
            lines.append(
                f"| **Median** | **{median_return:.2%}** | **{median_vs_cdi:.2%}** | "
                f"**{sorted([r['vs_bova11'] for r in results_by_seed.values()])[len(results_by_seed)//2]:.2%}** | "
                f"**{median_sharpe:.2f}** | | **decision gate** |"
            )

    return "\n".join(lines) + "\n"


def update_plan(phase_2_results_table):
    """Update EIIE_DIAGNOSIS_PLAN.md with Phase 2 results table."""
    plan_file = Path("EIIE_DIAGNOSIS_PLAN.md")

    with open(plan_file) as f:
        content = f.read()

    # Find and replace the Phase 2 Results Table section
    marker_start = "**Phase 2 Results Table** (to be filled):"
    marker_end = "---"

    if marker_start not in content:
        print(f"Warning: marker '{marker_start}' not found in {plan_file}")
        return

    parts = content.split(marker_start)
    before = parts[0] + marker_start + "\n\n"

    # Extract the part after the table
    after_marker = parts[1]
    if marker_end in after_marker:
        after_idx = after_marker.index(marker_end)
        after = "\n" + after_marker[after_idx:]
    else:
        after = ""

    # Write updated content
    updated = before + phase_2_results_table + after
    with open(plan_file, "w") as f:
        f.write(updated)

    print(f"Updated {plan_file}")


def main():
    # Find all Phase 2 runs (seed-based naming)
    exp_dir = Path("experiments")
    all_runs = sorted(exp_dir.glob("eiie_baseline_20260717T*"))

    if not all_runs:
        print("No experiments found yet. Waiting for seed runs to complete...")
        return

    results = extract_seed_results(all_runs)
    table = format_table(results)

    print("\n=== PHASE 2 RESULTS ===\n")
    print(table)

    # Update plan file
    update_plan(table)


if __name__ == "__main__":
    main()
