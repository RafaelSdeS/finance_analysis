#!/usr/bin/env python
"""
M4 Decision Gate analyzer.

Reads M2 and M3 results, applies the decision table from EIIE_IMPROVEMENT_PLAN.md
lines 789–810, and recommends next step.
"""

import json
from pathlib import Path
from typing import Optional


def load_m3_results(exp_dir: str = "experiments/eiie_features_supervised") -> Optional[dict]:
    """Load M3 supervised probe results."""
    results_path = Path(exp_dir) / "supervised_results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        return json.load(f)


def load_m2_results(beta_str: str = "6.3e-4") -> Optional[dict]:
    """Load M2 entropy calibration results (best run so far)."""
    exp_dir = Path("experiments")
    m2_dirs = sorted(
        exp_dir.glob(f"eiie_m2_calib_{beta_str}_*"),
        key=lambda p: p.name,
        reverse=True,
    )
    if not m2_dirs:
        return None

    results = []
    for d in m2_dirs:
        metrics_path = d / "metrics_summary.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                data = json.load(f)
                results.append({
                    "dir": d.name,
                    "effective_n": data["agent"].get("effective_n_holdings"),
                    "sharpe": data["agent"].get("sharpe"),
                    "entropy": data["agent"].get("allocation_entropy"),
                })
    return results if results else None


def classify_m3_signal(m3_results: dict) -> str:
    """Classify M3 signal strength: Null, Weak, or Strong."""
    if not m3_results:
        return "UNKNOWN"

    # Threshold definitions (from EIIE_IMPROVEMENT_PLAN.md)
    strong_threshold = 0.03
    weak_threshold = 0.03

    strong_count = 0
    weak_count = 0

    for k_str, res in m3_results.items():
        k = int(k_str)
        val_ic = res["val"]["daily_ic"]
        perm_null_975 = res["val"]["perm_null_975pct"]

        # Signal if IC > 97.5th percentile of permutation null
        if val_ic > perm_null_975:
            if val_ic >= strong_threshold:
                strong_count += 1
            else:
                weak_count += 1

    if strong_count > 0:
        return "STRONG"
    elif weak_count > 0:
        return "WEAK"
    else:
        return "NULL"


def main():
    print("\n" + "=" * 70)
    print("M4 DECISION GATE ANALYZER")
    print("=" * 70 + "\n")

    # Load M3 results
    print("Loading M3 supervised ranking probe results...")
    m3_results = load_m3_results()
    if m3_results:
        print(f"✓ Found M3 results")
        m3_signal = classify_m3_signal(m3_results)
        print(f"  Signal classification: {m3_signal}\n")

        # Print details
        for k_str in sorted(m3_results.keys(), key=lambda x: int(x)):
            res = m3_results[k_str]
            val_ic = res["val"]["daily_ic"]
            perm_null_975 = res["val"]["perm_null_975pct"]
            p_val = res["val"]["p_value"]
            signal = val_ic > perm_null_975
            print(f"  k={k_str:2s}: IC={val_ic:+.4f} | null_975={perm_null_975:+.4f} | "
                  f"signal={'✓' if signal else '✗'} | p={p_val:.3f}")
    else:
        print("✗ M3 results not found (still running or failed)")
        m3_signal = "UNKNOWN"

    print()

    # Load M2 results
    print("Loading M2 entropy calibration results (entropy_beta_end=6.3e-4)...")
    m2_results = load_m2_results("6.3e-4")
    if m2_results:
        print(f"✓ Found {len(m2_results)} M2 run(s)\n")
        m2_effective_n_vals = []
        for res in m2_results:
            print(f"  {res['dir']}")
            print(f"    effective_n_holdings: {res['effective_n']:.2f}")
            print(f"    entropy: {res['entropy']:.4f}")
            print(f"    Sharpe: {res['sharpe']:.4f}\n")
            if res["effective_n"] is not None:
                m2_effective_n_vals.append(res["effective_n"])

        # Assess M2
        if not m2_effective_n_vals:
            m2_status = "NULL/INCONCLUSIVE"
        elif all(5 <= n <= 15 for n in m2_effective_n_vals):
            m2_status = "TARGET_HIT"
        else:
            m2_status = "BIFURCATION (outside [5,15])"
    else:
        print("✗ M2 results not found")
        m2_status = "UNKNOWN"

    print()

    # M4 Decision Gate
    print("=" * 70)
    print("M4 DECISION GATE (from EIIE_IMPROVEMENT_PLAN.md lines 789–810)")
    print("=" * 70 + "\n")

    if m3_signal == "UNKNOWN":
        print("⚠ Cannot apply M4 gate: M3 results not yet available.\n")
        print("Run: python scripts/run_supervised_probe.py --config configs/eiie_features.json")
    elif m3_signal == "NULL":
        print("M3 RESULT: NULL (IC inside permutation null at all k)\n")
        if m2_status == "NULL/INCONCLUSIVE":
            print("DECISION: **STOP model-side work.**")
            print("  - Both M2 and M3 null → no evidence of extractable signal")
            print("  - Next: run confirmation on disjoint window")
            print("  - Then: pivot to objective-level research (risk mandate, macro-conditioning)")
        else:
            print("DECISION: Chase RL-specific signal")
            print("  - M3 null but M2 found something → the RL objective may encode")
            print("    information that supervised probe can't extract directly")
            print("  - Next: test RL config on disjoint window, no new features/capacity")
    elif m3_signal == "WEAK":
        print("M3 RESULT: WEAK (IC 0.02–0.03, above null but not strong)\n")
        print("DECISION: **Allocation research only**")
        print("  - Test portfolio layer (turnover-budgeted top-k ranking)")
        print("  - Do NOT run model-improvement experiments")
    elif m3_signal == "STRONG":
        print("M3 RESULT: STRONG (IC ≥ 0.03)\n")
        if m2_status == "NULL/INCONCLUSIVE" or m2_status == "BIFURCATION (outside [5,15])":
            print("DECISION: Head-to-head pipeline comparison")
            print("  - Supervised (predict-then-allocate) vs. RL fine-tune at winning k")
            print("  - Winner on both windows becomes the system")
        else:
            print("DECISION: **Objective-level regularization research**")
            print("  - M2 also positive (entropy floor helped RL)")
            print("  - Evidence: signal EXISTS and RL can EXPRESS it when regularized")
            print("  - Next: tune objective (entropy floor, risk-sensitive reward)")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
