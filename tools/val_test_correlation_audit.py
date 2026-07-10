"""
DIAGNOSIS_PLAN.md Phase 3.1: does "best val Sharpe" checkpoint selection actually track
test performance, or select regime luck?

Reads the per-window JSONL logs from a training run made with `trainer.py --log-test-eval`
(test_sharpe/test_sharpe_excess logged at every val-eval checkpoint, diagnostic only --
never used by early stopping or checkpoint selection during that run) and reports the
Spearman rank correlation between val_sharpe_excess and test_sharpe_excess across
checkpoints, per window and pooled.

Usage:
    python tools/val_test_correlation_audit.py <run_dir>
    # e.g. python tools/val_test_correlation_audit.py artifacts/logs/agent/runs/20260710-101817
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root, for `src.*` imports


def load_window_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Val->test checkpoint rank correlation (DIAGNOSIS_PLAN Phase 3.1)")
    parser.add_argument("run_dir", type=Path, help="artifacts/logs/agent/runs/<session_id> from a --log-test-eval run")
    args = parser.parse_args()

    window_files = sorted(args.run_dir.glob("window_*.jsonl")) + sorted(args.run_dir.glob("agent.jsonl"))
    if not window_files:
        raise SystemExit(f"No window_*.jsonl / agent.jsonl found in {args.run_dir}")

    all_val, all_test = [], []
    print("\n" + "=" * 90)
    print("VAL -> TEST CHECKPOINT RANK CORRELATION")
    print("=" * 90)
    for wf in window_files:
        records = load_window_jsonl(wf)
        records = [r for r in records if "test_sharpe_excess" in r]
        if len(records) < 3:
            print(f"  {wf.name}: only {len(records)} checkpoints with test data, skipping (need >=3)")
            continue
        val = np.array([r["val_sharpe_excess"] for r in records])
        test = np.array([r["test_sharpe_excess"] for r in records])
        rho, p = scipy_stats.spearmanr(val, test)
        print(f"  {wf.name}: n={len(records)}  spearman_rho={rho:+.3f}  p={p:.3f}  "
              f"(val range [{val.min():+.3f}, {val.max():+.3f}], test range [{test.min():+.3f}, {test.max():+.3f}])")
        all_val.extend(val.tolist())
        all_test.extend(test.tolist())

        # Does the checkpoint with the BEST val_sharpe_excess also have above-median test performance?
        best_idx = int(np.argmax(val))
        test_median = np.median(test)
        best_val_test = test[best_idx]
        print(f"    best-val checkpoint's test_sharpe_excess={best_val_test:+.3f} vs window's own "
              f"test median={test_median:+.3f} ({'ABOVE' if best_val_test >= test_median else 'BELOW'} median)")

    if len(all_val) >= 5:
        rho_pooled, p_pooled = scipy_stats.spearmanr(all_val, all_test)
        print(f"\nPooled across all windows: n={len(all_val)}  spearman_rho={rho_pooled:+.3f}  p={p_pooled:.3f}")
        if abs(rho_pooled) < 0.2:
            print("-> weak/no val-test correlation: early stopping may be selecting regime luck, "
                  "not real generalization. Consider purged/embargoed multi-fold validation.")
        else:
            print(f"-> {'positive' if rho_pooled > 0 else 'negative'} correlation detected: val Sharpe "
                  f"{'does' if rho_pooled > 0 else 'does NOT'} track test performance in this run.")
    else:
        print("\nToo few pooled checkpoints for a meaningful pooled correlation.")


if __name__ == "__main__":
    main()
