#!/bin/bash
# Phase 5 Item 4: risk_aversion sweep {0,1,5} x seed {0,1,2}, sequential, self-cleaning.
# Each run's heavy window_*.zip files are deleted right after its summary is extracted,
# to avoid the disk-full problem from the last attempt (9 full runs ~5-7GB, we have ~5.3GB free).
set -e
cd /home/rafael/Documents/finance_analysis

SUMMARY_DIR="artifacts/backtest/sweep_20260710"
mkdir -p "$SUMMARY_DIR"

for lambda in 0 1 5; do
  for seed in 0 1 2; do
    tag="ra${lambda}_seed${seed}"
    echo "=== Starting $tag at $(date) ==="
    python -m src.agent.trainer --risk-aversion "$lambda" --seed "$seed" \
      --timesteps 500000 --no-promote \
      > "$SUMMARY_DIR/${tag}.log" 2>&1
    echo "=== Finished $tag at $(date), exit=$? ==="

    # Find the run dir this just created (newest under artifacts/models/runs/)
    run_dir=$(ls -td artifacts/models/runs/*/ | head -1)
    echo "  run_dir=$run_dir"

    # Copy just the small artifacts (jsonl logs + sidecars), not the .zip model weights
    mkdir -p "$SUMMARY_DIR/$tag"
    cp "$run_dir"*.jsonl "$SUMMARY_DIR/$tag/" 2>/dev/null || true
    cp "$run_dir"*.json "$SUMMARY_DIR/$tag/" 2>/dev/null || true

    # Delete the heavy model weights now that jsonl/json are saved
    rm -rf "$run_dir"
    echo "  cleaned up $run_dir, summary saved to $SUMMARY_DIR/$tag/"
    df -h / | tail -1
  done
done

echo "ALL SWEEP RUNS COMPLETE"
