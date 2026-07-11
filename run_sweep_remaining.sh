#!/bin/bash
# Phase 5 Item 4 sweep, continued: remaining 8 configs (ra0_seed0 already done by run_sweep.sh).
# --n-envs 6 (down from default 16) to cut PPO rollout-buffer memory ~2.7x -- obs_dim=5757
# (full PIT 101-ticker universe x 55 features) is ~4.7x what n_envs=16 was benchmarked against
# in CLAUDE.md (obs_dim=1224, universe-size 50), and swap hit 95% full during ra0_seed0.
set -e
cd /home/rafael/Documents/finance_analysis

SUMMARY_DIR="artifacts/backtest/sweep_20260710"
mkdir -p "$SUMMARY_DIR"

# ra0_seed0 already completed by run_sweep.sh -- skip it here
declare -a CONFIGS=("0 1" "0 2" "1 0" "1 1" "1 2" "5 0" "5 1" "5 2")

for cfg in "${CONFIGS[@]}"; do
  read -r lambda seed <<< "$cfg"
  tag="ra${lambda}_seed${seed}"
  echo "=== Starting $tag at $(date) ==="
  python -m src.agent.trainer --risk-aversion "$lambda" --seed "$seed" \
    --timesteps 500000 --no-promote --n-envs 6 \
    > "$SUMMARY_DIR/${tag}.log" 2>&1
  echo "=== Finished $tag at $(date), exit=$? ==="

  run_dir=$(ls -td artifacts/models/runs/*/ | head -1)
  echo "  run_dir=$run_dir"

  mkdir -p "$SUMMARY_DIR/$tag"
  cp "$run_dir"*.jsonl "$SUMMARY_DIR/$tag/" 2>/dev/null || true
  cp "$run_dir"*.json "$SUMMARY_DIR/$tag/" 2>/dev/null || true

  rm -rf "$run_dir"
  echo "  cleaned up $run_dir, summary saved to $SUMMARY_DIR/$tag/"
  df -h / | tail -1
  free -h | head -2
done

echo "ALL REMAINING SWEEP RUNS COMPLETE"
