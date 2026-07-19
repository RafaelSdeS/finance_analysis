#!/usr/bin/env python
"""
CLI for M3 supervised ranking probe experiment.

Usage:
    python scripts/run_supervised_probe.py --config configs/eiie_features.json [--out-dir experiments]
"""

import argparse
from pathlib import Path

from src.rl_agent.config import ExperimentConfig
from src.rl_agent.supervised_experiment import run_supervised_experiment


def main():
    parser = argparse.ArgumentParser(description="Run M3 supervised ranking probe")
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    parser.add_argument("--out-dir", default="experiments", help="Output directory")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = ExperimentConfig.from_json(config_path)

    # Override experiment name for supervised probe
    config.experiment.name = f"{config.experiment.name}_supervised"

    out_dir = Path(args.out_dir) / config.experiment.name
    print(f"\nM3 Supervised Ranking Probe\n{'='*60}")
    print(f"Config: {config_path}")
    print(f"Output: {out_dir}\n")

    results = run_supervised_experiment(config, out_dir)

    # Print summary
    print(f"\n{'='*60}")
    print("Results Summary:")
    for k in sorted(results.keys()):
        res = results[k]
        train_ic = res["train"]["daily_ic"]
        val_ic = res["val"]["daily_ic"]
        val_signal = res["signal"]
        print(
            f"  k={k:2d}: train_IC={train_ic:+.4f}  "
            f"val_IC={val_ic:+.4f}  "
            f"signal={str(val_signal):>5s}"
        )


if __name__ == "__main__":
    main()
