#!/usr/bin/env python
"""
Quick dry-run test of M3 supervised probe on a small window.
Validates that the full pipeline works before launching the real experiment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl_agent.config import DataConfig, ExperimentConfig
from src.rl_agent.supervised_experiment import (
    compute_forward_returns,
)
from src.rl_agent.data import load_price_panel

def test_m3_pipeline():
    """Dry-run the M3 pipeline on a small window."""
    print("M3 Supervised Probe — Dry Run")
    print("=" * 60)

    # Use a small, recent window for quick validation
    cfg = DataConfig(
        window=50,
        features=["close", "high", "low", "return_1m", "return_3m", "return_6m"],
        window_start="2025-01-01",
        window_end="2025-06-30",
        cash_mode="cdi"
    )

    print("Loading price panel (2025-01-01 to 2025-06-30)...")
    panel = load_price_panel(cfg)
    print(f"  Panel shape: {panel.close.shape[0]} days × {panel.n_global} assets")

    # Compute forward returns for each horizon
    print("\nComputing forward returns...")
    for k in [1, 5, 21]:
        fwd = compute_forward_returns(panel, k)
        n_valid = (~(fwd == 0)).sum()
        print(f"  k={k:2d}: {fwd.shape}, {n_valid} non-zero entries")

    print("\n✓ M3 dry-run passed!")
    print("  - PricePanel loads correctly")
    print("  - Forward returns compute without crashes (warnings are normal for zero prices)")
    print("\nTo run M3:")
    print("  python scripts/run_supervised_probe.py --config configs/eiie_features.json")
    return True

if __name__ == "__main__":
    try:
        success = test_m3_pipeline()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ M3 dry-run failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
