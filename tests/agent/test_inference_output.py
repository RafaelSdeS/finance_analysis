#!/usr/bin/env python3
"""
Inference validation: weight invariants + fallback path.

Run from project root: python tests/agent/test_inference_output.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.infer import predict_weights


def main() -> None:
    print("=" * 60)
    print("TEST: inference output invariants")
    print("=" * 60)

    # --- 1. Latest date, real model ---
    w = predict_weights()
    assert abs(w["weight"].sum() - 1.0) < 1e-6, f"weights sum {w['weight'].sum()}"
    assert (w["weight"] > 0).all(), "non-positive weight in output"
    assert w["weight"].is_monotonic_decreasing, "not sorted by weight"
    assert not w["ticker"].duplicated().any(), "duplicate tickers"
    assert np.isfinite(w["weight"]).all(), "non-finite weight"
    assert "model" in w.attrs["source"], f"expected model source, got {w.attrs['source']}"
    print(f"✓ latest date ({w.attrs['date']}): {len(w)} positions, sum=1, sorted, model used")

    # --- 2. Specific historical date ---
    w2 = predict_weights(date="2023-06-15")
    assert w2.attrs["date"] <= "2023-06-15", "resolved date after requested"
    assert abs(w2["weight"].sum() - 1.0) < 1e-6
    print(f"✓ historical date: requested 2023-06-15 → resolved {w2.attrs['date']}")

    # --- 3. Fallback: nonexistent model → equal weight, no crash ---
    w3 = predict_weights(model_path=Path("data/models/DOES_NOT_EXIST.zip"))
    assert "FALLBACK" in w3.attrs["source"], "fallback not triggered"
    assert abs(w3["weight"].sum() - 1.0) < 1e-6
    # equal weight: all active tickers get identical weight
    assert w3["weight"].nunique() == 1, "fallback should be exactly equal-weight"
    print(f"✓ fallback: missing model → equal weight over {len(w3)} active tickers")

    # --- 4. Date before test range → clear error ---
    try:
        predict_weights(date="1999-01-01")
        raise AssertionError("expected ValueError for pre-range date")
    except ValueError:
        print("✓ pre-range date raises ValueError")

    print("\nALL INFERENCE TESTS PASSED ✓")


if __name__ == "__main__":
    main()
