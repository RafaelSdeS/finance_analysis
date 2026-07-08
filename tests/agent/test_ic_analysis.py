"""
Unit tests for ic_analysis module (synthetic panel, no data files).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.ic_analysis import daily_rank_ic, summarize


def main():
    """Synthetic IC tests: perfect signal, inverted, random, min_names."""
    print("✓ Test 1: Perfect positive IC (feature ranks match forward returns)")
    np.random.seed(42)
    n_dates, n_names = 50, 20
    feat = pd.DataFrame(
        np.random.randn(n_dates, n_names),
        columns=[f"t{i}" for i in range(n_names)],
        index=pd.date_range("2020-01-01", periods=n_dates),
    )
    fwd = feat.copy()  # Identical ranks → IC ≈ 1
    ic = daily_rank_ic(feat, fwd, min_names=10)
    assert ic.mean() > 0.99, f"Expected IC ≈ 1, got {ic.mean()}"
    print(f"  Mean IC: {ic.mean():.4f}")

    print("\n✓ Test 2: Perfect negative IC")
    fwd_inv = -feat  # Inverted ranks → IC ≈ -1
    ic_inv = daily_rank_ic(feat, fwd_inv, min_names=10)
    assert ic_inv.mean() < -0.99, f"Expected IC ≈ -1, got {ic_inv.mean()}"
    print(f"  Mean IC: {ic_inv.mean():.4f}")

    print("\n✓ Test 3: Random IC (should be small)")
    fwd_rand = pd.DataFrame(
        np.random.randn(n_dates, n_names),
        columns=[f"t{i}" for i in range(n_names)],
        index=feat.index,
    )
    ic_rand = daily_rank_ic(feat, fwd_rand, min_names=10)
    assert abs(ic_rand.mean()) < 0.1, f"Expected small IC, got {ic_rand.mean()}"
    print(f"  Mean IC: {ic_rand.mean():.4f}")

    print("\n✓ Test 4: Min names exclusion")
    feat_sparse = feat.copy()
    feat_sparse.iloc[:5] = np.nan  # First 5 dates have <min_names valid
    ic_sparse = daily_rank_ic(feat_sparse, fwd, min_names=10)
    assert ic_sparse.iloc[:5].isna().all(), "Days with <min_names should be NaN"
    print(f"  First 5 dates NaN: {ic_sparse.iloc[:5].isna().sum()}/5")

    print("\n✓ Test 5: Summarize IC")
    summary = summarize(ic, 5, "test_feature")
    assert summary["horizon"] == 5
    assert summary["n_days"] > 0
    assert "mean_ic" in summary
    assert "non_overlap_t_stat" in summary
    print(f"  Summary keys: {list(summary.keys())}")

    print("\n✓ All ic_analysis tests passed")


if __name__ == "__main__":
    main()
