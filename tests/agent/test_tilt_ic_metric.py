"""
Unit tests for metrics.daily_tilt_ic / tilt_ic_significance (synthetic, no data files).

DIAGNOSIS_PLAN.md Phase 0.5 (F3): a time-series excess-Sharpe test needs the effect
averaged over T_years of daily observations (SE ~ 1/sqrt(T_years)) -- the theoretical
ceiling for this strategy class (~0.2-0.4 Sharpe) is barely resolvable even pooled over
16 years. The tilt-IC test instead has one cross-section PER REBALANCE DATE with N names
each, giving far more effective observations for the same calendar span. Mirrors
test_significance_stats.py's calibration style: null false-positive rate near nominal,
then a power check at the magnitude Phase 0 (0.3/0.4) actually measured (IC ~0.03-0.10).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.metrics import daily_tilt_ic, tilt_ic_significance


def main() -> None:
    print("Test 1 (calibration): null false-positive rate is near the nominal 5%")
    # Repeated trials of tilts that are pure noise, independent of forward returns --
    # H0 (mean tilt-IC == 0) is true by construction in every trial.
    n_trials, T, N, min_names = 300, 100, 50, 10
    rejections = 0
    for trial in range(n_trials):
        r = np.random.default_rng(1000 + trial)
        ew = np.full((T, N), 1.0 / N)
        weights = ew + r.normal(scale=0.01, size=(T, N))
        fwd = r.normal(scale=0.02, size=(T, N))  # independent of the tilt
        result = tilt_ic_significance(weights, ew, fwd, min_names=min_names, lag=1)
        if result["p_value"] < 0.05:
            rejections += 1
    null_rate = rejections / n_trials
    print(f"  false-positive rate over {n_trials} null trials: {100 * null_rate:.1f}% (nominal 5%)")
    assert null_rate < 0.12, f"expected null false-positive rate near nominal 5%, got {null_rate:.1%}"

    print("\nTest 2 (power): a genuine IC~0.05 tilt signal is detected over ~16y of rebalances")
    # ~190 rebalance dates (matches ~16 years at a 21-day rebalance interval) x 50 names --
    # the actual scale of the project's tradable universe and window budget. Tilt is a
    # noisy linear mixture of the (standardized) forward return + independent noise, with
    # mixing weight chosen so the expected Pearson correlation (~= Spearman for small rho
    # on jointly-Gaussian data) is 0.05 -- the same order of magnitude Phase 0's IC audits
    # (0.3, 0.4) actually measured on this project's real data, not a toy near-1.0 signal.
    T2, N2 = 190, 50
    target_rho = 0.05
    a, b = target_rho, float(np.sqrt(1 - target_rho ** 2))
    r = np.random.default_rng(42)
    ew2 = np.full((T2, N2), 1.0 / N2)
    fwd2 = r.normal(scale=0.02, size=(T2, N2))
    fwd2_z = (fwd2 - fwd2.mean(axis=1, keepdims=True)) / (fwd2.std(axis=1, keepdims=True) + 1e-9)
    noise2 = r.normal(size=(T2, N2))
    tilt2 = a * fwd2_z + b * noise2
    weights2 = ew2 + 0.01 * tilt2  # scale doesn't matter for RANK correlation, kept small/realistic

    result2 = tilt_ic_significance(weights2, ew2, fwd2, min_names=min_names, lag=1)
    print(f"  mean daily IC: {result2['mean']:.4f} (target Pearson rho={target_rho}), "
          f"t_stat={result2['t_stat']:.2f}, p_value={result2['p_value']:.2e}, "
          f"n_valid_days={result2['n_valid_days']}/{result2['n_total_days']}")

    assert result2["n_valid_days"] == T2, "every day should have >=min_names active names here"
    assert 0.01 < result2["mean"] < 0.12, (
        f"expected realized mean IC in the same order of magnitude as Phase 0's measured "
        f"signal (0.03-0.10), got {result2['mean']}"
    )
    assert result2["p_value"] < 0.05, (
        f"expected the tilt-IC test to detect a genuine IC~0.05 signal over {T2} rebalance "
        f"dates x {N2} names -- got p={result2['p_value']}"
    )
    print("  ✓ detected as significant")

    print("\nTest 3: min_names / masking honored")
    thin_active = np.zeros((5, N), dtype=bool)
    thin_active[:, :5] = True  # only 5 active names, below min_names=10
    ic_thin = daily_tilt_ic(weights2[:5], ew2[:5], fwd2[:5], active=thin_active, min_names=10)
    assert np.isnan(ic_thin).all(), "days with <min_names active names must be NaN, not silently included"
    print("  ✓ under-min_names days correctly excluded (NaN)")

    print("\n✓ All tilt_ic_metric tests passed")


if __name__ == "__main__":
    main()
