#!/usr/bin/env python3
"""
hac_mean_test / block_bootstrap_mean_ci: autocorrelation-robust significance.

Confirms the exact problem that motivated these functions: a naive t-test on
autocorrelated data (e.g. daily returns under N-day rebalancing) massively
over-rejects H0, while HAC and block-bootstrap are much better calibrated.

Run from project root: python tests/agent/test_significance_stats.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.metrics import hac_mean_test, block_bootstrap_mean_ci


def main() -> None:
    rng = np.random.default_rng(42)
    n_trials = 300
    n_days = 500
    phi = 0.9   # strong autocorrelation, similar persistence to 21-day rebalance blocks
    lag = 21    # matches rebalance_interval_days

    naive_rejections = 0
    hac_rejections = 0
    bootstrap_rejections = 0

    for trial in range(n_trials):
        # AR(1) series with TRUE mean = 0 (H0 is correct) but strong serial correlation.
        eps = rng.normal(0, 1, n_days)
        x = np.zeros(n_days)
        for t in range(1, n_days):
            x[t] = phi * x[t - 1] + eps[t]
        x *= 0.001  # realistic daily-return scale

        _, p_naive = scipy_stats.ttest_1samp(x, 0.0)
        naive_rejections += p_naive < 0.05

        hac = hac_mean_test(x, lag=lag)
        hac_rejections += hac["p_value"] < 0.05

        boot = block_bootstrap_mean_ci(x, block_size=lag, n_resamples=200, seed=trial)
        bootstrap_rejections += not (boot["ci_low"] <= 0 <= boot["ci_high"])

    naive_rate = naive_rejections / n_trials
    hac_rate = hac_rejections / n_trials
    boot_rate = bootstrap_rejections / n_trials

    print(f"True H0 correct (mean=0), nominal false-positive rate = 5%")
    print(f"  naive t-test:     {100*naive_rate:.1f}% false rejections")
    print(f"  HAC t-test:       {100*hac_rate:.1f}% false rejections")
    print(f"  block bootstrap:  {100*boot_rate:.1f}% false rejections")

    assert naive_rate > 0.30, (
        f"expected the naive test to badly over-reject on autocorrelated data, got {naive_rate:.1%}"
    )
    assert hac_rate < naive_rate / 2, (
        f"HAC should cut the false-positive rate by more than half vs naive: "
        f"naive={naive_rate:.1%}, hac={hac_rate:.1%}"
    )
    assert boot_rate < naive_rate / 2, (
        f"bootstrap should cut the false-positive rate by more than half vs naive: "
        f"naive={naive_rate:.1%}, bootstrap={boot_rate:.1%}"
    )
    print("✓ naive test over-rejects under autocorrelation; HAC and bootstrap are far better calibrated")

    # --- Sanity: a series with a genuine, large mean shift should be detected by all three ---
    x_signal = rng.normal(0, 1, n_days) * 0.001 + 0.01  # mean=0.01, way outside noise scale
    _, p_naive_signal = scipy_stats.ttest_1samp(x_signal, 0.0)
    hac_signal = hac_mean_test(x_signal, lag=lag)
    boot_signal = block_bootstrap_mean_ci(x_signal, block_size=lag, n_resamples=200, seed=1)

    assert p_naive_signal < 0.05, "naive test should catch an obvious large mean shift"
    assert hac_signal["p_value"] < 0.05, "HAC test should catch an obvious large mean shift"
    assert not (boot_signal["ci_low"] <= 0 <= boot_signal["ci_high"]), (
        "bootstrap CI should exclude 0 for an obvious large mean shift"
    )
    print("✓ all three tests correctly detect a genuine, large mean shift (not just conservative)")

    print("\nALL SIGNIFICANCE-STATS TESTS PASSED ✓")


if __name__ == "__main__":
    main()
