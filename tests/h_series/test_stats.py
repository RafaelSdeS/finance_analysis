"""
Test: h_series/stats.py's generic cross-sectional and HAC statistics.
Deterministic synthetic data, hand-verified to 1e-6 where the arithmetic is
small enough to check by hand; independent-formula cross-checks otherwise.

Run from project root:
    python tests/h_series/test_stats.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.h_series.stats import (  # noqa: E402
    benjamini_hochberg, min_detectable_ic, min_detectable_ir, newey_west_tstat,
    rank_normalize, sector_demean, spearman_ic_by_group, winsorize_cross_sectional,
)
from test_utils import print_check, print_header, print_section_end  # noqa: E402

TOL = 1e-6


def test_rank_normalize_single_group(passed, failed):
    values = pd.Series([10.0, 20.0, 30.0, 40.0])
    groups = pd.Series(["a"] * 4)
    out = rank_normalize(values, groups)
    expected = np.array([1, 2, 3, 4]) / 5.0 - 0.5  # N=4 -> rank/(N+1)-0.5
    ok = np.allclose(out.to_numpy(), expected, atol=TOL)
    print_check("rank_normalize: single group of 4 matches rank/(N+1)-0.5 exactly", ok, str(out.tolist()))
    return passed + ok, failed + (not ok)


def test_rank_normalize_bounded_regardless_of_n(passed, failed):
    values = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0])
    groups = pd.Series(["a", "a", "a", "b", "b"])
    out = rank_normalize(values, groups)
    ok = bool((out.abs() <= 0.5 + TOL).all())
    print_check("rank_normalize: bounded to [-0.5, 0.5] regardless of fluctuating group size N",
                ok, str(out.tolist()))
    return passed + ok, failed + (not ok)


def test_rank_normalize_nan_excluded_from_n(passed, failed):
    values = pd.Series([10.0, 20.0, np.nan, 40.0])
    groups = pd.Series(["a"] * 4)
    out = rank_normalize(values, groups)
    # N=3 non-NaN values -> rank/(3+1)-0.5 for the three real values; NaN stays NaN
    expected_real = np.array([1, 2, 3]) / 4.0 - 0.5
    real_vals = out.dropna().sort_values().to_numpy()
    ok = np.allclose(real_vals, expected_real, atol=TOL) and bool(out.isna().sum() == 1)
    print_check("rank_normalize: NaN excluded from N and stays NaN in output", ok, str(out.tolist()))
    return passed + ok, failed + (not ok)


def test_winsorize_cross_sectional(passed, failed):
    values = pd.Series([-100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    groups = pd.Series(["a"] * 7)
    out = winsorize_cross_sectional(values, groups, lower=0.10, upper=0.90)
    ok = bool(out.min() > -100.0 and out.max() < 100.0)
    print_check("winsorize_cross_sectional: extreme tails clipped inward", ok, str(out.tolist()))
    return passed + ok, failed + (not ok)


def test_sector_demean_basic(passed, failed):
    values = pd.Series([1.0, 2.0, 3.0])
    dates = pd.Series(["d1"] * 3)
    sectors = pd.Series(["s1"] * 3)
    out = sector_demean(values, dates, sectors)
    expected = np.array([1.0, 2.0, 3.0]) - 2.0
    ok = np.allclose(out.to_numpy(), expected, atol=TOL)
    print_check("sector_demean: subtracts (date, sector) mean exactly", ok, str(out.tolist()))
    return passed + ok, failed + (not ok)


def test_sector_demean_singleton_is_nan(passed, failed):
    values = pd.Series([1.0, 5.0])
    dates = pd.Series(["d1", "d1"])
    sectors = pd.Series(["s1", "s2"])  # each sector has exactly one member on d1
    out = sector_demean(values, dates, sectors)
    ok = bool(out.isna().all())
    print_check("sector_demean: sector-of-one groups are NaN (own value would trivially demean to 0)",
                ok, str(out.tolist()))
    return passed + ok, failed + (not ok)


def test_spearman_ic_perfect_monotonic(passed, failed):
    x = pd.Series([1, 2, 3, 4, 5] * 2, dtype=float)
    y = pd.Series([10, 20, 30, 40, 50] * 2, dtype=float)  # perfectly monotonic with x, within each group
    groups = pd.Series(["a"] * 5 + ["b"] * 5)
    ic = spearman_ic_by_group(x, y, groups)
    ok = np.allclose(ic.to_numpy(), [1.0, 1.0], atol=TOL)
    print_check("spearman_ic_by_group: perfect monotonic relation -> IC=1.0 per group", ok, str(ic.tolist()))
    return passed + ok, failed + (not ok)


def test_spearman_ic_below_min_n_is_nan(passed, failed):
    x = pd.Series([1.0, 2.0])
    y = pd.Series([10.0, 20.0])
    groups = pd.Series(["a", "a"])
    ic = spearman_ic_by_group(x, y, groups, min_n=5)
    ok = bool(ic.isna().all())
    print_check("spearman_ic_by_group: group below min_n is NaN, not a spurious 2-point correlation",
                ok, str(ic.tolist()))
    return passed + ok, failed + (not ok)


def test_newey_west_lag0_matches_population_se(passed, failed):
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 2.0, size=200)
    mean, se, tstat = newey_west_tstat(x, lag=0)
    expected_se = float(np.std(x, ddof=0) / np.sqrt(len(x)))  # independent formula, lag=0 == iid case
    ok = (abs(mean - x.mean()) < TOL and abs(se - expected_se) < TOL
          and abs(tstat - mean / expected_se) < TOL)
    print_check("newey_west_tstat: lag=0 SE matches population-variance/sqrt(T) exactly",
                ok, f"se={se:.8f} expected={expected_se:.8f}")
    return passed + ok, failed + (not ok)


def test_newey_west_hand_computed_lag2(passed, failed):
    # x = [1,2,3,4,5], mean=3, resid=[-2,-1,0,1,2]; hand-verified gamma_0/1/2 and Bartlett weights.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, se, tstat = newey_west_tstat(x, lag=2)
    gamma0 = 2.0                                    # mean(resid^2) = (4+1+0+1+4)/5
    gamma1 = 1.0                                     # mean([-1*-2, 0*-1, 1*0, 2*1]) = mean([2,0,0,2])
    gamma2 = -1.0 / 3.0                              # mean([0*-2, 1*-1, 2*0]) = mean([0,-1,0])
    w1, w2 = 1.0 - 1.0 / 3.0, 1.0 - 2.0 / 3.0
    lrv = gamma0 + 2.0 * (w1 * gamma1 + w2 * gamma2)
    expected_se = np.sqrt(lrv / 5.0)
    ok = abs(mean - 3.0) < TOL and abs(se - expected_se) < TOL
    print_check("newey_west_tstat: lag=2 matches hand-computed Bartlett-kernel formula",
                ok, f"se={se:.8f} expected={expected_se:.8f}")
    return passed + ok, failed + (not ok)


def test_newey_west_too_short_is_nan(passed, failed):
    mean, se, tstat = newey_west_tstat(np.array([1.0]), lag=0)
    ok = np.isnan(mean) and np.isnan(se) and np.isnan(tstat)
    print_check("newey_west_tstat: fewer than 2 points returns all-NaN", ok)
    return passed + ok, failed + (not ok)


def test_bh_textbook_example(passed, failed):
    # p sorted ascending by construction: thresh_i = (i/5)*0.05 = [.01,.02,.03,.04,.05]
    # sorted_p<=thresh: .005<=.01 T, .01<=.02 T, .03<=.03 T, .04<=.04 T, .20<=.05 F -> reject first 4
    p = np.array([0.005, 0.01, 0.03, 0.04, 0.20])
    reject = benjamini_hochberg(p, alpha=0.05)
    ok = np.array_equal(reject, [True, True, True, True, False])
    print_check("benjamini_hochberg: textbook example rejects first 4 of 5 sorted p-values",
                ok, str(reject.tolist()))
    return passed + ok, failed + (not ok)


def test_bh_shuffled_order_preserved(passed, failed):
    # same 5 p-values as above, shuffled -- reject mask must map back to ORIGINAL order
    p = np.array([0.20, 0.04, 0.005, 0.03, 0.01])
    reject = benjamini_hochberg(p, alpha=0.05)
    ok = np.array_equal(reject, [False, True, True, True, True])
    print_check("benjamini_hochberg: reject mask correctly maps back to shuffled input order",
                ok, str(reject.tolist()))
    return passed + ok, failed + (not ok)


def test_bh_nan_never_rejected(passed, failed):
    p = np.array([0.001, np.nan, 0.002])
    reject = benjamini_hochberg(p, alpha=0.10)
    ok = not reject[1]
    print_check("benjamini_hochberg: NaN p-value is never rejected", ok, str(reject.tolist()))
    return passed + ok, failed + (not ok)


def test_bh_no_survivors(passed, failed):
    p = np.array([0.5, 0.6, 0.7])
    reject = benjamini_hochberg(p, alpha=0.05)
    ok = not reject.any()
    print_check("benjamini_hochberg: nothing significant -> empty reject mask", ok, str(reject.tolist()))
    return passed + ok, failed + (not ok)


def test_min_detectable_ic_closed_form(passed, failed):
    # sigma_ic_null = 1/sqrt(49); n_eff = 90/(2+1) = 30; floor = 2*sigma/sqrt(30)
    val = min_detectable_ic(n_obs=90, n_assets=50, lag=2, t_threshold=2.0)
    expected = 2.0 * (1.0 / np.sqrt(49)) / np.sqrt(30.0)
    ok = abs(val - expected) < TOL
    print_check("min_detectable_ic: matches closed-form sigma_null/sqrt(n_eff) formula",
                ok, f"{val:.6f} vs {expected:.6f}")
    return passed + ok, failed + (not ok)


def test_min_detectable_ic_monotone_in_n_obs(passed, failed):
    lo_n = min_detectable_ic(n_obs=30, n_assets=50, lag=0)
    hi_n = min_detectable_ic(n_obs=300, n_assets=50, lag=0)
    ok = hi_n < lo_n
    print_check("min_detectable_ic: floor shrinks as n_obs grows (more power)", ok,
                f"n=30: {lo_n:.4f}, n=300: {hi_n:.4f}")
    return passed + ok, failed + (not ok)


def test_min_detectable_ir_closed_form(passed, failed):
    val = min_detectable_ir(n_obs_monthly=90, periods_per_year=12, t_threshold=2.0)
    expected = 2.0 * np.sqrt(12.0 / 90.0)
    ok = abs(val - expected) < TOL
    print_check("min_detectable_ir: matches closed-form t*sqrt(ppy/n) formula",
                ok, f"{val:.6f} vs {expected:.6f}")
    return passed + ok, failed + (not ok)


def main() -> int:
    print_header("h_series/stats.py")
    passed = failed = 0
    for test_fn in [
        test_rank_normalize_single_group,
        test_rank_normalize_bounded_regardless_of_n,
        test_rank_normalize_nan_excluded_from_n,
        test_winsorize_cross_sectional,
        test_sector_demean_basic,
        test_sector_demean_singleton_is_nan,
        test_spearman_ic_perfect_monotonic,
        test_spearman_ic_below_min_n_is_nan,
        test_newey_west_lag0_matches_population_se,
        test_newey_west_hand_computed_lag2,
        test_newey_west_too_short_is_nan,
        test_bh_textbook_example,
        test_bh_shuffled_order_preserved,
        test_bh_nan_never_rejected,
        test_bh_no_survivors,
        test_min_detectable_ic_closed_form,
        test_min_detectable_ic_monotone_in_n_obs,
        test_min_detectable_ir_closed_form,
    ]:
        passed, failed = test_fn(passed, failed)
    print_section_end(passed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
