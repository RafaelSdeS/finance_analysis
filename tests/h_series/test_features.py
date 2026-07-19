"""
Test: h_series/features.py's pure derivations (no parquet IO) --
days_since_filing's algebra and the freshness_factor transform.

Run from project root:
    python tests/h_series/test_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.h_series.features import days_since_filing, freshness_factor  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402

TOL = 1e-6


def test_days_since_filing_cancels_reference_date(passed, failed):
    # days_since_fundamental = trade_date - reference_date = 100
    # filing_lag_days = fundamentals_available_date - reference_date = 30
    # days_since_filing = trade_date - fundamentals_available_date = 100 - 30 = 70
    dsf = pd.Series([100.0])
    lag = pd.Series([30.0])
    out = days_since_filing(dsf, lag)
    ok = abs(float(out.iloc[0]) - 70.0) < TOL
    print_check("days_since_filing: (trade-reference) - (available-reference) = trade-available",
                ok, str(out.tolist()))
    return passed + ok, failed + (not ok)


def test_days_since_filing_late_filer_not_penalized_twice(passed, failed):
    # A company that filed 90 days late (filing_lag_days=90) but the trade_date is
    # exactly its filing date -- information age should be 0, not artificially "stale"
    # just because the underlying fiscal data is old.
    dsf = pd.Series([90.0])   # trade_date is 90 days past the fiscal reference date
    lag = pd.Series([90.0])   # ...because the filing itself was 90 days late
    out = days_since_filing(dsf, lag)
    ok = abs(float(out.iloc[0])) < TOL
    print_check("days_since_filing: a late filer's disclosure DAY has age 0, not the fiscal lag",
                ok, str(out.tolist()))
    return passed + ok, failed + (not ok)


def test_freshness_factor_at_zero_is_one(passed, failed):
    out = freshness_factor(pd.Series([0.0]), tau=45.0)
    ok = abs(float(out.iloc[0]) - 1.0) < TOL
    print_check("freshness_factor: age=0 -> factor=1.0 (brand new data)", ok, str(out.tolist()))
    return passed + ok, failed + (not ok)


def test_freshness_factor_at_halflife_tau(passed, failed):
    out = freshness_factor(pd.Series([45.0]), tau=45.0)
    ok = abs(float(out.iloc[0]) - np.exp(-1.0)) < TOL
    print_check("freshness_factor: age=tau -> factor=exp(-1) exactly", ok, str(out.tolist()))
    return passed + ok, failed + (not ok)


def test_freshness_factor_monotone_decreasing(passed, failed):
    out = freshness_factor(pd.Series([0.0, 10.0, 30.0, 90.0]), tau=45.0)
    ok = bool((out.diff().dropna() < 0).all())
    print_check("freshness_factor: strictly decreasing in age (day 5-25 gap >> day 70-90 gap)",
                ok, str(out.tolist()))
    return passed + ok, failed + (not ok)


def test_freshness_factor_negative_age_clipped(passed, failed):
    # A defensive guard, not an expected real input: negative age should never blow up
    # the factor above 1.0.
    out = freshness_factor(pd.Series([-10.0]), tau=45.0)
    ok = abs(float(out.iloc[0]) - 1.0) < TOL
    print_check("freshness_factor: negative age clipped to 0 -> factor capped at 1.0", ok, str(out.tolist()))
    return passed + ok, failed + (not ok)


def main() -> int:
    print_header("h_series/features.py")
    passed = failed = 0
    for test_fn in [
        test_days_since_filing_cancels_reference_date,
        test_days_since_filing_late_filer_not_penalized_twice,
        test_freshness_factor_at_zero_is_one,
        test_freshness_factor_at_halflife_tau,
        test_freshness_factor_monotone_decreasing,
        test_freshness_factor_negative_age_clipped,
    ]:
        passed, failed = test_fn(passed, failed)
    print_section_end(passed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
