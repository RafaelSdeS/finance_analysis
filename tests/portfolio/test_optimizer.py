"""
test_optimizer.py -- checks solve() against an analytic no-trade band
(proposal §5's central claim: the L1 term's subgradient creates an *exact*
no-trade region, not an approximate one) on a 1-risky-asset toy where the
band is hand-derivable, plus the forced-exit hard constraint (P6).

For a single risky asset + cash, cash's alpha=0 and w>=0/w<=w_max non-binding,
the KKT/subgradient condition at w=w_prev is:
    |alpha - lam*Sigma_11*w_prev| <= c1  =>  no trade
Outside that band it moves to w* = (alpha -/+ c1) / (lam*Sigma_11).

Fast group (synthetic only, cvxpy is deterministic here). Run:
    python tests/portfolio/test_optimizer.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.portfolio.optimizer import solve  # noqa: E402
from src.portfolio.risk import add_cash_row_col  # noqa: E402
from tests.test_utils import print_check, print_header, print_section_end  # noqa: E402

LAM = 1.0
SIGMA_11 = 0.04
W_PREV = 0.5
C1 = 0.01


def _run(alpha_risky: float):
    alpha = pd.Series({"A": alpha_risky, "cash": 0.0})
    sigma = add_cash_row_col(pd.DataFrame([[SIGMA_11]], index=["A"], columns=["A"]))
    w_prev = pd.Series({"A": W_PREV, "cash": 1 - W_PREV})
    w = solve(alpha, sigma, w_prev, c1=C1, lam=LAM, w_max=1.0)
    return w["A"]


def main():
    print_header("test_optimizer")
    passed = failed = 0

    # Case A: alpha inside the band [lam*Sigma*w_prev - c1, lam*Sigma*w_prev + c1]
    # = [0.02-0.01, 0.02+0.01] = [0.01, 0.03] -- no trade.
    w_a = _run(alpha_risky=0.02)
    ok = np.isclose(w_a, W_PREV, atol=1e-4)
    print_check("alpha inside the no-trade band -> no trade (w unchanged)", bool(ok),
                f"w_prev={W_PREV}, w_new={w_a:.5f}")
    passed, failed = passed + ok, failed + (not ok)

    # Case B: alpha above the band -> trades up to (alpha - c1) / (lam*Sigma_11)
    w_b = _run(alpha_risky=0.04)
    expected_b = (0.04 - C1) / (LAM * SIGMA_11)
    ok = np.isclose(w_b, expected_b, atol=1e-4)
    print_check("alpha above the band -> trades up to the analytic target", bool(ok),
                f"expected={expected_b:.5f}, got={w_b:.5f}")
    passed, failed = passed + ok, failed + (not ok)

    # Case C: alpha below the band -> trades down to (alpha + c1) / (lam*Sigma_11)
    w_c = _run(alpha_risky=0.005)
    expected_c = (0.005 + C1) / (LAM * SIGMA_11)
    ok = np.isclose(w_c, expected_c, atol=1e-4)
    print_check("alpha below the band -> trades down to the analytic target", bool(ok),
                f"expected={expected_c:.5f}, got={w_c:.5f}")
    passed, failed = passed + ok, failed + (not ok)

    # Forced exit (P6): B is held from a prior period but is no longer in the
    # current universe (absent from alpha) -- must be hard-pinned to 0, not
    # just "unattractive".
    alpha = pd.Series({"A": 0.5, "cash": 0.0})  # huge alpha on A, to see if it'd
                                                  # ever "want" to also hold B
    sigma = add_cash_row_col(pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.04]], index=["A", "B"], columns=["A", "B"]
    ))
    # drop B from sigma's *investable* view -- solve() only takes alpha's
    # index as the current universe, so pass sigma restricted to A/cash;
    # B only appears in w_prev.
    sigma_current = sigma.loc[["A", "cash"], ["A", "cash"]]
    alpha_current = alpha  # already only A/cash
    w_prev = pd.Series({"A": 0.0, "B": 0.3, "cash": 0.7})
    w = solve(alpha_current, sigma_current, w_prev, c1=0.01, lam=1.0, w_max=1.0)
    forced_ok = np.isclose(w["B"], 0.0, atol=1e-6) and "B" in w.index
    print_check("a name absent from alpha but present in w_prev is hard-pinned to 0",
                bool(forced_ok), f"w_B={w.get('B')}")
    passed, failed = passed + forced_ok, failed + (not forced_ok)

    sums_to_one_ok = np.isclose(w.sum(), 1.0, atol=1e-6)
    print_check("weights still sum to 1 with a forced exit in the mix", bool(sums_to_one_ok))
    passed, failed = passed + sums_to_one_ok, failed + (not sums_to_one_ok)

    print_section_end(passed, failed)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
