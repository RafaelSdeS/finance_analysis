"""
test_metrics.py -- sanity checks for src/portfolio/metrics.py: each stat
against an independently-obvious hand computation, plus qualitative
sanity checks for the deflated Sharpe ratio (monotonic in n_trials, bounded
in [0, 1], distinguishes noise from a clear positive edge).

Fast group (synthetic only). Run: python tests/portfolio/test_metrics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.portfolio.metrics import (  # noqa: E402
    CRISIS_WINDOWS, annualized_return, deflated_sharpe_ratio, max_drawdown,
    regime_slice, sharpe_ratio, turnover_stats,
)
from tests.test_utils import print_check, print_header, print_section_end  # noqa: E402


def main():
    print_header("test_metrics")
    passed = failed = 0

    # annualized_return: constant daily r compounded over 252 days
    r = 0.002
    returns = pd.Series([r] * 252)
    expected = (1 + r) ** 252 - 1
    ok = np.isclose(annualized_return(returns), expected)
    print_check("annualized_return matches direct compounding", bool(ok))
    passed, failed = passed + ok, failed + (not ok)

    # sharpe_ratio: known mean/std
    returns2 = pd.Series([0.0, 0.002] * 126)  # mean=0.001, std computable directly
    expected_sr = (returns2.mean() / returns2.std(ddof=1)) * np.sqrt(252)
    ok = np.isclose(sharpe_ratio(returns2), expected_sr)
    print_check("sharpe_ratio matches direct mean/std formula", bool(ok))
    passed, failed = passed + ok, failed + (not ok)

    # sharpe_ratio: a self-vs-self diff has std at float64 noise floor (~1e-16),
    # not exactly 0 -- must read as NaN (degenerate), not an arbitrary huge/tiny
    # ratio from dividing noise by noise (found 2026-07-25: 100% CDI's
    # excess-over-CDI Sharpe printed a spurious 0.133 from exactly this).
    degenerate = pd.Series([0.0005 + 1e-17 * ((-1) ** i) for i in range(300)])
    ok = np.isnan(sharpe_ratio(degenerate))
    print_check("sharpe_ratio treats float-noise-floor std as degenerate (NaN)", bool(ok),
                f"got {sharpe_ratio(degenerate)}")
    passed, failed = passed + ok, failed + (not ok)

    # max_drawdown: known path
    curve = pd.Series([1.0, 1.1, 1.05, 0.9, 1.2])
    expected_dd = min(0.0, 1.05 / 1.1 - 1, 0.9 / 1.1 - 1, 0.0)
    ok = np.isclose(max_drawdown(curve), expected_dd)
    print_check("max_drawdown matches hand-computed path", bool(ok), f"got {max_drawdown(curve):.4f}")
    passed, failed = passed + ok, failed + (not ok)

    # deflated_sharpe_ratio: bounded, monotonic in n_trials, ranks signal above noise
    rng = np.random.default_rng(0)
    strong = pd.Series(rng.normal(0.002, 0.005, 500))   # clear positive edge
    noise = pd.Series(rng.normal(0.0, 0.01, 500))        # pure noise around 0 (this specific
                                                          # draw has a negative sample mean by
                                                          # chance -- that's fine, see below)

    dsr_strong = deflated_sharpe_ratio(strong, n_trials=1)
    dsr_noise = deflated_sharpe_ratio(noise, n_trials=1)
    bounded_ok = all(0.0 <= v <= 1.0 for v in (dsr_strong, dsr_noise))
    print_check("deflated_sharpe_ratio stays in [0, 1]", bool(bounded_ok))
    passed, failed = passed + bounded_ok, failed + (not bounded_ok)

    # Ranking, not an absolute band: a single noise draw can legitimately land
    # anywhere (its realized sample Sharpe isn't exactly 0), so the only
    # invariant that must hold regardless of the draw is strong > noise.
    ranks_ok = dsr_strong > dsr_noise
    print_check("ranks a clear positive edge above a noisy draw", bool(ranks_ok),
                f"strong={dsr_strong:.3f}, noise={dsr_noise:.3f}")
    passed, failed = passed + ranks_ok, failed + (not ranks_ok)

    # Deterministic (no sampling luck): a perfectly symmetric series has sample
    # mean and skew exactly 0, so the observed Sharpe is exactly 0 -- PSR
    # against a 0 benchmark must then be exactly neutral (0.5).
    zero_edge = pd.Series([0.01, -0.01] * 100)
    dsr_zero = deflated_sharpe_ratio(zero_edge, n_trials=1)
    neutral_ok = np.isclose(dsr_zero, 0.5, atol=1e-6)
    print_check("an exactly-zero observed Sharpe reads as neutral (0.5)", bool(neutral_ok),
                f"got {dsr_zero:.6f}")
    passed, failed = passed + neutral_ok, failed + (not neutral_ok)

    dsr_1trial = deflated_sharpe_ratio(strong, n_trials=1)
    dsr_100trial = deflated_sharpe_ratio(strong, n_trials=100)
    monotonic_ok = dsr_100trial <= dsr_1trial
    print_check("more trials (multiple-testing penalty) never raises the deflated Sharpe",
                bool(monotonic_ok), f"1 trial={dsr_1trial:.3f}, 100 trials={dsr_100trial:.3f}")
    passed, failed = passed + monotonic_ok, failed + (not monotonic_ok)

    # turnover_stats: known rebalance log
    log = pd.DataFrame({"turnover": [1.0, 0.0, 0.5, 0.0]})
    stats = turnover_stats(log, rebalances_per_year=4)
    expected_annual = log["turnover"].mean() * 4
    ok = np.isclose(stats["annual_turnover"], expected_annual)
    print_check("turnover_stats.annual_turnover matches direct mean*freq", bool(ok))
    passed, failed = passed + ok, failed + (not ok)
    ok = np.isclose(stats["no_trade_fraction"], 0.5)
    print_check("turnover_stats.no_trade_fraction counts the exact zero-turnover rebalances",
                bool(ok), f"got {stats['no_trade_fraction']}")
    passed, failed = passed + ok, failed + (not ok)

    # regime_slice: returns split cleanly around the SELIC median + a crisis window
    dates = pd.date_range("2008-08-01", periods=250, freq="D")
    returns3 = pd.Series(np.linspace(-0.01, 0.01, 250), index=dates)
    selic = pd.Series(np.where(np.arange(250) % 2 == 0, 0.05, 0.03), index=dates)
    slices = regime_slice(returns3, selic)
    split_ok = slices["high_selic"]["n_days"] + slices["low_selic"]["n_days"] == 250
    print_check("high/low SELIC slices partition all observations", bool(split_ok))
    passed, failed = passed + split_ok, failed + (not split_ok)

    start, end = CRISIS_WINDOWS["gfc_2008"]
    expected_gfc_days = returns3[(returns3.index >= start) & (returns3.index <= end)].shape[0]
    ok = slices["gfc_2008"]["n_days"] == expected_gfc_days
    print_check("gfc_2008 crisis window picks the right date range", bool(ok))
    passed, failed = passed + ok, failed + (not ok)

    print_section_end(passed, failed)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
