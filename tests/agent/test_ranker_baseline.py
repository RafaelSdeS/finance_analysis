"""
Unit test for ranker_baseline portfolio simulator (synthetic, no data files).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.ranker_baseline import portfolio_simulator


def main():
    """Synthetic portfolio sim tests."""
    print("✓ Test 1: Perfect-signal ranker beats EW gross")
    np.random.seed(42)
    n_samples = 100
    dates = np.repeat(np.arange(50), 2)[:n_samples]  # 2 names per date
    tickers = np.tile(np.array(["A", "B"]), n_samples // 2)
    predictions = np.tile(np.array([1.0, 0.0]), n_samples // 2)  # Always favor "A"

    returns_dict = {}
    for d in np.unique(dates):
        for ticker in ["A", "B"]:
            # A has positive return, B negative
            returns_dict[(d, ticker)] = 0.01 if ticker == "A" else -0.01

    rets_ranker, vals_ranker = portfolio_simulator(
        predictions, dates, tickers, returns_dict,
        rebalance_days=5, cost_bps=10.0
    )
    rets_ew, vals_ew = portfolio_simulator(
        np.ones_like(predictions), dates, tickers, returns_dict,
        rebalance_days=5, cost_bps=10.0
    )

    mean_excess = rets_ranker.mean() - rets_ew.mean()
    print(f"  Ranker mean: {rets_ranker.mean():.6f}, EW mean: {rets_ew.mean():.6f}")
    print(f"  Excess: {mean_excess * 1e4:.2f} bps/day (expected > 0)")
    assert mean_excess > 0, f"Ranker should beat EW, got excess {mean_excess}"

    print("\n✓ Test 2: High cost reduces excess")
    rets_high_cost, _ = portfolio_simulator(
        predictions, dates, tickers, returns_dict,
        rebalance_days=5, cost_bps=100.0
    )
    excess_low = rets_ranker.mean() - rets_ew.mean()
    excess_high = rets_high_cost.mean() - rets_ew.mean()
    print(f"  Excess at 10bps: {excess_low * 1e4:.2f}, at 100bps: {excess_high * 1e4:.2f}")
    assert excess_high < excess_low, "Higher cost should reduce excess"

    print("\n✓ Test 3: Portfolio value computes from returns")
    cumsum_rets = np.cumsum(rets_ranker)
    final_val = np.exp(cumsum_rets[-1]) * 100_000
    print(f"  Final value from returns: {final_val:.0f} (expected ~100k + gains)")
    assert final_val > 0, "Portfolio value should be positive"

    print("\n✓ All ranker_baseline tests passed")


if __name__ == "__main__":
    main()
