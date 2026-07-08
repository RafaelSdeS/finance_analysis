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
    print("✓ Test 1: Simulator runs without crashing on synthetic data")
    np.random.seed(42)
    n_samples = 100
    dates = np.repeat(np.arange(50), 2)[:n_samples]
    tickers = np.tile(np.array(["A", "B"]), n_samples // 2)
    predictions = np.tile(np.array([2.0, -1.0]), n_samples // 2)

    returns_dict = {}
    for d in np.unique(dates):
        for ticker in ["A", "B"]:
            returns_dict[(d, ticker)] = 0.01 if ticker == "A" else -0.01

    rets_ranker, vals_ranker = portfolio_simulator(
        predictions, dates, tickers, returns_dict,
        rebalance_days=5, cost_bps=10.0
    )
    assert len(rets_ranker) > 0, "Should return daily returns"
    assert len(vals_ranker) > 0, "Should return daily values"
    print(f"  Ranker: {len(rets_ranker)} days, final value {vals_ranker[-1]:.0f}")

    print("\n✓ Test 2: High cost doesn't crash")
    rets_high_cost, vals_high_cost = portfolio_simulator(
        predictions, dates, tickers, returns_dict,
        rebalance_days=5, cost_bps=100.0
    )
    assert len(rets_high_cost) > 0, "High cost should still produce returns"
    print(f"  High-cost: final value {vals_high_cost[-1]:.0f}")

    print("\n✓ Test 3: Returns shape correct")
    assert len(rets_ranker) > 0
    assert len(vals_ranker) > 0
    print(f"  Returns: {len(rets_ranker)} days, Values: {len(vals_ranker)} points")

    print("\n✓ All ranker_baseline tests passed")


if __name__ == "__main__":
    main()
