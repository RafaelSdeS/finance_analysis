"""
Unit tests for supervised_strategy.py (synthetic data, no data files).

DIAGNOSIS_PLAN.md Phase 1.1: this is the ceiling any RL work has to beat, so its own
mechanics need to be trustworthy before its verdict is. Covers: select_topk_weights'
invariants (sums to 1, respects cap, only active names), the turnover band's actual
behavior (no cost when the top-k set is unchanged, real cost when it changes), and a
no-signal degeneracy check (random scores should behave like equal-weight, not manufacture
returns from nothing).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.supervised_strategy import backtest_topk_strategy, select_topk_weights


def main() -> None:
    print("Test 1: select_topk_weights invariants")
    scores = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
    active = np.array([True, True, True, True, True])
    w = select_topk_weights(scores, active, k=2, cap=1.0)
    assert np.isclose(w.sum(), 1.0), f"weights must sum to 1, got {w.sum()}"
    assert (w > 0).sum() == 2, f"expected exactly 2 nonzero weights, got {(w > 0).sum()}"
    assert set(np.where(w > 0)[0]) == {0, 2}, "expected the top-2 scores (indices 0, 2) selected"
    print(f"  top-2 of 5: {w}")

    print("\nTest 2: select_topk_weights respects the cap (widens count instead of overflowing)")
    w_capped = select_topk_weights(scores, active, k=2, cap=0.30)  # 1/2=0.5 > cap=0.30 -> widen to ceil(1/0.3)=4
    assert w_capped.max() <= 0.30 + 1e-9, f"no weight should exceed the cap, got max={w_capped.max()}"
    assert np.isclose(w_capped.sum(), 1.0)
    print(f"  k=2 with cap=0.30 widens to {(w_capped > 0).sum()} names: {w_capped}")

    print("\nTest 3: select_topk_weights only assigns weight to active names")
    active_partial = np.array([True, False, True, False, True])
    w_partial = select_topk_weights(scores, active_partial, k=2, cap=1.0)
    assert (w_partial[~active_partial] == 0).all(), "inactive names must never receive weight"
    assert set(np.where(w_partial > 0)[0]) <= set(np.where(active_partial)[0])
    print(f"  active-only selection: {w_partial}")

    print("\nTest 4: no active names -> all-zero vector (not a crash)")
    w_none = select_topk_weights(scores, np.zeros(5, dtype=bool), k=2, cap=1.0)
    assert (w_none == 0).all()
    print("  ✓ all-zero, no crash")

    print("\nTest 5 (problem/solution — turnover band): no cost when the top-k SET is "
          "unchanged; real cost when it changes")
    # 5 tickers, 3 rebalance windows of 21 days each (D = 3*21+1 so the last decision has a
    # valid next-day return). Ticker 0 and 1 are always the top-2 by score (set NEVER
    # changes across all 3 decisions) -- with the turnover band, only the FIRST decision
    # (a real "discovery" trade from nothing) should incur cost; decisions 2 and 3 should
    # be free, since the strategy already holds the right names.
    rebal_days = 21
    D = rebal_days * 3 + 1
    N = 5
    active6 = np.ones((D, N), dtype=bool)
    scores6 = np.tile(np.array([5.0, 4.0, 3.0, 2.0, 1.0]), (D, 1))  # ticker 0,1 always on top
    rng = np.random.default_rng(0)
    simple_rets6 = rng.normal(scale=0.01, size=(D, N))

    res_band = backtest_topk_strategy(scores6, active6, simple_rets6, k=2, cap=1.0,
                                       cost_bps=10.0, rebalance_interval_days=rebal_days, turnover_band=True)
    res_noband = backtest_topk_strategy(scores6, active6, simple_rets6, k=2, cap=1.0,
                                         cost_bps=10.0, rebalance_interval_days=rebal_days, turnover_band=False)

    # Reconstruct per-rebalance transaction cost from the day-1 return of each hold period
    # by comparing pre-cost vs realized return isn't directly exposed, so instead verify
    # via the more direct signal: rebal_weights should be IDENTICAL across all 3 decisions
    # under the band (same target held throughout), which by construction of the backtest
    # loop (retarget=False -> no cost applied) proves no cost was charged after decision 1.
    assert len(res_band["rebal_weights"]) == 3, f"expected 3 rebalance decisions, got {len(res_band['rebal_weights'])}"
    for i in range(3):
        assert set(np.where(res_band["rebal_weights"][i] > 0)[0]) == {0, 1}, (
            f"decision {i}: expected {{0,1}} always selected, got {np.where(res_band['rebal_weights'][i] > 0)[0]}"
        )
    print("  ✓ turnover-band strategy holds the same {0,1} target across all 3 decisions")

    # Check cost DIRECTLY rather than total P&L: periodic reset-to-equal-weight has its own
    # "rebalancing premium" effect on returns (independent of cost, and can go either way on
    # random data), so comparing cumulative return would conflate two different mechanisms.
    # Cost is unambiguous: band should charge only on the first ("discovery") decision;
    # no-band resets to a fresh 1/k target (paying cost) on literally every decision, even
    # though the SET never changes.
    assert res_band["rebal_costs"][0] > 0, "the first allocation is a real trade from nothing and must cost something"
    assert np.allclose(res_band["rebal_costs"][1:], 0.0), (
        f"decisions 2+ hold the SAME set under the band and must be free, got {res_band['rebal_costs'][1:]}"
    )
    assert (res_noband["rebal_costs"] > 0).all(), (
        f"without the band, every decision resets to a fresh target and must cost something, "
        f"got {res_noband['rebal_costs']}"
    )
    assert res_band["rebal_costs"].sum() < res_noband["rebal_costs"].sum(), (
        "the band must charge strictly less total cost than always resetting"
    )
    print(f"  ✓ band costs: {res_band['rebal_costs']} (only decision 1 costs anything)")
    print(f"  ✓ no-band costs: {res_noband['rebal_costs']} (every decision costs something)")

    print("\nTest 6: rank change beyond the band DOES trigger a trade")
    # Same setup, but tickers 0/1 swap with 2/3 in the second window -- the SET must change.
    scores_shift = scores6.copy()
    scores_shift[rebal_days:2 * rebal_days] = np.array([1.0, 2.0, 5.0, 4.0, 3.0])  # now 2,3 on top
    res_shift = backtest_topk_strategy(scores_shift, active6, simple_rets6, k=2, cap=1.0,
                                        cost_bps=10.0, rebalance_interval_days=rebal_days, turnover_band=True)
    sets = [set(np.where(w > 0)[0]) for w in res_shift["rebal_weights"]]
    assert sets[0] == {0, 1} and sets[1] == {2, 3}, f"expected a genuine set change, got {sets}"
    print(f"  ✓ decisions: {sets} — genuine set change correctly triggers a new target")

    print("\nTest 7 (no-signal degeneracy): random scores behave like equal-weight-of-active, "
          "not a manufactured edge")
    D2, N2 = 21 * 20 + 1, 30  # ~20 rebalances, 30 names -- enough for the means to converge
    active7 = np.ones((D2, N2), dtype=bool)
    rng2 = np.random.default_rng(7)
    simple_rets7 = rng2.normal(scale=0.015, size=(D2, N2))
    scores7 = rng2.normal(size=(D2, N2))  # pure noise, independent of returns

    res7 = backtest_topk_strategy(scores7, active7, simple_rets7, k=10, cap=0.10,
                                   cost_bps=10.0, rebalance_interval_days=21, turnover_band=True)
    strategy_mean = res7["daily_log_returns"].mean()
    # EW-of-all-active daily log return, for comparison (same population, no selection)
    ew_daily = np.log1p(simple_rets7[1:].mean(axis=1))
    ew_mean = ew_daily.mean()
    print(f"  strategy mean daily log-return: {strategy_mean:.6f}  |  EW-of-all mean: {ew_mean:.6f}")
    assert abs(strategy_mean - ew_mean) < 0.001, (
        f"with zero real signal, a top-k subset's mean return should track full EW closely, "
        f"got strategy={strategy_mean:.6f} vs EW={ew_mean:.6f}"
    )
    print("  ✓ no-signal strategy tracks equal-weight (doesn't manufacture an edge from noise)")

    print("\nTest 8: portfolio invariants hold throughout")
    assert np.allclose(res7["daily_weights"].sum(axis=1), 1.0, atol=1e-6), "daily weights must always sum to 1"
    # The cap is a REBALANCE-TIME constraint (matching env.py exactly): a name's weight is
    # capped when a target is SET, but is then allowed to drift with realized returns during
    # the hold, same as env.py's own step() (target capped once via action_to_weights, then
    # drifts unconstrained). Check the cap where it's actually enforced: at rebal_weights.
    assert res7["rebal_weights"].max() <= 0.10 + 1e-9, "no REBALANCE-TIME weight should exceed the cap"
    assert res7["daily_weights"].max() > 0.10, (
        "sanity: with 20 rebalances of real drift, at least one day should show SOME "
        "weight drifted above the 0.10 rebalance-time target (else the drift math is a no-op)"
    )
    print(f"  ✓ daily weights always sum to 1; rebal-time weights respect the cap "
          f"(max drifted daily weight {res7['daily_weights'].max():.4f} > 0.10, as expected)")

    print("\n✓ All supervised_strategy tests passed")


if __name__ == "__main__":
    main()
