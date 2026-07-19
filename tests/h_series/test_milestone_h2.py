"""
Test: h_series/milestone_h2.py's Construction A weight math -- hand-computed
multiplicative/additive formulas, the iterative cap-enforcement loop (a
single clip-then-renormalize pass provably re-violates the cap; the loop
must not), the no-trade band's dependence on the ACTUAL drifted portfolio
(never a remembered prior target -- the weight_fn contract only ever exposes
w_drift), and the permutation-null helper's whole-cross-section swap
(never a within-date ticker reshuffle, which would break the
correlation-preserving property the ablation relies on).

Synthetic fixtures, deterministic. No parquet IO.

Run from project root:
    python tests/h_series/test_milestone_h2.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import src.h_series.milestone_h2 as h2  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402


def test_construction_a_multiplicative_hand_check(passed, failed):
    anchor = np.array([0.5, 0.3, 0.2])
    score = np.array([0.4, -0.2, 0.0])
    gamma = 1.0
    raw = anchor * (1.0 + gamma * score)
    expected = raw / raw.sum()
    got = h2.construction_a_multiplicative(anchor, score, gamma, max_weight=1.0)
    ok = bool(np.allclose(got, expected, atol=1e-12))
    print_check("construction_a_multiplicative: matches hand-computed anchor*(1+gamma*score), renormalized",
                ok, f"got={got}, expected={expected}")
    return passed + ok, failed + (not ok)


def test_construction_a_multiplicative_clips_negative(passed, failed):
    anchor = np.array([0.5, 0.3, 0.2])
    score = np.array([-3.0, 0.0, 0.0])  # gamma*score << -1 -> raw negative at index 0
    gamma = 1.0
    raw = anchor * (1.0 + gamma * score)
    expected_unnorm = np.maximum(raw, 0.0)
    expected = expected_unnorm / expected_unnorm.sum()
    got = h2.construction_a_multiplicative(anchor, score, gamma, max_weight=1.0)
    ok = bool(np.allclose(got, expected, atol=1e-12)) and got[0] == 0.0
    print_check("construction_a_multiplicative: negative raw weight clipped to exactly 0, mass redistributed",
                ok, f"got={got}")
    return passed + ok, failed + (not ok)


def test_construction_a_additive_hand_check(passed, failed):
    anchor = np.array([0.4, 0.35, 0.25])
    score = np.array([0.3, -0.1, -0.2])  # mean == 0.0 exactly -> active sums to 0 -> raw sums to 1
    gamma_add = 0.5
    centered = score - score.mean()
    denom = np.sum(np.abs(score))
    active = gamma_add * centered / denom
    expected = anchor + active
    got = h2.construction_a_additive(anchor, score, gamma_add, max_weight=1.0)
    ok = bool(np.allclose(got, expected, atol=1e-12))
    print_check("construction_a_additive: matches hand-computed anchor + gamma*(score-mean)/sum|score|",
                ok, f"got={got}, expected={expected}")
    return passed + ok, failed + (not ok)


def test_enforce_max_weight_fixes_single_pass_violation(passed, failed):
    raw = np.array([0.35, 0.35, 0.2, 0.1])
    max_weight = 0.3

    naive = np.clip(raw, 0, max_weight)
    naive = naive / naive.sum()
    naive_violates = bool(np.any(naive > max_weight + 1e-9))
    print_check("fixture sanity: a single clip-then-renormalize pass provably re-violates the cap",
                naive_violates, f"naive weights={naive.round(4).tolist()}")

    fixed = h2.enforce_max_weight(raw, max_weight)
    in_bounds = bool(np.all(fixed <= max_weight + 1e-9))
    sums_to_one = bool(np.isclose(fixed.sum(), 1.0, atol=1e-9))
    ok = naive_violates and in_bounds and sums_to_one
    print_check("enforce_max_weight: iterative loop satisfies the cap AND sums to 1 on the same input",
                ok, f"fixed weights={fixed.round(4).tolist()}")
    return passed + ok, failed + (not ok)


def test_no_trade_band_uses_actual_drift_not_prior_target(passed, failed):
    target = np.array([0.0, 0.5, 0.3, 0.2])
    fn = h2.make_precomputed_weight_fn({5: target}, no_trade_band=0.05)

    close_drift = np.array([0.0, 0.49, 0.30, 0.21])     # L1 dist ~0.02 < band -> skip
    far_drift = np.array([0.0, 0.3, 0.3, 0.4])           # L1 dist ~0.4 >= band -> trade
    w_prev_irrelevant = np.array([0.9, 0.1, 0.0, 0.0])   # must be ignored -- not w_drift

    out_skip = fn(5, w_prev_irrelevant, close_drift, None)
    out_trade = fn(5, w_prev_irrelevant, far_drift, None)
    out_no_decision = fn(6, w_prev_irrelevant, far_drift, None)

    skip_ok = bool(np.array_equal(out_skip, close_drift))
    trade_ok = bool(np.array_equal(out_trade, target))
    passthrough_ok = bool(np.array_equal(out_no_decision, far_drift))
    ok = skip_ok and trade_ok and passthrough_ok
    print_check("make_precomputed_weight_fn: band check + rebalance decision depend ONLY on the "
                "actual w_drift argument (never w_prev), and non-decision days always drift",
                ok, f"skip_ok={skip_ok}, trade_ok={trade_ok}, passthrough_ok={passthrough_ok}")
    return passed + ok, failed + (not ok)


def test_permutation_null_preserves_cross_section(passed, failed):
    """With only 2 dates, a single rng.permutation draw is exactly one of two
    outcomes: identity (each date keeps its own score cross-section) or swap
    (d0 uses d1's whole score vector, and vice versa). If the implementation
    instead reshuffled scores at the ticker level WITHIN a date, the result
    would match neither hand-computed outcome."""
    d0, d1 = pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")
    tickers = ["A", "B", "C", "D"]
    scores = {d0: [0.4, -0.1, 0.2, -0.3], d1: [-0.2, 0.3, 0.1, -0.1]}
    fwd = {d0: [0.05, -0.02, 0.01, 0.03], d1: [0.02, 0.04, -0.01, 0.00]}
    rows = []
    for d in (d0, d1):
        for i, tk in enumerate(tickers):
            rows.append({"decision_date": d, "ticker": tk, "score": scores[d][i],
                         "anchor_capw": 0.25, "fwd": fwd[d][i]})
    panel = pd.DataFrame(rows)

    null_irs = h2.permutation_null(panel, "score", "anchor_capw", 1.0,
                                    h2.construction_a_multiplicative, "fwd", n_draws=1, seed=0)

    def _ir_for(score_map: dict) -> float:
        active = {}
        for d in (d0, d1):
            g = panel[panel["decision_date"] == d]
            w = h2.construction_a_multiplicative(g["anchor_capw"].to_numpy(), np.array(score_map[d]), 1.0)
            active[d] = float(np.dot(w, g["fwd"].to_numpy()))
        return h2.ir_stats(pd.Series(active))["ir_annualized"]

    ir_identity = _ir_for(scores)
    ir_swap = _ir_for({d0: scores[d1], d1: scores[d0]})

    ok = bool(np.isclose(null_irs[0], ir_identity, atol=1e-9) or np.isclose(null_irs[0], ir_swap, atol=1e-9))
    print_check("permutation_null: a draw matches one of the two whole-cross-section swap outcomes "
                "(never a within-date ticker reshuffle)", ok,
                f"draw={null_irs[0]:.6f}, identity={ir_identity:.6f}, swap={ir_swap:.6f}")
    return passed + ok, failed + (not ok)


def main() -> int:
    print_header("h_series/milestone_h2.py (Construction A)")
    passed = failed = 0
    for test_fn in [
        test_construction_a_multiplicative_hand_check,
        test_construction_a_multiplicative_clips_negative,
        test_construction_a_additive_hand_check,
        test_enforce_max_weight_fixes_single_pass_violation,
        test_no_trade_band_uses_actual_drift_not_prior_target,
        test_permutation_null_preserves_cross_section,
    ]:
        passed, failed = test_fn(passed, failed)
    print_section_end(passed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
