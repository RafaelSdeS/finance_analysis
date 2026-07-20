"""
blend.py — H3 Design §c: fitted blend between the risk-based anchor
(anchor.py) and the combination layer's view (composite.select_model),

    w_posterior = w_anchor + kappa * (w_view - w_anchor)

kappa is DERIVED from the winning model's own pre-2024 out-of-fold mean
Spearman IC (select_model's diagnostics), not grid-searched against the
reported backtest IR -- that circularity is exactly what this replaces H2's
hand-picked gamma-tilt to avoid.

kappa = sigmoid(K_SLOPE * (IC_OOF - IC_MIN)), with two calibration points,
both reused from already-established project numbers rather than invented:
kappa=0.5 at IC_OOF=IC_MIN (true by construction of the sigmoid's center,
IC_MIN = H0's own pre-registered minimum-detectable mean IC at k=21,
H0_FINDINGS.md) and kappa=KAPPA_AT_BEST_SURVIVOR at IC_OOF=IC_BEST_SURVIVOR
(H1's single strongest real survivor IC, momentum_vs_market_12m
sector-neutral, H1_FINDINGS.md). K_SLOPE is fixed by solving those two
points before any backtest is run, exactly like IC_MIN itself -- not a free
knob left to taste. See H3_PORTFOLIO_CONSTRUCTION_PLAN.md Design §c.
"""

import numpy as np

IC_MIN = 0.0300            # H0's pre-registered min-detectable mean IC, k=21 (H0_FINDINGS.md)
IC_BEST_SURVIVOR = 0.088   # H1's strongest real survivor IC, momentum_vs_market_12m, sector-neutral, k=21
KAPPA_AT_BEST_SURVIVOR = 0.9
K_SLOPE = np.log(9) / (IC_BEST_SURVIVOR - IC_MIN)  # solves sigmoid(k*(0.088-0.0300))=0.9 -> k ~= 37.9


def kappa_from_ic(ic_oof: float) -> float:
    """kappa = sigmoid(K_SLOPE * (ic_oof - IC_MIN)) in (0, 1) -- 0.5 exactly
    at ic_oof == IC_MIN by construction of the sigmoid's center; -> 1 well
    above it, -> 0 well below (a model with no measurable OOF skill trusts
    the anchor almost entirely)."""
    return float(1.0 / (1.0 + np.exp(-K_SLOPE * (ic_oof - IC_MIN))))


def view_weights_from_score(score: np.ndarray) -> np.ndarray:
    """Renormalizes the combination layer's per-ticker score onto the
    long-only simplex. Scores entering here are already cross-sectionally
    rank-normalized (stats.rank_normalize -- composite.select_model's
    stitched output) into [-0.5, 0.5] per date, so a +0.5 shift is exact,
    not a heuristic clip -- every score maps to a distinct nonnegative
    value before the sum-to-1 scaling."""
    shifted = np.clip(np.asarray(score, dtype=float) + 0.5, 0.0, 1.0)
    total = shifted.sum()
    if total <= 0:
        return np.full(len(score), 1.0 / len(score))
    return shifted / total


def blend_weights(w_anchor: np.ndarray, w_view: np.ndarray, kappa: float) -> np.ndarray:
    """w_posterior = w_anchor + kappa*(w_view - w_anchor). Both inputs must
    already be nonnegative and sum to 1 (simplex) over the same ticker
    ordering; the result stays on the simplex by construction (a convex
    combination of two simplex points, kappa in [0, 1]) but is NOT
    clipped/capped here -- the caller applies the project's standard
    max_weight cap (milestone_h2.enforce_max_weight) afterward, same as
    every other H-series weight construction."""
    w_anchor = np.asarray(w_anchor, dtype=float)
    w_view = np.asarray(w_view, dtype=float)
    return w_anchor + kappa * (w_view - w_anchor)


def _demo() -> None:
    """Runnable self-check: kappa's two calibration points reproduce
    KAPPA_AT_BEST_SURVIVOR and 0.5 exactly, kappa is monotone in IC, and a
    blended weight vector stays on the simplex for the same synthetic
    anchor/score pair across the full kappa range."""
    assert abs(kappa_from_ic(IC_MIN) - 0.5) < 1e-9
    assert abs(kappa_from_ic(IC_BEST_SURVIVOR) - KAPPA_AT_BEST_SURVIVOR) < 1e-9
    assert kappa_from_ic(IC_MIN - 0.05) < kappa_from_ic(IC_MIN) < kappa_from_ic(IC_BEST_SURVIVOR)

    rng = np.random.default_rng(0)
    n = 10
    w_anchor = rng.dirichlet(np.ones(n))
    score = rng.uniform(-0.5, 0.5, n)
    w_view = view_weights_from_score(score)
    assert abs(w_view.sum() - 1.0) < 1e-9
    assert (w_view >= 0).all()

    for ic_oof in (-0.1, 0.0, IC_MIN, 0.05, IC_BEST_SURVIVOR, 0.2):
        kappa = kappa_from_ic(ic_oof)
        assert 0.0 < kappa < 1.0
        w = blend_weights(w_anchor, w_view, kappa)
        assert abs(w.sum() - 1.0) < 1e-9, f"kappa={kappa}: blend left the simplex (sum={w.sum()})"
        assert (w >= -1e-12).all(), f"kappa={kappa}: blend produced a negative weight"

    print(f"blend.py self-check: OK (K_SLOPE={K_SLOPE:.3f})")


if __name__ == "__main__":
    _demo()
