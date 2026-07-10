"""
Walk-forward top-k equal-weight strategy (DIAGNOSIS_PLAN.md Phase 1.1): the best simple,
NON-RL portfolio buildable from this project's features, evaluated honestly on the
point-in-time universe. This is the ceiling any RL work (Phase 2) has to beat to justify
its complexity.

Deliberately does NOT reuse PortfolioEnv. PortfolioEnv's iterative cap-redistribution
machinery exists to handle arbitrary softmax-derived weight vectors, but a top-k
equal-weight target with k >= 10 never actually needs it: 1/k <= 0.10 = the project's
max_position_weight for every k this project's sweep uses. A direct, independent
implementation is simpler here and, as a side benefit, is a second, independently-derived
check on the portfolio math env.py already implements — matching numbers on any case where
both apply would be reassuring, and this module has zero dependency on env.py.
"""

import numpy as np


def select_topk_weights(scores: np.ndarray, active: np.ndarray, k: int, cap: float = 1.0) -> np.ndarray:
    """Equal-weight target over the top-k active names by score.

    If 1/k would exceed cap, widens the selection to ceil(1/cap) names — since every
    selected name gets the SAME weight, a uniform cap violation only needs more names,
    not iterative redistribution (unlike env.py's general softmax case).

    Returns an all-zero vector if no active names (caller decides what that means for CASH).
    """
    active = np.asarray(active, dtype=bool)
    n = len(active)
    active_idx = np.where(active)[0]
    if len(active_idx) == 0:
        return np.zeros(n)

    eff_k = min(k, len(active_idx))
    n_names = max(eff_k, int(np.ceil(1.0 / cap))) if cap > 0 else eff_k
    n_names = min(n_names, len(active_idx))

    top_local = np.argsort(scores[active_idx])[-n_names:]
    top_idx = active_idx[top_local]

    weights = np.zeros(n)
    weights[top_idx] = 1.0 / len(top_idx)
    return weights


def backtest_topk_strategy(
    scores: np.ndarray,          # [D, N] predicted scores (ignored on inactive names)
    active: np.ndarray,          # [D, N] bool, True where tradable that day (CASH excluded by caller)
    simple_returns: np.ndarray,  # [D, N] next-day simple return, 0 where inactive (env.py convention)
    k: int,
    cap: float = 0.10,
    cost_bps: float = 10.0,
    rebalance_interval_days: int = 21,
    turnover_band: bool = True,
) -> dict:
    """Simulate a top-k equal-weight strategy with periodic rebalancing and transaction
    costs, optionally skipping retargeting when the top-k SET hasn't changed.

    D = number of trading days; the last day has no next-day return, so decisions run
    through D-1. Mirrors env.py's step() semantics exactly (decide using day-t info, hold
    for `rebalance_interval_days`, one-time cost on day 1 of the hold, weights drift daily
    with realized per-name returns): a strategy family PortfolioEnv could also score, just
    computed independently here.

    turnover_band: if True (default), a rebalance decision whose top-k SET is identical to
    the currently-held set is a no-op (no cost, no reset to a fresh 1/k target) — this is
    the "band" the plan asks for: retarget only on a genuine change in WHICH names are
    selected, not on every rank wobble that leaves the set unchanged. If False, every
    decision resets to a fresh equal-weight target (paying turnover even when the set is
    unchanged, since drifted weights are never exactly 1/k). `weights` starts all-zero
    ("100% cash", matching env.py's own reset convention), so the very first allocation
    always costs something too — a real deploy-from-cash trade, not a free discovery.
    """
    D, N = scores.shape
    n_steps = D - 1
    weights = np.zeros(N)
    target_set: frozenset | None = None

    daily_log_rets, daily_weights = [], []
    rebal_weights, rebal_fwd_rets, rebal_costs, rebal_t = [], [], [], []

    t = 0
    while t < n_steps:
        n_days = min(rebalance_interval_days, n_steps - t)

        new_weights = select_topk_weights(scores[t], active[t], k, cap)
        new_set = frozenset(np.where(new_weights > 0)[0].tolist())

        retarget = (not turnover_band) or (new_set != target_set)
        cost = 0.0
        if retarget:
            # `weights` starts all-zero ("100% cash", matching env.py's own reset
            # convention), so the very first allocation IS a real deploy-from-cash trade
            # and costs something too -- not a free "discovery" (env.py: "the first
            # allocation is scored as a real deploy-or-wait decision, not forced to ~0").
            cost = float(np.abs(new_weights - weights).sum()) * (cost_bps / 10_000)
            w = new_weights.copy()
            target_set = new_set
        else:
            w = weights.copy()

        rebal_weights.append(w.copy())
        rebal_costs.append(cost)
        rebal_t.append(t)
        window_simple = simple_returns[t + 1: t + n_days + 1]  # [n_days, N]
        rebal_fwd_rets.append(np.prod(1.0 + window_simple, axis=0) - 1.0)

        for day_offset in range(n_days):
            r_vec = simple_returns[t + 1 + day_offset]
            r_p_pre_cost = float(np.dot(w, r_vec))
            r_p = r_p_pre_cost - cost if day_offset == 0 else r_p_pre_cost
            r_p = max(r_p, -0.9999)
            daily_log_rets.append(float(np.log1p(r_p)))

            if r_p_pre_cost > -0.9999:
                w_new = w * (1.0 + r_vec) / (1.0 + r_p_pre_cost)
                if w_new.sum() > 0:
                    w = w_new / w_new.sum()
            daily_weights.append(w.copy())

        weights = w
        t += n_days

    return {
        "daily_log_returns": np.asarray(daily_log_rets, dtype=np.float64),
        "daily_weights": np.asarray(daily_weights, dtype=np.float64),
        "rebal_weights": np.asarray(rebal_weights, dtype=np.float64),
        "rebal_fwd_rets": np.asarray(rebal_fwd_rets, dtype=np.float64),
        "rebal_costs": np.asarray(rebal_costs, dtype=np.float64),
        "rebal_t": np.asarray(rebal_t, dtype=np.int64),  # day-index (into the input arrays) of each decision
    }
