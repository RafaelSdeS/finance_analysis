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


def compute_alpha_waterfall(
    scores: np.ndarray,          # [D, N]
    active: np.ndarray,          # [D, N]
    simple_returns: np.ndarray,  # [D, N] next-day simple return, 0 if inactive
    k: int,
    cap: float = 0.10,
    cost_bps: float = 10.0,
    rebalance_interval_days: int = 21,
) -> dict:
    """5-stage alpha waterfall (DIAGNOSIS_PLAN.md Phase 1.2) isolating WHERE raw
    cross-sectional signal is destroyed by realistic portfolio construction. Every stage
    reports the mean PERIOD excess return vs equal-weight-of-active across all rebalance
    decisions, in the SAME units, so stages are directly comparable — monotone
    non-increasing is the expected (not per-sample guaranteed, but expected in aggregate)
    shape for a genuine signal being progressively constrained.

    Stages:
      1. long_short:       top-k long, bottom-k short, EW within each leg, no costs/cap —
                            the purest measure, captures both tails of the signal.
      2. long_only:        top-k long only (no short leg), EW, no costs/cap — removing the
                            short leg is itself a real constraint (most funds can't short).
      3. long_only_capped: same, weights capped at `cap` (select_topk_weights widens the
                            selection when 1/k > cap) — shows whether the position-size
                            constraint itself bites (use a small k to actually trigger it;
                            k >= ceil(1/cap) never will, by construction).
      4. plus_costs:       same, minus `cost_bps` transaction costs EVERY period (worst
                            case: no turnover band, a fresh target every decision).
      5. plus_holding:     the full realistic backtest (backtest_topk_strategy, turnover
                            band ON) — weights drift for `rebalance_interval_days` between
                            decisions instead of being marked-to-market instantly, and
                            costs are actually amortized via the band.

    Stages 1-4 are strictly nested degradations (each adds a friction with no offsetting
    benefit vs the last), so monotone non-increasing is the correct expectation there.
    Stage 5 is NOT guaranteed to sit below stage 4: the turnover band can save real cost
    when the top-k SET is stable across periods (stage 4 assumes a full reset every single
    period, band-free), so a signal whose top-k membership churns rarely can show stage 5 >
    stage 4 -- that is the band doing its job, not a bug in the waterfall.
    """
    D, N = scores.shape
    n_steps = D - 1
    rebal_idx = list(range(0, n_steps, rebalance_interval_days))

    long_short, long_only, capped, plus_costs = [], [], [], []
    prev_capped_w = None
    for t in rebal_idx:
        n_days = min(rebalance_interval_days, n_steps - t)
        active_idx = np.where(active[t])[0]
        if len(active_idx) < 2 * k:
            continue  # not enough names for a clean long/short split this period

        fwd_cum = np.prod(1.0 + simple_returns[t + 1: t + n_days + 1], axis=0) - 1.0  # [N]
        ew_ret = float(fwd_cum[active_idx].mean())

        order = active_idx[np.argsort(scores[t, active_idx])]
        bottom_k, top_k = order[:k], order[-k:]
        long_leg, short_leg = float(fwd_cum[top_k].mean()), float(fwd_cum[bottom_k].mean())

        long_short.append(long_leg - short_leg)
        long_only.append(long_leg - ew_ret)

        w_capped = select_topk_weights(scores[t], active[t], k, cap=cap)
        capped_ret = float(np.dot(w_capped, fwd_cum))
        capped.append(capped_ret - ew_ret)

        cost = float(np.abs(w_capped - prev_capped_w).sum()) * (cost_bps / 10_000) if prev_capped_w is not None \
            else float(w_capped.sum()) * (cost_bps / 10_000)  # first period: real deploy-from-cash cost too
        plus_costs.append(capped_ret - cost - ew_ret)
        prev_capped_w = w_capped

    res = backtest_topk_strategy(scores, active, simple_returns, k=k, cap=cap,
                                  cost_bps=cost_bps, rebalance_interval_days=rebalance_interval_days,
                                  turnover_band=True)
    plus_holding = []
    for i, t in enumerate(res["rebal_t"]):
        active_idx = np.where(active[t])[0]
        if len(active_idx) == 0:
            continue
        ew_ret = float(np.mean(res["rebal_fwd_rets"][i][active_idx]))
        held_ret = float(np.dot(res["rebal_weights"][i], res["rebal_fwd_rets"][i])) - res["rebal_costs"][i]
        plus_holding.append(held_ret - ew_ret)

    return {
        "long_short": float(np.mean(long_short)) if long_short else 0.0,
        "long_only": float(np.mean(long_only)) if long_only else 0.0,
        "long_only_capped": float(np.mean(capped)) if capped else 0.0,
        "plus_costs": float(np.mean(plus_costs)) if plus_costs else 0.0,
        "plus_holding": float(np.mean(plus_holding)) if plus_holding else 0.0,
        "n_periods": len(long_short),
    }
