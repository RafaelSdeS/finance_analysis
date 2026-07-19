"""
risk_portfolios.py — structural (no-alpha) portfolio policies for the
risk/diversification mandate (RISK_MANDATE_PLAN.md, RISK_MANDATE_IMPL_PLAN.md
Option A). M1-M3 (M4_DECISION_FINAL.md) found no measurable cross-sectional
mu; with mu treated as homogeneous across assets, Merton's
w* = (1/gamma) Sigma^-1 (mu - r) degenerates to the minimum-variance
direction Sigma^-1 1 -- these policies harvest that (and its risk-parity
sibling) instead of forecasting returns.

Every policy is an ordinary weight_fn(t, w_prev, w_drift, panel) -> w_target,
plugged into environment.run_backtest exactly like every baseline in
baselines.py -- same costs, same drift mechanics, same treatment as the
trained agent.
"""

from typing import Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from .config import RiskConfig
from .data import CASH_GIDX, PricePanel
from .metrics import TRADING_DAYS_PER_YEAR

RISK_POLICY_NAMES = (
    "min_variance", "risk_parity", "min_variance_voltarget", "risk_parity_voltarget",
)


def _active_gidx(t: int, panel: PricePanel) -> np.ndarray:
    mask = panel.valid[t]
    return panel.slot_gidx[t][mask]


def trailing_returns(panel: PricePanel, t: int, lookback: int, gidx: np.ndarray) -> np.ndarray:
    """[lookback, len(gidx)] simple daily returns ending at t (inclusive),
    built from the same day-t-close information set drift_weights uses --
    zero lookahead by construction. Requires t >= lookback."""
    lo = t - lookback + 1
    if lo < 1:
        raise ValueError(f"trailing_returns needs t >= lookback ({lookback}), got t={t}")
    t_idx = np.arange(lo, t + 1)
    y = panel.price_relative_batch(t_idx)
    return y[:, gidx] - 1.0


def eligible_mask(panel: PricePanel, t: int, lookback: int, gidx: np.ndarray,
                   min_history_frac: float) -> np.ndarray:
    """A column counts as seasoned once it has >= min_history_frac * lookback
    days of genuine (non-backfilled) price history within the window.

    data.py flat-backfills a ticker's pre-listing days with its first real
    price (paper Sec. 3.3 zero-decay convention) rather than leaving them
    NaN, so a plain isfinite check on returns can't see a recent IPO/entrant
    -- this instead looks for the leading run of closes identical to the
    window's first value, which is exactly what a backfilled prefix looks
    like (a real top-50-liquid name essentially never holds its close dead
    flat for that many consecutive trading days). Requires t >= lookback."""
    lo = t - lookback + 1
    if lo < 1:
        raise ValueError(f"eligible_mask needs t >= lookback ({lookback}), got t={t}")
    prices = panel.close[lo:t + 1, gidx]  # (lookback, len(gidx))
    same_as_first = prices == prices[0]
    leading_run = np.cumprod(same_as_first, axis=0).sum(axis=0)
    real_frac = 1.0 - leading_run / lookback
    return real_frac >= min_history_frac


def estimate_cov(returns: np.ndarray, cfg: RiskConfig) -> np.ndarray:
    """Annualized covariance from an all-finite [L, N] simple-return slice
    (caller restricts columns via eligible_mask first). Ledoit-Wolf
    shrinkage handles the L~126-250, N<=50 near-singular regime with no
    tuning; a scale-relative jitter is always added on top so the result is
    PD even in pathological (e.g. duplicate-column) inputs."""
    if cfg.cov_estimator == "ewma":
        L = returns.shape[0]
        lam = np.exp(-np.log(2) / cfg.ewma_halflife)
        age = np.arange(L - 1, -1, -1)  # most recent row = age 0
        w = lam ** age
        w /= w.sum()
        mean = (w[:, None] * returns).sum(axis=0)
        # EWMA's effective sample size (~2*halflife/ln2 days) is SMALLER than the
        # window, so it needs MORE shrinkage, not less -- feed the reweighted rows
        # through the same LedoitWolf call as the primary path rather than using
        # the raw weighted product directly, one code path handles both estimators'
        # conditioning. z_i = sqrt(w_i * L) * (x_i - mean) makes mean(z z^T) over L
        # samples equal exactly sum_i w_i (x_i-mean)(x_i-mean)^T.
        centered = (returns - mean) * np.sqrt(w[:, None] * L)
        cov = LedoitWolf(assume_centered=True).fit(centered).covariance_
    else:
        cov = LedoitWolf().fit(returns).covariance_

    cov = cov * TRADING_DAYS_PER_YEAR
    eps = 1e-8 * (np.trace(cov) / cov.shape[0])
    return cov + eps * np.eye(cov.shape[0])


def min_variance_weights(cov: np.ndarray, max_weight: float = 1.0,
                          x0: Optional[np.ndarray] = None, tol: float = 1e-9) -> tuple:
    """Long-only simplex QP: min w'Sigma*w s.t. sum(w)=1, 0<=w<=max_weight.
    Solved on a trace-rescaled Sigma for numerical conditioning (argmin is
    invariant to positive scaling). Returns (w, converged: bool)."""
    n = cov.shape[0]
    if max_weight * n < 1.0 - 1e-9:
        return np.full(n, 1.0 / n), False  # infeasible cap for this many names

    scale = np.trace(cov) / n
    cov_hat = cov / scale

    if x0 is None:
        inv_var = 1.0 / np.diag(cov_hat)
        x0 = inv_var / inv_var.sum()
    else:
        x0 = np.clip(x0, 0.0, max_weight)
        total = x0.sum()
        x0 = x0 / total if total > 0 else np.full(n, 1.0 / n)

    res = minimize(lambda w: w @ cov_hat @ w, x0, jac=lambda w: 2 * cov_hat @ w,
                    method="SLSQP", bounds=[(0.0, max_weight)] * n,
                    constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                    options={"ftol": tol, "maxiter": 500})

    w = np.clip(res.x, 0.0, None)
    total = w.sum()
    if total <= 0 or not np.all(np.isfinite(w)):
        return x0, False
    return w / total, bool(res.success)


def risk_parity_weights(cov: np.ndarray, tol: float = 1e-8, max_iter: int = 500) -> tuple:
    """Equal risk contribution via Spinu's convex reformulation
    (min 1/2 x'Sigma*x - c*sum(log x_i), any c > 0, unique interior
    minimizer), solved by cyclical coordinate descent with the closed-form
    per-coordinate update (Griveau-Billion et al.): with
    b_i = (Sigma*x)_i - Sigma_ii*x_i, the positive root of
    Sigma_ii*x_i^2 + b_i*x_i - c = 0. No bounds needed (interior by
    construction). Returns (w, converged: bool)."""
    n = cov.shape[0]
    diag = np.diag(cov)
    x = 1.0 / np.sqrt(diag)
    x = x / x.sum()
    c = 1.0  # any c > 0 fixes the same minimizer's direction; washes out at normalization

    for _ in range(max_iter):
        Sx = cov @ x
        max_err = 0.0
        for i in range(n):
            b_i = Sx[i] - diag[i] * x[i]
            new_xi = max((-b_i + np.sqrt(b_i ** 2 + 4 * diag[i] * c)) / (2 * diag[i]), 1e-12)
            Sx += cov[:, i] * (new_xi - x[i])  # keep Sx in sync for the rest of this sweep
            max_err = max(max_err, abs(new_xi - x[i]))
            x[i] = new_xi
        if max_err < tol:
            return x / x.sum(), True

    return x / x.sum(), False


def vol_target_overlay(w_eq: np.ndarray, cov: np.ndarray, vol_target_ann: float) -> tuple:
    """Scales an equity-sleeve simplex against its own ex-ante annualized vol
    so the blended equity+cash sleeve holds vol_target_ann; residual to
    cash. Returns (f, sigma_p): f in [0, 1] is the equity-sleeve fraction
    (caller applies f*w_eq into global space, 1-f to CASH_GIDX). f=1 (no
    cash top-up) if the sleeve is already at or below target."""
    sigma_p = float(np.sqrt(max(w_eq @ cov @ w_eq, 0.0)))
    if sigma_p <= 0:
        return 1.0, sigma_p
    return min(vol_target_ann / sigma_p, 1.0), sigma_p


def _solve_with_fallback(base: str, cov: np.ndarray, cfg: RiskConfig,
                          x0: Optional[np.ndarray]) -> tuple:
    """Ordered fallback chain (RISK_MANDATE_IMPL_PLAN.md Sec 3.4): retry once
    with heavier jitter, then an analytic inverse-variance/inverse-vol
    proxy -- a backtest never dies on a solver hiccup. Returns
    (w, used_fallback: bool)."""
    def _solve(c: np.ndarray) -> tuple:
        if base == "min_variance":
            return min_variance_weights(c, max_weight=cfg.max_weight, x0=x0, tol=cfg.solver_tol)
        return risk_parity_weights(c, tol=cfg.solver_tol)

    w, ok = _solve(cov)
    if ok and np.all(np.isfinite(w)):
        return w, False

    n = cov.shape[0]
    jitter = 1e-7 * (np.trace(cov) / n)
    w, ok = _solve(cov + jitter * np.eye(n))
    if ok and np.all(np.isfinite(w)):
        return w, True

    diag = np.diag(cov)
    proxy = 1.0 / diag if base == "min_variance" else 1.0 / np.sqrt(diag)
    return proxy / proxy.sum(), True


def make_risk_weight_fn(policy: str, cfg: RiskConfig, panel: PricePanel, start_idx: int,
                         log_fn: Optional[callable] = None):
    """weight_fn factory for one of RISK_POLICY_NAMES, following the
    make_*_weight_fn closure pattern in baselines.py. start_idx anchors the
    rebalance schedule to the BACKTEST's first day (not panel.start_idx --
    same reasoning as baselines.make_ubah_weight_fn). log_fn(msg), if given,
    is called once per solver fallback."""
    if policy not in RISK_POLICY_NAMES:
        raise ValueError(f"unknown risk policy: {policy!r} (available: {RISK_POLICY_NAMES})")
    voltarget = policy.endswith("_voltarget")
    base = "min_variance" if policy.startswith("min_variance") else "risk_parity"

    # ponytail: state only tracks last-solve weights (for warm start), not Sigma itself --
    # an ex-ante diversification ratio needs Sigma per rebalance; add a `cov_by_t` side-channel
    # here if/when R1 wants that KPI surfaced (RISK_MANDATE_IMPL_PLAN.md R0.6).
    state = {"w_eq_by_gidx": {}}

    def fn(t, w_prev, w_drift, panel):
        if (t - start_idx) % cfg.rebalance_every != 0:
            return w_drift

        active = _active_gidx(t, panel)
        lo = t - cfg.lookback + 1
        if lo < 1:
            return w_drift  # not enough lookback history yet this early in the panel

        elig = eligible_mask(panel, t, cfg.lookback, active, cfg.min_history_frac)
        eligible = active[elig]
        if len(eligible) < 2:
            return w_drift  # can't build a covariance from < 2 names

        returns = np.nan_to_num(trailing_returns(panel, t, cfg.lookback, eligible), nan=0.0)
        cov = estimate_cov(returns, cfg)

        x0 = None
        if base == "min_variance" and cfg.warm_start and state["w_eq_by_gidx"]:
            prev = state["w_eq_by_gidx"]
            x0 = np.array([prev.get(int(g), 0.0) for g in eligible])

        w_eq, used_fallback = _solve_with_fallback(base, cov, cfg, x0)
        if used_fallback and log_fn is not None:
            log_fn(f"t={t} policy={policy}: solver fallback engaged ({len(eligible)} names)")

        state["w_eq_by_gidx"] = {int(g): float(w) for g, w in zip(eligible, w_eq)}

        w_global = np.zeros(panel.n_global)
        if voltarget:
            f, _ = vol_target_overlay(w_eq, cov, cfg.vol_target_ann)
            w_global[eligible] = f * w_eq
            w_global[CASH_GIDX] = 1.0 - f
        else:
            w_global[eligible] = w_eq
        return w_global

    return fn
