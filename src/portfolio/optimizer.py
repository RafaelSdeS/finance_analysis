"""
optimizer.py -- the cost-aware convex program (proposal §5, Stage B):

    maximize_w  a^T w - (lam/2) w^T Sigma w - c1^T |w - w_prev| - (c2/2) ||w - w_prev||^2
    s.t.        sum(w) == 1, w >= 0, w_i <= w_max (except cash), w_cash >= 0

c1 is a ONE-WAY per-asset cost (proposal §5 fix: NOT the round-trip fee --
`|w - w_prev|` already charges once per leg, so a full entry+exit position
naturally costs ~2*c1 over its lifecycle without pre-doubling c1 itself).
"""

import cvxpy as cp
import numpy as np
import pandas as pd


def solve(alpha: pd.Series, sigma: pd.DataFrame, w_prev: pd.Series, c1,
          c2: float = 0.0, lam: float = 1.0, w_max: float = 1.0,
          cash_key: str = "cash") -> pd.Series:
    """
    alpha: expected return for the CURRENT investable universe, incl. `cash_key`
        (the CDI carry) -- exactly what an alpha model + cash carry produce
        each rebalance.
    sigma: covariance for that same current universe, incl. a zero cash
        row/col (risk.add_cash_row_col) -- must be PSD (risk.is_psd).
    w_prev: previous weights, ANY index -- may include names no longer in
        `alpha` (a ticker that fell out of the liquid universe since the
        last rebalance).
    c1: one-way per-asset transaction cost -- a scalar, or a Series aligned
        to `alpha`'s index for per-asset, liquidity-scaled costs (proposal
        §6). Cash's own c1 is forced to 0 internally regardless of what's
        passed (moving into/out of cash isn't an equity trade).

    Names in `w_prev` but absent from `alpha` (proposal §5 pothole P6 -- a
    universe exit) are hard-constrained to w=0 rather than left to the
    objective: there's no current alpha estimate to justify holding them at
    any weight, so this is a forced liquidation, not an optimizer choice --
    but it's still counted in the realized turnover, same as any other trade.

    Returns target weights over union(alpha.index, w_prev.index), long-only,
    summing to 1 (incl. cash), w_max applied to every asset except cash.
    """
    exited = [tkr for tkr in w_prev.index if tkr not in alpha.index and tkr != cash_key]
    assets = list(alpha.index) + exited
    n = len(assets)

    alpha_vec = np.concatenate([alpha.to_numpy(dtype=float), np.zeros(len(exited))])
    sigma_full = sigma.reindex(index=assets, columns=assets).fillna(0.0).to_numpy(dtype=float)
    w_prev_vec = w_prev.reindex(assets).fillna(0.0).to_numpy(dtype=float)

    if np.isscalar(c1):
        c1_vec = np.full(n, float(c1))
    else:
        c1_vec = c1.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    cash_pos = assets.index(cash_key)
    c1_vec[cash_pos] = 0.0

    w = cp.Variable(n)
    delta = w - w_prev_vec

    objective = cp.Maximize(
        alpha_vec @ w
        - (lam / 2) * cp.quad_form(w, cp.psd_wrap(sigma_full))
        - c1_vec @ cp.abs(delta)
        - (c2 / 2) * cp.sum_squares(delta)
    )
    constraints = [cp.sum(w) == 1, w >= 0]
    for i, tkr in enumerate(assets):
        if tkr == cash_key:
            continue
        constraints.append(w[i] == 0 if tkr in exited else w[i] <= w_max)

    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"optimizer did not converge: status={problem.status}")

    return pd.Series(w.value, index=assets)
