"""
environment.py — market mechanics shared by the trained agent AND every
baseline (docs/EIIE_AGENT_PLAN.md "Transaction cost model" and "Global
asset indexing" sections). All math operates in GLOBAL space (cash + 171
union tickers) -- this is what lets a departing ticker's forced sale price
correctly even on its last day, without any special-casing: environment.py
never has to know which columns are "currently in the top-50", it just
solves the paper's equations on whatever w_prev/w_target vectors it's given.

Reference (Jiang, Xu & Liang 2017):
  eq. 7  w'_t = (y_t ⊙ w_{t-1}) / (y_t . w_{t-1})          -- price-drifted weights
  eq. 10 r_t  = ln(mu_t * y_t . w_{t-1})                    -- log return
  eq. 14 mu_t = 1/(1-c_p*w_{t,0}) * [1 - c_p*w'_{t,0}
                 - (c_s+c_p-c_s*c_p) * sum_i (w'_{t,i} - mu_t*w_{t,i})^+]
  eq. 16 mu_0 = c * sum_i |w'_{t,i} - w_{t,i}|               -- initial guess (c_s=c_p=c case)
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

from .data import CASH_GIDX, PricePanel


def drift_weights(y_t: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
    """w'_t (eq. 7): weights after one period's price movement, before any
    rebalancing trade. y_t can carry NaN in a global-space column with no
    price data at all yet (e.g. a ticker whose IPO postdates window_end) --
    zeroed before the multiply so that one absent, always-zero-weight column
    can't poison the whole normalization (0*NaN=NaN otherwise); a no-op
    wherever the column actually has data, which is every column in any
    normal (full-history) experiment window."""
    unnorm = np.nan_to_num(y_t, nan=0.0) * w_prev
    return unnorm / unnorm.sum()


def drift_weights_torch(y_t: torch.Tensor, w_prev: torch.Tensor) -> torch.Tensor:
    """Batched, differentiable version of drift_weights (eq. 7), for
    train.py's loss (w_prev comes from a detached PVM read, but the
    normalization stays a plain differentiable op for consistency/testing).
    Same NaN guard as drift_weights above -- y_t is market data, never
    gradient-tracked, so nan_to_num here has no autograd implication."""
    unnorm = torch.nan_to_num(y_t, nan=0.0) * w_prev
    return unnorm / unnorm.sum(dim=1, keepdim=True)


def solve_mu(w_prime: np.ndarray, w_target: np.ndarray, c_sell: float, c_buy: float,
             tol: float = 1e-10, max_iter: int = 100) -> float:
    """Transaction remainder factor mu_t (eq. 14), converged fixed-point
    iteration (Theorem 1) -- used at backtest/eval time. w_prime/w_target
    are global-space vectors (cash at CASH_GIDX); guaranteed to converge
    for any initial guess in [0, 1] (Theorem 1), so tol/max_iter only
    control precision, not correctness.
    """
    w0_prime = w_prime[CASH_GIDX]
    w0_target = w_target[CASH_GIDX]
    non_cash_prime = np.delete(w_prime, CASH_GIDX)
    non_cash_target = np.delete(w_target, CASH_GIDX)

    denom = 1.0 - c_buy * w0_target
    const = 1.0 - c_buy * w0_prime
    csum_coef = c_sell + c_buy - c_sell * c_buy

    # eq. 16's initial guess assumes c_s == c_p == c; c_sell is used as that
    # shared rate (this repo's config always sets c_sell == c_buy for B3's
    # flat 0.03% -- see CostConfig). Any other starting guess in [0, 1] would
    # still converge per Theorem 1, just possibly slower.
    mu = np.clip(c_sell * np.sum(np.abs(non_cash_prime - non_cash_target)), 0.0, 1.0)

    for _ in range(max_iter):
        pos = np.clip(non_cash_prime - mu * non_cash_target, 0.0, None)
        mu_new = (const - csum_coef * pos.sum()) / denom
        if abs(mu_new - mu) < tol:
            return float(mu_new)
        mu = mu_new
    return float(mu)


def solve_mu_torch(w_prime: torch.Tensor, w_target: torch.Tensor,
                    c_sell: float, c_buy: float, k: int = 1) -> torch.Tensor:
    """Batched, differentiable approximation of solve_mu (eq. 14) for the
    training loss: k fixed Theorem-1 iterations from the eq-16 initial
    guess. The paper leaves k unspecified for training; k=1 is this
    project's documented choice (CostConfig.train_mu_iters).
    w_prime, w_target: [B, n_global], cash at column CASH_GIDX (== 0).
    Returns mu: [B].
    """
    w0_prime = w_prime[:, CASH_GIDX]
    w0_target = w_target[:, CASH_GIDX]
    non_cash_prime = w_prime[:, 1:]
    non_cash_target = w_target[:, 1:]

    denom = 1.0 - c_buy * w0_target
    const = 1.0 - c_buy * w0_prime
    csum_coef = c_sell + c_buy - c_sell * c_buy

    mu = torch.clamp(c_sell * torch.abs(non_cash_prime - non_cash_target).sum(dim=1), 0.0, 1.0)
    for _ in range(k):
        pos = torch.clamp(non_cash_prime - mu.unsqueeze(1) * non_cash_target, min=0.0)
        mu = (const - csum_coef * pos.sum(dim=1)) / denom
    return mu


@dataclass
class BacktestResult:
    dates: pd.DatetimeIndex     # length T, one per simulated trading day
    portfolio_value: np.ndarray  # length T+1: [0] = 1.0 anchor, [i] = value after day i
    log_returns: np.ndarray       # length T, r_t (eq. 10)
    mu: np.ndarray                 # length T
    turnover: np.ndarray            # length T, 0.5 * L1 trade distance (non-cash)
    cost: np.ndarray                 # length T, fractional value lost to costs that day
    weights: np.ndarray                # (T, n_global), w_t actually held each day


WeightFn = Callable[[int, np.ndarray, np.ndarray, PricePanel], np.ndarray]


def run_backtest(panel: PricePanel, weight_fn: WeightFn, c_sell: float, c_buy: float,
                  start_idx: Optional[int] = None, end_idx: Optional[int] = None,
                  mu_tol: float = 1e-10, mu_max_iter: int = 100,
                  on_step: Optional[Callable[[int], None]] = None) -> BacktestResult:
    """Simulate one policy (agent or baseline) over [start_idx, end_idx].

    weight_fn(t, w_prev, w_drift, panel) -> w_target decides the portfolio
    for day t given the previous day's weights and today's price-drifted
    weights (eq. 7); it must return a vector summing to 1. This is the one
    seam between "how a policy decides" (networks.py / baselines.py) and
    "what a decision costs and returns" (this function) -- the agent and
    every baseline share this exact loop, so all get identical treatment.

    on_step(t), if given, runs after period t's bookkeeping is complete --
    train.py's OSBL online backtest hooks in here to run its `rolling_steps`
    gradient updates after each period, without this function needing to
    know anything about training.
    """
    start_idx = panel.start_idx if start_idx is None else start_idx
    end_idx = panel.end_idx if end_idx is None else end_idx
    T = end_idx - start_idx + 1

    n_global = panel.n_global
    w_prev = np.zeros(n_global)
    w_prev[CASH_GIDX] = 1.0  # eq. 5: start all-cash

    portfolio_value = np.empty(T + 1)
    portfolio_value[0] = 1.0
    log_returns = np.empty(T)
    mus = np.empty(T)
    turnovers = np.empty(T)
    costs = np.empty(T)
    weights = np.empty((T, n_global))

    for i, t in enumerate(range(start_idx, end_idx + 1)):
        y_t = panel.price_relative(t)
        w_drift = drift_weights(y_t, w_prev)
        w_target = weight_fn(t, w_prev, w_drift, panel)

        mu = solve_mu(w_drift, w_target, c_sell, c_buy, tol=mu_tol, max_iter=mu_max_iter)
        growth = float(np.dot(y_t, w_prev))

        portfolio_value[i + 1] = portfolio_value[i] * mu * growth
        log_returns[i] = np.log(mu * growth)
        mus[i] = mu
        turnovers[i] = 0.5 * np.sum(np.abs(w_target - w_drift))
        costs[i] = growth * (1.0 - mu)
        weights[i] = w_target

        w_prev = w_target
        if on_step is not None:
            on_step(t)

    return BacktestResult(
        dates=panel.dates[start_idx:end_idx + 1],
        portfolio_value=portfolio_value,
        log_returns=log_returns,
        mu=mus,
        turnover=turnovers,
        cost=costs,
        weights=weights,
    )
