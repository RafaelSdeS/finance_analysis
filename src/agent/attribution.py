"""
Return attribution (roadmap M5.5): split the agent's daily return into SELIC
carry, market exposure, and a stock-selection residual, so a single cumulative
return number can't hide "how much of any edge comes from real cross-sectional
judgment" vs. just holding cash or riding the market.

Decomposition is exact per day on SIMPLE returns (log returns aren't additive
across weighted components):
    agent_return_t = w_cash_t * selic_t + (1 - w_cash_t) * ew_stocks_t
                                          + (1 - w_cash_t) * (sleeve_t - ew_stocks_t)
                    = carry_t            + market_exposure_t
                                          + selection_residual_t
where sleeve_t is the agent's actual (cash-excluded) stock-sleeve return, so
selection_residual_t = agent_return_t - carry_t - market_exposure_t exactly.

Cumulative figures compound each stream separately (standard Brinson-style
attribution) -- they're parallel "what if you'd only had this stream"
illustrations, not a partition of the total cumulative return (the per-day
arithmetic decomposition is what's exact, not the compounded totals).
"""

import numpy as np


def decompose_returns(
    agent_log_returns: np.ndarray,
    cash_weights: np.ndarray,
    selic_log_returns: np.ndarray,
    ew_stocks_log_returns: np.ndarray,
) -> dict:
    """
    Args:
        agent_log_returns: [T] agent's total daily log return
        cash_weights: [T] agent's daily CASH weight (0-1)
        selic_log_returns: [T] daily SELIC (risk-free) log return
        ew_stocks_log_returns: [T] equal-weight-of-active-stocks log return (cash excluded)

    Returns dict with per-stream cumulative returns, mean cash weight, and the
    daily arrays (for plotting / further analysis).
    """
    agent_r = np.expm1(np.asarray(agent_log_returns, dtype=np.float64))
    selic_r = np.expm1(np.asarray(selic_log_returns, dtype=np.float64))
    ew_r = np.expm1(np.asarray(ew_stocks_log_returns, dtype=np.float64))
    w_cash = np.clip(np.asarray(cash_weights, dtype=np.float64), 0.0, 1.0)

    carry = w_cash * selic_r
    market_exposure = (1.0 - w_cash) * ew_r
    selection_residual = agent_r - carry - market_exposure

    return {
        "carry_cumulative": float(np.prod(1.0 + carry) - 1.0),
        "market_exposure_cumulative": float(np.prod(1.0 + market_exposure) - 1.0),
        "selection_residual_cumulative": float(np.prod(1.0 + selection_residual) - 1.0),
        "agent_cumulative": float(np.prod(1.0 + agent_r) - 1.0),
        "mean_cash_weight": float(w_cash.mean()),
        "carry_daily": carry,
        "market_exposure_daily": market_exposure,
        "selection_residual_daily": selection_residual,
    }
