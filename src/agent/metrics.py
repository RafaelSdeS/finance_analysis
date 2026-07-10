"""
Portfolio performance metrics.

Shared by the training callback (validation Sharpe for early stopping)
and the evaluation/backtesting module. All functions take a 1-D array of
daily log returns unless stated otherwise.
"""

import numpy as np
from scipy import stats as scipy_stats

TRADING_DAYS = 252
EPS = 1e-12


def sharpe_ratio(log_returns: np.ndarray) -> float:
    """Annualized Sharpe ratio from daily log returns (risk-free rate 0)."""
    r = np.asarray(log_returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    return float(r.mean() / (r.std() + EPS) * np.sqrt(TRADING_DAYS))


def excess_sharpe_ratio(strategy_log_returns: np.ndarray, selic_log_returns: np.ndarray) -> float:
    """Annualized Sharpe ratio of excess returns (strategy − SELIC risk-free rate).

    Excess returns remove the carry from the risk-free rate, leaving only alpha.
    """
    s = np.asarray(strategy_log_returns, dtype=np.float64)
    rf = np.asarray(selic_log_returns, dtype=np.float64)
    if len(s) < 2 or len(rf) != len(s):
        return 0.0
    excess = s - rf
    return float(excess.mean() / (excess.std() + EPS) * np.sqrt(TRADING_DAYS))


def sortino_ratio(log_returns: np.ndarray) -> float:
    """Annualized Sortino ratio: penalizes downside deviation only."""
    r = np.asarray(log_returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    downside_std = downside.std() if len(downside) > 0 else EPS
    return float(r.mean() / (downside_std + EPS) * np.sqrt(TRADING_DAYS))


def excess_sortino_ratio(strategy_log_returns: np.ndarray, selic_log_returns: np.ndarray) -> float:
    """Annualized Sortino ratio of excess returns (strategy − SELIC).

    Penalizes downside deviation only, on the excess (alpha) stream.
    """
    s = np.asarray(strategy_log_returns, dtype=np.float64)
    rf = np.asarray(selic_log_returns, dtype=np.float64)
    if len(s) < 2 or len(rf) != len(s):
        return 0.0
    excess = s - rf
    downside = excess[excess < 0]
    downside_std = downside.std() if len(downside) > 0 else EPS
    return float(excess.mean() / (downside_std + EPS) * np.sqrt(TRADING_DAYS))


def max_drawdown(portfolio_values: np.ndarray) -> float:
    """Maximum peak-to-trough decline, as a positive fraction (0.25 = -25%)."""
    v = np.asarray(portfolio_values, dtype=np.float64)
    if len(v) < 2:
        return 0.0
    peaks = np.maximum.accumulate(v)
    drawdowns = (peaks - v) / peaks
    return float(drawdowns.max())


def cumulative_return(portfolio_values: np.ndarray) -> float:
    """Total return over the period: (V_final - V_0) / V_0."""
    v = np.asarray(portfolio_values, dtype=np.float64)
    if len(v) < 2:
        return 0.0
    return float(v[-1] / v[0] - 1.0)


def annualized_return(portfolio_values: np.ndarray) -> float:
    """Geometric annualized return from a daily value series."""
    v = np.asarray(portfolio_values, dtype=np.float64)
    if len(v) < 2 or v[0] <= 0:
        return 0.0
    years = (len(v) - 1) / TRADING_DAYS
    return float((v[-1] / v[0]) ** (1.0 / max(years, EPS)) - 1.0)


def win_rate(log_returns: np.ndarray) -> float:
    """Fraction of days with positive return."""
    r = np.asarray(log_returns, dtype=np.float64)
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def probabilistic_sharpe_ratio(log_returns: np.ndarray, benchmark_sr: float = 0.0) -> float:
    """P(true per-period Sharpe > benchmark_sr), correcting for skew/kurtosis and
    finite sample size (Bailey & Lopez de Prado, 2012). Operates on per-period
    (non-annualized) Sharpe internally — annualization would need a rescaled SE.
    """
    r = np.asarray(log_returns, dtype=np.float64)
    n = len(r)
    if n < 3:
        return 0.5
    sr = r.mean() / (r.std() + EPS)
    skew = scipy_stats.skew(r)
    kurt = scipy_stats.kurtosis(r, fisher=False)  # normal distribution = 3
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, EPS))
    z = (sr - benchmark_sr) * np.sqrt(n - 1) / denom
    return float(scipy_stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """Expected maximum per-period Sharpe across n_trials independent trials whose
    Sharpe estimates are ~ N(0, sharpe_std^2) (Bailey & Lopez de Prado, 2014)."""
    if n_trials < 2 or sharpe_std <= 0:
        return 0.0
    euler_mascheroni = 0.5772156649
    z1 = scipy_stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = scipy_stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(sharpe_std * ((1 - euler_mascheroni) * z1 + euler_mascheroni * z2))


def deflated_sharpe_ratio(log_returns: np.ndarray, trial_sharpes: np.ndarray) -> float:
    """Probabilistic Sharpe ratio vs. the expected-max-Sharpe benchmark implied by
    trial_sharpes (per-period Sharpes of all N candidate trials, e.g. rolling
    windows) — answers whether the deployed model still looks good after
    correcting for having picked the best of N.
    """
    trial_sharpes = np.asarray(trial_sharpes, dtype=np.float64)
    if len(trial_sharpes) < 2:
        return probabilistic_sharpe_ratio(log_returns, benchmark_sr=0.0)
    benchmark = expected_max_sharpe(len(trial_sharpes), float(trial_sharpes.std()))
    return probabilistic_sharpe_ratio(log_returns, benchmark_sr=benchmark)


def hac_mean_test(x: np.ndarray, lag: int) -> dict:
    """Newey-West HAC t-test for H0: mean(x) == 0, robust to autocorrelation
    up to `lag` periods (Bartlett kernel). Use this instead of a naive
    scipy.stats.ttest_1samp whenever `x` has serial correlation — e.g. daily
    excess returns under N-day rebalancing, which correlate within each
    N-day block. The naive test treats every day as an independent draw and
    understates the true standard error, over-rejecting H0.

    Returns dict with mean, se (HAC standard error of the mean), t_stat,
    p_value (two-sided, normal approximation — standard for HAC/asymptotic
    tests), and n.
    """
    r = np.asarray(x, dtype=np.float64)
    n = len(r)
    if n < 3:
        return {"mean": 0.0, "se": float("nan"), "t_stat": 0.0, "p_value": 1.0, "n": n}

    xbar = r.mean()
    demeaned = r - xbar
    gamma0 = float(np.dot(demeaned, demeaned) / n)

    lag = max(0, min(lag, n - 1))
    var = gamma0
    for k in range(1, lag + 1):
        gamma_k = float(np.dot(demeaned[k:], demeaned[:-k]) / n)
        weight = 1.0 - k / (lag + 1)  # Bartlett kernel
        var += 2 * weight * gamma_k

    se_mean = np.sqrt(max(var, 0.0) / n)
    if se_mean < EPS:
        return {"mean": float(xbar), "se": 0.0, "t_stat": 0.0, "p_value": 1.0, "n": n}

    t_stat = xbar / se_mean
    p_value = float(2 * (1 - scipy_stats.norm.cdf(abs(t_stat))))
    return {"mean": float(xbar), "se": float(se_mean), "t_stat": float(t_stat), "p_value": p_value, "n": n}


def block_bootstrap_mean_ci(
    x: np.ndarray, block_size: int, n_resamples: int = 2000, ci: float = 0.95, seed: int = 0,
) -> dict:
    """Moving-block bootstrap confidence interval on mean(x).

    Resamples overlapping blocks of length `block_size` (with replacement)
    to reconstruct series of the original length, preserving within-block
    autocorrelation structure that an i.i.d. bootstrap would destroy. Use
    alongside hac_mean_test for a second, non-parametric read on significance.
    """
    r = np.asarray(x, dtype=np.float64)
    n = len(r)
    if n < 3:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_resamples": 0}

    block_size = max(1, min(block_size, n))
    n_blocks_needed = int(np.ceil(n / block_size))
    starts = np.arange(n - block_size + 1)  # valid block start positions
    rng = np.random.default_rng(seed)

    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        chosen = rng.choice(starts, size=n_blocks_needed, replace=True)
        sample = np.concatenate([r[s:s + block_size] for s in chosen])[:n]
        boot_means[i] = sample.mean()

    alpha = (1 - ci) / 2
    ci_low, ci_high = np.quantile(boot_means, [alpha, 1 - alpha])
    return {
        "mean": float(r.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_resamples": n_resamples,
    }


def daily_tilt_ic(weights: np.ndarray, ew_weights: np.ndarray, fwd_returns: np.ndarray,
                   active: np.ndarray | None = None, min_names: int = 10) -> np.ndarray:
    """Daily cross-sectional rank-IC of ACTIVE TILTS (weights - equal-weight) vs forward
    returns: does the strategy systematically overweight names that go on to outperform?

    DIAGNOSIS_PLAN.md F3: a time-series Sharpe needs an edge to show up averaged over
    T_years of daily observations (SE ~ 1/sqrt(T_years), so a realistic ~0.2-0.4 Sharpe
    effect is barely resolvable even pooled over 16 years). This test instead has one
    cross-section PER REBALANCE DATE with N names each — for a ~50-name universe rebalanced
    monthly over 16 years, that's ~190 independent-ish cross-sections x 50 names, sized to
    detect an IC on the order of what Phase 0 actually measured (0.03-0.10), not just a
    diluted-by-time-series-noise annualized alpha number.

    Args:
        weights:      [T, N] strategy weights per rebalance date
        ew_weights:   [T, N] equal-weight (benchmark) weights, same shape/dates
        fwd_returns:  [T, N] forward returns aligned to each row's decision date (caller's
                      responsibility — e.g. the return realized over the NEXT rebalance
                      window, never the return already realized when the weights were set)
        active:       [T, N] boolean mask, True where a name is tradable that day
                      (default: every name active)
        min_names:    minimum active names on a date to include that day (else NaN)

    Returns:
        [T] daily rank-IC of (weights - ew_weights) vs fwd_returns; NaN where <min_names
        active names or degenerate (zero-variance) cross-sections.
    """
    w = np.asarray(weights, dtype=np.float64)
    ew = np.asarray(ew_weights, dtype=np.float64)
    fwd = np.asarray(fwd_returns, dtype=np.float64)
    T, N = w.shape
    mask = np.ones((T, N), dtype=bool) if active is None else np.asarray(active, dtype=bool)

    tilt = w - ew
    ic = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        m = mask[t] & ~np.isnan(tilt[t]) & ~np.isnan(fwd[t])
        if m.sum() < min_names:
            continue
        tilt_rank = scipy_stats.rankdata(tilt[t, m])
        fwd_rank = scipy_stats.rankdata(fwd[t, m])
        if tilt_rank.std() < EPS or fwd_rank.std() < EPS:
            continue
        ic[t] = float(np.corrcoef(tilt_rank, fwd_rank)[0, 1])
    return ic


def tilt_ic_significance(weights: np.ndarray, ew_weights: np.ndarray, fwd_returns: np.ndarray,
                          active: np.ndarray | None = None, min_names: int = 10, lag: int = 1) -> dict:
    """HAC-corrected significance test on the daily tilt-IC series (see daily_tilt_ic).

    `lag` should match the rebalance interval in cross-section STEPS (not trading days —
    daily_tilt_ic already returns one value per REBALANCE date, so consecutive entries are
    already `rebalance_interval_days` apart; lag=1 is usually enough unless tilts
    themselves are held for multiple rebalances without change).
    """
    ic = daily_tilt_ic(weights, ew_weights, fwd_returns, active, min_names)
    valid = ic[~np.isnan(ic)]
    result = hac_mean_test(valid, lag=lag)
    result["n_total_days"] = int(len(ic))
    result["n_valid_days"] = int(len(valid))
    return result


def effective_n_positions(weights: np.ndarray) -> float:
    """Effective number of positions (1/HHI) averaged over the period."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2 or len(w) == 0:
        return 0.0
    hhi = (w ** 2).sum(axis=1)
    return float(np.mean(1.0 / np.maximum(hhi, EPS)))


def compute_all(log_returns: np.ndarray, portfolio_values: np.ndarray, weights: np.ndarray | None = None,
                daily_costs: np.ndarray | None = None, cost_bps: float | None = None,
                selic_log_returns: np.ndarray | None = None) -> dict:
    """All metrics in one dict (for metrics.json / comparison tables).

    Args:
        log_returns: daily log returns
        portfolio_values: daily portfolio values (length = len(log_returns) + 1, including initial capital)
        weights: optional [days, assets] weight matrix
        daily_costs: optional [days] transaction cost per day (cost on rebalance day, 0 elsewhere)
        cost_bps: optional cost basis points; if given with weights, override turnover calculation
        selic_log_returns: optional daily SELIC returns for excess-of-SELIC metrics
    """
    metrics = {
        "cumulative_return": cumulative_return(portfolio_values),
        "annualized_return": annualized_return(portfolio_values),
        "sharpe": sharpe_ratio(log_returns),
        "sortino": sortino_ratio(log_returns),
        "max_drawdown": max_drawdown(portfolio_values),
        "win_rate": win_rate(log_returns),
        "n_days": int(len(log_returns)),
    }

    # Excess-of-SELIC metrics (if SELIC returns provided)
    if selic_log_returns is not None:
        metrics["excess_sharpe"] = excess_sharpe_ratio(log_returns, selic_log_returns)
        metrics["excess_sortino"] = excess_sortino_ratio(log_returns, selic_log_returns)

    # Gross Sharpe and cost drag (if daily_costs provided)
    if daily_costs is not None:
        daily_costs = np.asarray(daily_costs, dtype=np.float64)
        # Additive approximation: gross log return ≈ net log return + cost
        gross_log_rets = log_returns + daily_costs
        metrics["gross_sharpe"] = sharpe_ratio(gross_log_rets)
        metrics["annualized_cost_drag"] = float(daily_costs.mean() * TRADING_DAYS)

    if weights is not None:
        metrics["effective_n"] = effective_n_positions(weights)
        # Concentration: mean of daily max weight (0.004 for equal weight, >0.01 for conviction)
        metrics["max_weight"] = float(weights.max(axis=1).mean())

        # Turnover: override if cost_bps given (more accurate than diff-based)
        if cost_bps is not None and daily_costs is not None:
            # Recover one-way turnover from cost: cost = turnover * (bps / 1e4)
            daily_costs_clean = np.asarray(daily_costs, dtype=np.float64)
            # Cost on rebalance days; derive turnover
            rebalance_days = np.where(daily_costs_clean > 0)[0]
            if len(rebalance_days) > 0:
                # Average turnover per rebalance day
                rebalance_turnover = daily_costs_clean[rebalance_days] / (cost_bps / 1e4)
                # Daily average (turnover concentrated on rebalance days)
                metrics["avg_daily_turnover"] = float(rebalance_turnover.mean() / 1)  # per-day on rebalance
            else:
                metrics["avg_daily_turnover"] = 0.0
        else:
            # Fallback: turnover from weight diffs (less accurate on drifted weights, but works)
            w = np.asarray(weights, dtype=np.float64)
            daily_turnover = np.abs(np.diff(w, axis=0)).sum(axis=1)
            metrics["avg_daily_turnover"] = float(daily_turnover.mean())

    return metrics
