"""
stats.py — generic cross-sectional and time-series statistics shared by the
H-series milestones (spine.py's target construction, features.py's
sector-neutral characteristics, milestone_h0.py's power analysis,
milestone_h1.py's IC screen). Every function takes plain pandas/numpy
inputs, not a dataset-specific object, so each is independently testable.
"""

import numpy as np
import pandas as pd

MIN_GROUP_N = 5  # minimum non-NaN pair count for a per-date Spearman IC to be meaningful


def rank_normalize(values: pd.Series, groups: pd.Series) -> pd.Series:
    """Per-group centered-uniform rank: rank/(N+1) - 0.5, where N is the
    number of non-NaN values IN THAT GROUP (not the group's raw row count).
    Bounded to [-0.5, 0.5] regardless of how N fluctuates across groups
    (e.g. a shrinking eligible universe) -- an ordinal 1..N rank would let
    feature/target variance drift with N across history, distorting a
    regularized linear model's weights."""
    df = pd.DataFrame({"v": values.to_numpy(), "g": groups.to_numpy()})

    def _norm(s: pd.Series) -> pd.Series:
        n = s.notna().sum()
        if n == 0:
            return s * np.nan
        return s.rank(method="average") / (n + 1) - 0.5

    out = df.groupby("g")["v"].transform(_norm)
    out.index = values.index
    return out


def winsorize_cross_sectional(values: pd.Series, groups: pd.Series,
                               lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Per-group winsorization at the [lower, upper] quantiles -- H2's
    outlier-guard ablation for raw-return targets (rank targets don't need
    this; they're bounded by construction)."""
    df = pd.DataFrame({"v": values.to_numpy(), "g": groups.to_numpy()})

    def _clip(s: pd.Series) -> pd.Series:
        lo, hi = s.quantile(lower), s.quantile(upper)
        return s.clip(lo, hi)

    out = df.groupby("g")["v"].transform(_clip)
    out.index = values.index
    return out


def sector_demean(values: pd.Series, dates: pd.Series, sectors: pd.Series) -> pd.Series:
    """Characteristic value minus its (date, sector) mean -- the guard
    against scoring a static sector tilt (e.g. "banks are cheap") as
    stock-picking alpha. Sector-of-one groups (mean collapses to the
    stock's own value) are NaN'd out, matching build_dataset/
    cross_sectional.py's convention for the same guard."""
    df = pd.DataFrame({"v": values.to_numpy(), "d": dates.to_numpy(), "s": sectors.to_numpy()})
    grp = df.groupby(["d", "s"])["v"]
    demeaned = df["v"] - grp.transform("mean")
    size = grp.transform("size")
    out = demeaned.where(size > 1)
    out.index = values.index
    return out


def spearman_ic_by_group(x: pd.Series, y: pd.Series, groups: pd.Series,
                          min_n: int = MIN_GROUP_N) -> pd.Series:
    """Spearman rank correlation between x and y, computed separately per
    group (one IC value per decision date) -- the core primitive of the H1
    screen. A group with fewer than min_n finite (x, y) pairs is NaN, not
    dropped, so the caller can see exactly which dates were unscoreable."""
    df = pd.DataFrame({"x": x.to_numpy(), "y": y.to_numpy(), "g": groups.to_numpy()})
    valid = df.dropna(subset=["x", "y"])

    def _ic(g: pd.DataFrame) -> float:
        if len(g) < min_n:
            return np.nan
        return float(g["x"].corr(g["y"], method="spearman"))

    all_groups = pd.Index(sorted(df["g"].unique()), name="g")
    if len(valid) == 0:
        return pd.Series(np.nan, index=all_groups)
    ic = valid.groupby("g").apply(_ic, include_groups=False)
    return ic.reindex(all_groups).sort_index()


def newey_west_tstat(series: np.ndarray, lag: int) -> tuple:
    """Newey-West (Bartlett kernel) HAC t-statistic for whether a series'
    mean is zero -- the significance test for both an IC time series (H1's
    gate) and an active-return series (H0's power analysis). lag should be
    >= the overlap induced by the target horizon (see spine.hac_lag_for_horizon).
    lag=0 reduces to the population-variance (ddof=0) iid case. Returns
    (mean, se, tstat); (nan, nan, nan) if fewer than 2 finite points."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 2:
        return float("nan"), float("nan"), float("nan")
    mean = float(x.mean())
    resid = x - mean
    gamma0 = float(np.mean(resid * resid))
    lrv = gamma0
    for j in range(1, min(lag, T - 1) + 1):
        w = 1.0 - j / (lag + 1)  # Bartlett kernel weight
        gamma_j = float(np.mean(resid[j:] * resid[:-j]))
        lrv += 2.0 * w * gamma_j
    lrv = max(lrv, 1e-12)  # guard: a pathological negative HAC estimate never yields a bogus finite t-stat
    se = float(np.sqrt(lrv / T))
    tstat = float(mean / se) if se > 0 else float("nan")
    return mean, se, tstat


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg FDR procedure. Returns a boolean reject-null mask
    aligned to pvalues' original order. NaN p-values are never rejected."""
    p = np.asarray(pvalues, dtype=float)
    m = int(np.isfinite(p).sum())
    reject = np.zeros(len(p), dtype=bool)
    if m == 0:
        return reject

    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    sorted_p = p[order]
    thresh = (np.arange(1, m + 1) / m) * alpha
    below = sorted_p[:m] <= thresh
    if not below.any():
        return reject
    k = int(np.max(np.where(below)[0]))  # largest rank i (0-indexed) with p_(i) <= (i/m)*alpha
    reject[order[:k + 1]] = True
    return reject


def min_detectable_ic(n_obs: int, n_assets: int, lag: int, t_threshold: float = 2.0) -> float:
    """Closed-form minimum detectable mean rank-IC at t_threshold, given
    n_obs monthly observations, a cross-section of n_assets, and lag
    periods of NW-induced overlap. sigma_ic_null = 1/sqrt(n_assets-1) is
    the standard error of a Spearman rank correlation between two
    INDEPENDENT rank vectors of length n_assets (the pure-noise floor with
    no signal and no autocorrelation); lag deflates n_obs to its
    overlap-adjusted effective count. Pre-registered by H0 before any real
    characteristic is examined -- not fit to observed data."""
    sigma_ic_null = 1.0 / np.sqrt(max(n_assets - 1, 1))
    n_eff = max(n_obs / (lag + 1), 1.0)
    return float(t_threshold * sigma_ic_null / np.sqrt(n_eff))


def min_detectable_ir(n_obs_monthly: int, periods_per_year: int = 12, t_threshold: float = 2.0) -> float:
    """Closed-form minimum detectable annualized Information Ratio at
    t_threshold, given n_obs_monthly active-return observations:
    t = (IR / sqrt(periods_per_year)) * sqrt(n_obs)  =>  IR = t*sqrt(ppy/n_obs)."""
    return float(t_threshold * np.sqrt(periods_per_year / max(n_obs_monthly, 1)))
