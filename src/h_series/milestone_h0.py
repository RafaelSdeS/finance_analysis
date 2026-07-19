"""
milestone_h0.py — H-series Milestone H0: walk-forward baselines + the
block-bootstrap / Newey-West power analysis that freezes H1's kill-gate
thresholds BEFORE any characteristic is examined
(MEDIUM_HORIZON_RESEARCH_PLAN.md sec 3, H0).

Run:
    python -m src.h_series.milestone_h0

Writes H0_FINDINGS.md (human-readable) and H0_FINDINGS.json (the frozen
gate numbers milestone_h1.py reads).

Design note on "expanding windows, step annually, stitch OOS": every H0
baseline (UCRP, BOVA11, min-variance, classical MV) has NO fitted global
parameter -- min-variance and classical MV re-estimate Sigma (and, for
classical MV, mu) from a trailing `LOOKBACK`-day window AT EVERY rebalance,
which is already causal by construction (risk_portfolios.trailing_returns
never reads past t). Re-running these under an annual "refit" wrapper
would produce byte-identical output to one continuous backtest -- there is
no parameter an annual boundary could change. spine.iter_expanding_folds
is used only to derive the stitched-OOS START boundary (fold[0].train_end,
matching "initial train 2011-2018"); H1/H2's ridge regression is where
per-fold refitting first has real content, since ridge coefficients ARE
global fitted parameters.
"""

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..rl_agent.baselines import run_baseline
from ..rl_agent.config import CostConfig, DataConfig, RiskConfig
from ..rl_agent.data import PricePanel, load_price_panel
from ..rl_agent.environment import run_backtest
from ..rl_agent.metrics import TRADING_DAYS_PER_YEAR, block_bootstrap_ci
from ..rl_agent.risk_portfolios import eligible_mask, estimate_cov, trailing_returns
from .paths import H0_FINDINGS_JSON, H0_FINDINGS_MD
from .spine import hac_lag_for_horizon, iter_expanding_folds
from .stats import min_detectable_ic, min_detectable_ir, newey_west_tstat

INITIAL_TRAIN_END = "2018-12-31"
STEP_MONTHS = 12
LOOKBACK = 126                    # trading days; matches R-series RiskConfig default
REBALANCE_EVERY = 21              # ~monthly, trading days
N_ASSETS = 50                     # top-50 universe, by design (CLAUDE.md)
MV_RISK_AVERSION = 3.0            # fixed convention for the classical straw man, never tuned
MV_TURNOVER_PENALTY_MULT = 10.0   # x c_sell; internalizes roughly what run_backtest will charge anyway
BOOTSTRAP_N = 2000
BOOTSTRAP_BLOCK_MONTHS = 4        # preserves autocorrelation from k=63's overlapping target
T_THRESHOLD = 2.0
K_HORIZONS = (21, 63)


def _active_gidx(t: int, panel: PricePanel) -> np.ndarray:
    mask = panel.valid[t]
    return panel.slot_gidx[t][mask]


def make_classical_mv_weight_fn(cfg: RiskConfig, start_idx: int, risk_aversion: float = MV_RISK_AVERSION,
                                 turnover_penalty: float = 0.003):
    """Long-only mean-variance with a historical-mean mu estimate and an L1
    turnover penalty vs the previous weights -- the ONE baseline in this
    module that uses a return estimate, deliberately: it exists as the
    "beat this" straw man H2 must clear (MEDIUM_HORIZON_RESEARCH_PLAN.md
    sec 3, H0), not as a proposed strategy. risk_portfolios.py's own
    policies stay mu-free per the risk mandate's "no return view" contract
    -- this function does NOT belong there, and doesn't touch it.

    max_w  mu'w - (risk_aversion/2) w'Sigma w - turnover_penalty*||w-w_prev||_1
    s.t. sum(w)=1, 0<=w<=1

    No analytic gradient supplied (SLSQP finite-differences it) -- the L1
    term's kink makes a clean analytic jac not worth deriving for a
    ~monthly-rebalance, n<=50-dim solve.

    Mirrors risk_portfolios.make_risk_weight_fn's rebalance-gate and
    early-lookback guard exactly -- without them this would resolve a
    fresh SLSQP problem every trading day instead of every
    cfg.rebalance_every days, making it a daily-rebalanced strategy
    compared against monthly-rebalanced baselines, an apples-to-oranges
    H0 comparison."""

    def fn(t, w_prev, w_drift, panel):
        if (t - start_idx) % cfg.rebalance_every != 0:
            return w_drift
        if t - cfg.lookback + 1 < 1:
            return w_drift  # not enough lookback history yet this early in the panel

        active = _active_gidx(t, panel)
        elig = eligible_mask(panel, t, cfg.lookback, active, cfg.min_history_frac)
        eligible = active[elig]
        if len(eligible) < 2:
            return w_drift

        returns = np.nan_to_num(trailing_returns(panel, t, cfg.lookback, eligible), nan=0.0)
        cov = estimate_cov(returns, cfg)
        mu = returns.mean(axis=0) * TRADING_DAYS_PER_YEAR

        w_prev_eq = w_prev[eligible]
        n = len(eligible)

        def objective(w):
            return -(mu @ w) + 0.5 * risk_aversion * (w @ cov @ w) + turnover_penalty * np.sum(np.abs(w - w_prev_eq))

        x0 = w_prev_eq if w_prev_eq.sum() > 0 else np.full(n, 1.0 / n)
        res = minimize(objective, x0, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                        options={"ftol": cfg.solver_tol, "maxiter": 500})
        w_eq = np.clip(res.x, 0.0, None)
        total = w_eq.sum()
        w_eq = w_eq / total if total > 0 else np.full(n, 1.0 / n)

        w_global = np.zeros(panel.n_global)
        w_global[eligible] = w_eq
        return w_global

    return fn


def _monthly_log_returns(dates: pd.DatetimeIndex, log_returns: np.ndarray) -> pd.Series:
    """Aggregate a daily log-return series to monthly (sum of daily log
    returns within a calendar month == log of that month's gross return)
    -- H0's stitched-OOS series is ~90 monthly points, not ~1,900 daily
    ones, matching the frequency H1/H2 actually operate at. dates and
    log_returns are the same length T (BacktestResult's convention:
    log_returns[i] is the return realized ON dates[i])."""
    s = pd.Series(np.asarray(log_returns), index=pd.DatetimeIndex(dates))
    return s.groupby(s.index.to_period("M")).sum()


def run_h0_baselines() -> dict:
    """Every baseline/policy over the stitched OOS span (first fold's
    train_end through the dataset's last date), same costs as the
    R-series (3bps/side). Returns {name: monthly_return_series}."""
    data_cfg = DataConfig(features=("close", "high", "low"))
    costs = CostConfig()
    panel = load_price_panel(data_cfg, n_slots=50)

    folds = iter_expanding_folds(data_cfg.window_end, INITIAL_TRAIN_END, STEP_MONTHS)
    oos_start = folds[0].train_end
    start_idx = int(np.searchsorted(panel.dates.values, np.datetime64(oos_start), side="right"))
    end_idx = panel.end_idx

    risk_cfg = RiskConfig(lookback=LOOKBACK, rebalance_every=REBALANCE_EVERY)

    results = {}
    for name in ("ucrp", "bova11"):
        results[name] = run_baseline(name, panel, costs.c_sell, costs.c_buy,
                                      start_idx=start_idx, end_idx=end_idx)
    results["min_variance"] = run_baseline("min_variance", panel, costs.c_sell, costs.c_buy,
                                            start_idx=start_idx, end_idx=end_idx, risk_cfg=risk_cfg)

    mv_fn = make_classical_mv_weight_fn(risk_cfg, start_idx, turnover_penalty=MV_TURNOVER_PENALTY_MULT * costs.c_sell)
    results["classical_mv"] = run_backtest(panel, mv_fn, costs.c_sell, costs.c_buy, start_idx, end_idx)

    dates = panel.dates[start_idx:end_idx + 1]
    cdi_daily = panel.cdi_factor[start_idx:end_idx + 1] - 1.0
    rf_monthly = _monthly_log_returns(dates, np.log(1.0 + cdi_daily))

    monthly = {}
    active_vs_bova = {}
    r_bova_monthly = None
    for name, result in results.items():
        r_monthly = _monthly_log_returns(dates, result.log_returns)
        simple_monthly = np.exp(r_monthly) - 1.0
        monthly[name] = simple_monthly
        if name == "bova11":
            r_bova_monthly = simple_monthly
    for name, simple_monthly in monthly.items():
        bench = r_bova_monthly if r_bova_monthly is not None else pd.Series(0.0, index=simple_monthly.index)
        active_vs_bova[name] = (simple_monthly - bench.reindex(simple_monthly.index)).dropna()

    return {"monthly_return": monthly, "active_vs_bova11": active_vs_bova,
            "risk_free_monthly": np.exp(rf_monthly) - 1.0}


def main() -> None:
    data = run_h0_baselines()
    monthly = data["monthly_return"]
    active = data["active_vs_bova11"]
    n_obs = len(next(iter(monthly.values())))

    summary = {}
    for name, series in monthly.items():
        point, lo, hi = block_bootstrap_ci(
            series.to_numpy(), lambda r: float(np.prod(1.0 + r) - 1.0),
            n_bootstrap=BOOTSTRAP_N, block_size=BOOTSTRAP_BLOCK_MONTHS, seed=42,
        )
        mean_active, se_active, t_active = newey_west_tstat(active[name].to_numpy(), lag=0)
        summary[name] = {
            "n_monthly_obs": int(len(series)),
            "total_return": {"point": point, "ci_lo": lo, "ci_hi": hi},
            "monthly_active_return_vs_bova11": {"mean": mean_active, "se": se_active, "tstat": t_active},
        }

    floors = {"n_monthly_obs": n_obs}
    for k in K_HORIZONS:
        lag = hac_lag_for_horizon(k)
        floors[str(k)] = {
            "min_detectable_ic": min_detectable_ic(n_obs, N_ASSETS, lag, T_THRESHOLD),
            "hac_lag_months": lag,
        }
    floors["min_detectable_ir_annual"] = min_detectable_ir(n_obs, 12, T_THRESHOLD)

    H0_FINDINGS_JSON.write_text(json.dumps({"baselines": summary, "power_floors": floors},
                                            indent=2, default=str))
    _write_findings_md(summary, floors)
    print(f"H0 complete: {n_obs} stitched monthly OOS obs. "
          f"IC floor k=21: {floors['21']['min_detectable_ic']:.4f}, "
          f"k=63: {floors['63']['min_detectable_ic']:.4f}. "
          f"IR floor: {floors['min_detectable_ir_annual']:.3f}. "
          f"Findings: {H0_FINDINGS_MD}")


def _write_findings_md(summary: dict, floors: dict) -> None:
    lines = [
        "# H0 Findings — Walk-Forward Spine, Baselines, Power Analysis",
        "",
        f"Stitched OOS: {floors['n_monthly_obs']} monthly observations "
        f"({INITIAL_TRAIN_END} onward through the dataset's last date).",
        "",
        "## Power floors (frozen BEFORE any H1 characteristic is examined)",
        "",
        f"- Min detectable mean IC (t=2, k=21): {floors['21']['min_detectable_ic']:.4f} "
        f"(NW lag {floors['21']['hac_lag_months']} months)",
        f"- Min detectable mean IC (t=2, k=63): {floors['63']['min_detectable_ic']:.4f} "
        f"(NW lag {floors['63']['hac_lag_months']} months)",
        f"- Min detectable annualized IR (t=2): {floors['min_detectable_ir_annual']:.3f}",
        "",
        "## Baseline summary (monthly, net of 3bps/side costs; block-bootstrap CI, block=4 months)",
        "",
    ]
    for name, s in summary.items():
        tr = s["total_return"]
        act = s["monthly_active_return_vs_bova11"]
        lines.append(f"### {name}")
        lines.append(f"- Total return (stitched OOS): {tr['point']:.3f} [{tr['ci_lo']:.3f}, {tr['ci_hi']:.3f}]")
        lines.append(f"- Monthly active return vs BOVA11: mean={act['mean']:.5f}, "
                      f"NW-t={act['tstat']:.2f}")
        lines.append("")

    if "classical_mv" in summary:
        cmv_tr = summary["classical_mv"]["total_return"]
        cmv_act = summary["classical_mv"]["monthly_active_return_vs_bova11"]
        lines += [
            "## Interpretation: classical_mv's point estimate is not a robust bar",
            "",
            f"classical_mv shows the strongest point estimate here (total return "
            f"{cmv_tr['point']:.2f}, active-return NW-t={cmv_act['tstat']:.2f}) but also by far "
            f"the widest bootstrap CI ({cmv_tr['ci_hi']:.1f} / {cmv_tr['ci_lo']:.1f} = "
            f"{cmv_tr['ci_hi'] / max(cmv_tr['ci_lo'], 1e-9):.1f}x spread). This is the textbook "
            "instability of naive sample-mean Markowitz optimization (Michaud 1989's "
            "\"optimization enigma\"): with no return VIEW, only a noisy trailing-mean ESTIMATE, "
            "and no per-name weight cap, the optimizer concentrates hard in whatever name had the "
            "best noisy trailing mu -- a few lucky/unlucky realizations dominate the whole stitched "
            "path. It is exactly why risk_portfolios.py's own policies (min_variance, risk_parity) "
            "deliberately carry no mu estimate at all (RISK_MANDATE_PLAN.md). **H2's bar is NOT "
            "\"beat classical_mv's point estimate\" -- a lucky concentrated bet can do that by "
            "construction. The bar is beating it with a MATERIALLY NARROWER bootstrap CI, i.e. "
            "genuine breadth-of-evidence outperformance, not concentrated luck.**",
            "",
        ]
    H0_FINDINGS_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
