"""
milestone_h3.py — H-series Milestone H3: fitted portfolio construction
(H3_PORTFOLIO_CONSTRUCTION_PLAN.md). Replaces H2's two hand-picked pieces
-- the cap-weight anchor and the ad hoc gamma-tilt -- with fitted
equivalents: a competed model class (composite.select_model), a
Sharpe/IR-selected risk-based anchor (anchor.py), and a kappa-derived blend
(blend.py). Tests whether the H1 survivors carry real incremental value
once H2's diagnosed anchor/blend confound is removed.

Runs on the SAME H1 survivor set H2 used, not an expanded one -- so a
pass/fail here is attributable to the anchor/blend fix alone
(H3_PORTFOLIO_CONSTRUCTION_PLAN.md Rationale).

Run:
    python -m src.h_series.milestone_h3

Reads H1_FINDINGS.json (via composite.load_survivors(), same PASS-or-raise
guard H2 uses). Writes H3_FINDINGS.md + H3_FINDINGS.json.
"""

import json

import numpy as np
import pandas as pd

from ..rl_agent.baselines import run_baseline
from ..rl_agent.config import CostConfig, DataConfig
from ..rl_agent.data import load_price_panel
from .anchor import ANCHOR_TYPES, anchor_weights_by_date
from .blend import blend_weights, kappa_from_ic, view_weights_from_score
from .composite import CONFIRMATION_START, build_feature_matrix, load_survivors, select_model
from .features import _load_daily_prices, build_monthly_panel
from .milestone_h2 import (
    active_monthly_returns,
    construction_a_multiplicative,
    enforce_max_weight,
    ir_bootstrap_ci,
    ir_stats,
    permutation_null,
    quintile_monotonicity,
    run_construction_backtest,
    split_pre_post,
    within_date_permutation_null,
    _decision_date_to_t,
)
from .paths import H3_FINDINGS_JSON, H3_FINDINGS_MD
from .spine import iter_expanding_folds

MAX_WEIGHT = 0.10
K_PRIMARY = 21
K_QUARTERLY = 63
INITIAL_TRAIN_END = "2018-12-31"  # same spine start as H0/H1/H2
STEP_MONTHS = 12
BOOTSTRAP_N = 2000
BOOTSTRAP_BLOCK_MONTHS = 4
P_THRESHOLD = 0.10
PERMUTATION_DRAWS = 200


# ---------------------------------------------------------------------------
# Blend construction, adapted to H2's construction(anchor, score, gamma)
# call signature so H2's build_target_weights_by_t/permutation_null/
# within_date_permutation_null (all written against that signature) can be
# reused verbatim.
# ---------------------------------------------------------------------------

def h3_construction(anchor: np.ndarray, score: np.ndarray, kappa: float) -> np.ndarray:
    """gamma's slot in H2's construction signature carries kappa here --
    already frozen by select_model's pre-2024 OOF IC (Design §c) before
    this is ever called, never swept against a reported metric the way
    H2's gamma was."""
    w_view = view_weights_from_score(score)
    w = blend_weights(anchor, w_view, kappa)
    return enforce_max_weight(w, MAX_WEIGHT)


def build_target_weights_by_t(panel: pd.DataFrame, score_col: str, anchor_col: str,
                               kappa: float, ticker_to_gidx: dict, n_global: int,
                               date_to_t: dict) -> dict:
    """Same contract as milestone_h2.build_target_weights_by_t, specialized
    to h3_construction -- kept local (not imported) because milestone_h2's
    version is already imported here under a different role (the
    anchor-only diagnostic uses construction_a_multiplicative instead)."""
    out = {}
    for date, g in panel.groupby("decision_date"):
        if date not in date_to_t:
            continue
        gidx = g["ticker"].map(ticker_to_gidx).to_numpy()
        anchor = g[anchor_col].fillna(0.0).to_numpy()
        score = g[score_col].fillna(0.0).to_numpy()
        w_local = h3_construction(anchor, score, kappa)
        w_global = np.zeros(n_global)
        w_global[gidx] = w_local
        out[date_to_t[date]] = w_global
    return out


def anchor_only_target_weights_by_t(panel: pd.DataFrame, anchor_col: str, ticker_to_gidx: dict,
                                     n_global: int, date_to_t: dict) -> dict:
    """The 'anchor alone, before any blend' diagnostic H2's own zero_score
    pattern used, reused verbatim: raw = anchor*(1 + 1.0*0) = anchor,
    capped the same way. Needed both for anchor-TYPE selection and for the
    Failure Criteria's "anchor alone underperforms BOVA11/CDI" check."""
    out = {}
    for date, g in panel.groupby("decision_date"):
        if date not in date_to_t:
            continue
        gidx = g["ticker"].map(ticker_to_gidx).to_numpy()
        anchor = g[anchor_col].fillna(0.0).to_numpy()
        zero_score = np.zeros(len(g))
        w_local = construction_a_multiplicative(anchor, zero_score, 1.0, max_weight=MAX_WEIGHT)
        w_global = np.zeros(n_global)
        w_global[gidx] = w_local
        out[date_to_t[date]] = w_global
    return out


# ---------------------------------------------------------------------------
# Anchor-type selection (Design §b): fit, not asserted
# ---------------------------------------------------------------------------

def select_anchor_type(panel: pd.DataFrame, anchor_cols: dict, ticker_to_gidx: dict,
                        rl_panel, costs: CostConfig, start_idx: int, end_idx: int,
                        date_to_t: dict, bench_result, confirmation_start: pd.Timestamp = CONFIRMATION_START) -> tuple:
    """Design §b: anchor TYPE is fit, not asserted. Backtests each
    ANCHOR_TYPES-alone portfolio on pre-2024 data only, picks whichever has
    the higher realized annualized IR vs BOVA11, freezes that choice, and
    reports both splits (post-2024 is reported, never used to pick).
    Returns (winning_type, summary_by_type)."""
    summary = {}
    best_type, best_ir = None, -np.inf
    for atype in ANCHOR_TYPES:
        acol = anchor_cols[atype]
        target_by_t = anchor_only_target_weights_by_t(panel, acol, ticker_to_gidx, rl_panel.n_global, date_to_t)
        result = run_construction_backtest(target_by_t, rl_panel, costs, start_idx, end_idx)
        active = active_monthly_returns(result, bench_result)
        pre, post = split_pre_post(active, confirmation_start)
        pre_ir = ir_stats(pre)["ir_annualized"]
        summary[atype] = {"pre2024": ir_stats(pre), "post2024": ir_stats(post)}
        if np.isfinite(pre_ir) and pre_ir > best_ir:
            best_type, best_ir = atype, pre_ir
    return best_type, summary


# ---------------------------------------------------------------------------
# Bootstrap CI on the IR delta (blended vs anchor-alone)
# ---------------------------------------------------------------------------

def ir_delta_bootstrap_ci(blend_active: pd.Series, anchor_active: pd.Series,
                           n_bootstrap: int = BOOTSTRAP_N, block_size: int = BOOTSTRAP_BLOCK_MONTHS,
                           seed: int = 42) -> tuple:
    """Bootstrap CI on IR(blended) - IR(anchor-alone) -- PAIRED block
    resampling (the same date-blocks drawn for both series each draw,
    preserving their shared market history/correlation) rather than
    bootstrapping each IR independently, which would overstate the delta's
    uncertainty by ignoring that both portfolios share the same market
    history. Returns (point_estimate, lo, hi), same shape as
    rl_agent.metrics.block_bootstrap_ci."""
    idx = blend_active.index.intersection(anchor_active.index)
    b = blend_active.reindex(idx).to_numpy()
    a = anchor_active.reindex(idx).to_numpy()
    T = len(idx)
    if T < 2:
        return float("nan"), float("nan"), float("nan")
    n_blocks = int(np.ceil(T / block_size))
    rng = np.random.default_rng(seed)

    def _ir(r: np.ndarray) -> float:
        s = r.std(ddof=1)
        return float(r.mean() / s * np.sqrt(12)) if s > 0 else 0.0

    deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        starts = rng.integers(0, max(T - block_size, 1) + 1, size=n_blocks)
        b_s = np.concatenate([b[s:s + block_size] for s in starts])[:T]
        a_s = np.concatenate([a[s:s + block_size] for s in starts])[:T]
        deltas[i] = _ir(b_s) - _ir(a_s)

    point = _ir(b) - _ir(a)
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return float(point), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    survivors = load_survivors()
    panel = build_monthly_panel(k_horizons=(K_PRIMARY, K_QUARTERLY))
    folds = iter_expanding_folds(panel["decision_date"].max(), INITIAL_TRAIN_END, STEP_MONTHS)

    # --- (a) Combination layer: competed model class (Design §a) ---
    X, feature_cols = build_feature_matrix(panel, survivors)
    model_result = select_model(panel, X, feature_cols, f"target_rank_k{K_PRIMARY}", folds)
    panel["score_k21"] = model_result["scores"]

    # --- (b) Anchor layer: per-type weights (Design §b) ---
    prices_wide, _bench = _load_daily_prices()
    anchor_cols = {}
    for atype in ANCHOR_TYPES:
        w = anchor_weights_by_date(panel, prices_wide, atype).rename(columns={"weight": f"anchor_{atype}"})
        panel = panel.merge(w, on=["decision_date", "ticker"], how="left")
        anchor_cols[atype] = f"anchor_{atype}"

    data_cfg = DataConfig(features=("close", "high", "low"))
    costs = CostConfig()
    rl_panel = load_price_panel(data_cfg, n_slots=50)
    ticker_to_gidx = rl_panel.asset_index.ticker_to_gidx
    n_global = rl_panel.n_global

    oos_start = folds[0].train_end
    start_idx = int(np.searchsorted(rl_panel.dates.values, np.datetime64(oos_start), side="right"))
    end_idx = rl_panel.end_idx

    oos_panel = panel[panel["decision_date"] > oos_start].copy()
    date_to_t = _decision_date_to_t(rl_panel, oos_panel["decision_date"])

    bench_result = run_baseline("bova11", rl_panel, costs.c_sell, costs.c_buy,
                                 start_idx=start_idx, end_idx=end_idx)
    cdi_result = run_baseline("constant_cash", rl_panel, costs.c_sell, costs.c_buy,
                               start_idx=start_idx, end_idx=end_idx)

    winning_anchor_type, anchor_type_summary = select_anchor_type(
        oos_panel, anchor_cols, ticker_to_gidx, rl_panel, costs, start_idx, end_idx,
        date_to_t, bench_result)
    anchor_col = anchor_cols[winning_anchor_type]

    # Anchor-alone series for the WINNING type -- feeds both the Failure
    # Criteria's "anchor alone underperforms BOVA11/CDI" check and the
    # IR-delta bootstrap below.
    anchor_target_by_t = anchor_only_target_weights_by_t(oos_panel, anchor_col, ticker_to_gidx, n_global, date_to_t)
    anchor_result = run_construction_backtest(anchor_target_by_t, rl_panel, costs, start_idx, end_idx)
    anchor_vs_bova11 = active_monthly_returns(anchor_result, bench_result)
    anchor_vs_bova11_pre, anchor_vs_bova11_post = split_pre_post(anchor_vs_bova11)
    anchor_vs_cdi = active_monthly_returns(anchor_result, cdi_result)
    anchor_vs_cdi_pre, _anchor_vs_cdi_post = split_pre_post(anchor_vs_cdi)

    # --- (c) Blend layer: kappa derived from the winning model's own
    # pre-2024 OOF mean Spearman IC (Design §c) ---
    ic_oof = model_result["diagnostics"]["ic_by_class"][model_result["model_class"]]
    kappa = kappa_from_ic(ic_oof)

    blend_target_by_t = build_target_weights_by_t(oos_panel, "score_k21", anchor_col, kappa,
                                                    ticker_to_gidx, n_global, date_to_t)
    blend_result = run_construction_backtest(blend_target_by_t, rl_panel, costs, start_idx, end_idx)
    blend_active = active_monthly_returns(blend_result, bench_result)
    blend_pre, blend_post = split_pre_post(blend_active)
    blend_summary = {
        "pre2024": ir_stats(blend_pre), "pre2024_ir_ci": ir_bootstrap_ci(blend_pre),
        "post2024": ir_stats(blend_post), "post2024_ir_ci": ir_bootstrap_ci(blend_post),
    }

    ir_delta_pre = ir_delta_bootstrap_ci(blend_pre, anchor_vs_bova11_pre)
    ir_delta_post = ir_delta_bootstrap_ci(blend_post, anchor_vs_bova11_post)

    # --- Validation: two permutation-null tests, both splits ---
    pre_oos_panel = oos_panel[oos_panel["decision_date"] < CONFIRMATION_START]
    post_oos_panel = oos_panel[oos_panel["decision_date"] >= CONFIRMATION_START]

    def _perm_summary(sub_panel: pd.DataFrame, observed_ir: float) -> dict:
        fwd_col = f"fwd_rel_return_k{K_PRIMARY}"
        cross_null = permutation_null(sub_panel, "score_k21", anchor_col, kappa, h3_construction,
                                       fwd_col, n_draws=PERMUTATION_DRAWS)
        within_null = within_date_permutation_null(sub_panel, "score_k21", anchor_col, kappa, h3_construction,
                                                     fwd_col, n_draws=PERMUTATION_DRAWS)
        cross_p = float((1 + np.sum(cross_null >= observed_ir)) / (1 + len(cross_null)))
        within_p = float((1 + np.sum(within_null >= observed_ir)) / (1 + len(within_null)))
        return {"observed_ir": observed_ir,
                "cross_date": {"null_mean": float(np.nanmean(cross_null)), "p_value": cross_p},
                "within_date": {"null_mean": float(np.nanmean(within_null)), "p_value": within_p}}

    perm_pre = _perm_summary(pre_oos_panel, blend_summary["pre2024"]["ir_annualized"])
    perm_post = _perm_summary(post_oos_panel, blend_summary["post2024"]["ir_annualized"])

    # --- Validation: quintile monotonicity, both splits ---
    monotone_pre, means_pre = quintile_monotonicity(pre_oos_panel, "score_k21", f"fwd_rel_return_k{K_PRIMARY}")
    monotone_post, means_post = quintile_monotonicity(post_oos_panel, "score_k21", f"fwd_rel_return_k{K_PRIMARY}")

    # --- Gate (H3_PORTFOLIO_CONSTRUCTION_PLAN.md Success/Failure Criteria) ---
    gate = {
        "cross_date_p_pre_below_0.10": perm_pre["cross_date"]["p_value"] < P_THRESHOLD,
        "within_date_p_pre_below_0.10": perm_pre["within_date"]["p_value"] < P_THRESHOLD,
        "cross_date_p_post_below_0.10": perm_post["cross_date"]["p_value"] < P_THRESHOLD,
        "within_date_p_post_below_0.10": perm_post["within_date"]["p_value"] < P_THRESHOLD,
        "quintile_monotone_pre2024": bool(monotone_pre),
        "quintile_monotone_post2024": bool(monotone_post),
        "ir_delta_ci_excludes_zero_pre2024": bool(np.isfinite(ir_delta_pre[1]) and ir_delta_pre[1] > 0),
        "ir_delta_ci_excludes_zero_post2024": bool(np.isfinite(ir_delta_post[1]) and ir_delta_post[1] > 0),
        "anchor_alone_beats_bova11_pre2024": bool(ir_stats(anchor_vs_bova11_pre)["ir_annualized"] > 0),
        "anchor_alone_beats_cdi_pre2024": bool(ir_stats(anchor_vs_cdi_pre)["mean_monthly_active"] > 0),
    }
    verdict = "PASS" if all(gate.values()) else "FAIL"

    output = {
        "verdict": verdict,
        "gate": gate,
        "survivors_used": survivors,
        "model": {"class": model_result["model_class"], "hyperparams": model_result["hyperparams"],
                   "mean_pre2024_oof_ic": ic_oof, "diagnostics": model_result["diagnostics"]},
        "anchor": {"winning_type": winning_anchor_type, "by_type": anchor_type_summary},
        "kappa": kappa,
        "blend": blend_summary,
        "ir_delta_bootstrap_ci": {"pre2024": ir_delta_pre, "post2024": ir_delta_post},
        "permutation_null": {"pre2024": perm_pre, "post2024": perm_post},
        "quintile_monotonicity": {
            "pre2024": {"monotone": monotone_pre, "means": means_pre.to_dict()},
            "post2024": {"monotone": monotone_post, "means": means_post.to_dict()},
        },
    }
    H3_FINDINGS_JSON.write_text(json.dumps(output, indent=2, default=str))
    _write_findings_md(output)
    print(f"H3 verdict: {verdict}. model={model_result['model_class']}, "
          f"anchor={winning_anchor_type}, kappa={kappa:.3f}. Findings: {H3_FINDINGS_MD}")


def _write_findings_md(o: dict) -> None:
    lines = [
        "# H3 Findings — Fitted Portfolio Construction",
        "",
        f"**Verdict: {o['verdict']}**",
        "",
        "Gate: blended IR exceeds anchor-alone IR with both cross-date and within-date "
        "permutation-null tests at p < 0.10 on both pre-2024 and post-2024 splits; OOS "
        "score-quintile monotonicity on both splits; bootstrap CI on the IR delta "
        "(blended vs. anchor-alone) excludes 0 on both splits; the anchor alone beats "
        "BOVA11 and CDI pre-2024.",
        "",
        "## Gate checklist",
        "",
        json.dumps(o["gate"], indent=2),
        "",
        "## Combination layer (Design §a): model class selected by walk-forward CV",
        "",
        json.dumps(o["model"], indent=2, default=str),
        "",
        "## Anchor layer (Design §b): type selected by pre-2024 anchor-alone Sharpe/IR",
        "",
        json.dumps(o["anchor"], indent=2, default=str),
        "",
        f"## Blend layer (Design §c): kappa = {o['kappa']:.4f}, derived from the winning "
        "model's own pre-2024 OOF mean Spearman IC",
        "",
        json.dumps(o["blend"], indent=2, default=str),
        "",
        "## Bootstrap CI on the IR delta (blended vs. anchor-alone)",
        "",
        json.dumps(o["ir_delta_bootstrap_ci"], indent=2, default=str),
        "",
        "## Permutation-null tests (cross-date and within-date, both splits)",
        "",
        json.dumps(o["permutation_null"], indent=2, default=str),
        "",
        "## Quintile monotonicity",
        "",
        json.dumps(o["quintile_monotonicity"], indent=2, default=str),
        "",
    ]
    H3_FINDINGS_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
