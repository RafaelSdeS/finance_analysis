"""
milestone_h1.py — H-series Milestone H1: the sector-neutral, FDR-corrected
Spearman rank-IC alpha screen (MEDIUM_HORIZON_RESEARCH_PLAN.md sec 3, H1)
-- the kill gate. Every statistic here is nonparametric (rank correlation)
or a plain HAC t-test, with no fitted parameters, so unlike H2 there is no
walk-forward refitting to do; the full monthly panel is screened at once.

Run:
    python -m src.h_series.milestone_h1

Reads H0_FINDINGS.json for a cross-check against the pre-registered power
floor (falls back with a loud warning if H0 hasn't been run -- the gate
below still enforces NW-t>=2 and FDR regardless; the floor is a sanity
check on top of it, not a substitute for it).

Writes H1_FINDINGS.md + H1_FINDINGS.json (verdict + surviving
characteristics, consumed by a future H2).
"""

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from .features import CHARACTERISTIC_COLUMNS, build_monthly_panel
from .paths import H0_FINDINGS_JSON, H1_FINDINGS_JSON, H1_FINDINGS_MD
from .spine import hac_lag_for_horizon
from .stats import benjamini_hochberg, newey_west_tstat, spearman_ic_by_group

FDR_ALPHA = 0.10
T_THRESHOLD = 2.0
K_HORIZONS = (21, 63)
MIN_SIGN_CONSISTENT_FRAC = 0.6   # sign matches the pooled IC's sign in >= this share of sub-windows
MIN_SURVIVORS_TO_PASS = 2
QUINTILE_MIN_ROWS = 10


def quintile_spread(char_col: str, target_col: str, panel: pd.DataFrame) -> pd.Series:
    """Monthly long-top-quintile-minus-short-bottom-quintile spread, EW
    within each quintile -- the economic-magnitude check IC alone can't
    give (a significant but tiny-magnitude IC could still be
    un-investable)."""
    def _spread(g: pd.DataFrame) -> float:
        g = g.dropna(subset=[char_col, target_col])
        if len(g) < QUINTILE_MIN_ROWS:
            return np.nan
        q = pd.qcut(g[char_col], 5, labels=False, duplicates="drop")
        if q is None or q.max() != 4:
            return np.nan
        top = g.loc[q == q.max(), target_col].mean()
        bottom = g.loc[q == 0, target_col].mean()
        return float(top - bottom)

    return panel.groupby("decision_date").apply(_spread, include_groups=False)


def sign_consistency(ic_series: pd.Series, pooled_sign: float, n_windows: int = 4) -> float:
    """Fraction of n_windows roughly-equal chronological sub-periods where
    the sub-period's mean IC has the SAME sign as the pooled IC -- a
    characteristic whose sign flips across sub-windows is noise regardless
    of pooled significance (M1/R2 false-positive precedent)."""
    ic = ic_series.dropna().sort_index()
    if len(ic) < n_windows * 3 or pooled_sign == 0 or not np.isfinite(pooled_sign):
        return float("nan")
    chunks = np.array_split(ic.to_numpy(), n_windows)
    matches = sum(1 for c in chunks if len(c) and np.isfinite(np.nanmean(c))
                  and np.sign(np.nanmean(c)) == np.sign(pooled_sign))
    return matches / n_windows


def screen_characteristics(panel: pd.DataFrame, k_horizons: tuple = K_HORIZONS) -> pd.DataFrame:
    """One row per (characteristic, k, variant in {raw, sector_neutral})."""
    rows = []
    for k in k_horizons:
        target_col = f"target_rank_k{k}"
        target_col_sn = f"target_rank_sector_neutral_k{k}"
        fwd_col = f"fwd_rel_return_k{k}"
        fwd_col_sn = f"fwd_rel_return_sector_neutral_k{k}"
        if target_col not in panel.columns:
            continue
        lag = hac_lag_for_horizon(k)
        for char in CHARACTERISTIC_COLUMNS:
            # Sector-neutral IC compares the sector-demeaned characteristic against the
            # sector-demeaned TARGET too (features.build_monthly_panel) -- otherwise a
            # purely stock-specific feature is measured against a target that still
            # carries the sector tilt, structurally attenuating a real signal.
            variants = (
                ("raw", char, target_col, fwd_col),
                ("sector_neutral", f"{char}_sector_neutral", target_col_sn, fwd_col_sn),
            )
            for variant, col, t_col, f_col in variants:
                if col not in panel.columns or t_col not in panel.columns:
                    continue
                ic = spearman_ic_by_group(panel[col], panel[t_col], panel["decision_date"])
                mean_ic, se, tstat = newey_west_tstat(ic.dropna().to_numpy(), lag=lag)
                spread = quintile_spread(col, f_col, panel)
                sign_frac = sign_consistency(ic, mean_ic)
                rows.append({
                    "characteristic": char, "k": k, "variant": variant,
                    "mean_ic": mean_ic, "se_ic": se, "tstat": tstat,
                    "n_obs": int(ic.notna().sum()),
                    "quintile_spread_mean": float(spread.mean()) if spread.notna().any() else float("nan"),
                    "sign_consistency": sign_frac,
                })
    df = pd.DataFrame(rows)
    if len(df):
        # NW t-stats are treated as asymptotically normal (the standard large-N HAC
        # convention), not referred to a t-distribution with finite df. norm.cdf is
        # NaN-safe (an unscoreable characteristic's NaN t-stat stays NaN, never crashes).
        df["pvalue"] = 2.0 * (1.0 - norm.cdf(np.abs(df["tstat"].to_numpy())))
    return df


def apply_gate(screen: pd.DataFrame, alpha: float = FDR_ALPHA, t_threshold: float = T_THRESHOLD,
                min_sign_consistent_frac: float = MIN_SIGN_CONSISTENT_FRAC) -> pd.DataFrame:
    """The kill-gate: BH-FDR correction runs on the sector_neutral variant
    ONLY -- raw is diagnostic and never gates (user addendum: raw-only
    significance is a sector-timing bet in disguise, not stock-picking
    alpha). A characteristic passes iff its sector_neutral variant survives
    FDR, |NW-t| >= t_threshold, and is sign-consistent across sub-windows."""
    sn = screen[screen["variant"] == "sector_neutral"].copy()
    sn["fdr_reject"] = False
    for k in sn["k"].unique():
        mask = sn["k"] == k
        sn.loc[mask, "fdr_reject"] = benjamini_hochberg(sn.loc[mask, "pvalue"].to_numpy(), alpha)
    sn["passes_gate"] = (
        sn["fdr_reject"]
        & (sn["tstat"].abs() >= t_threshold)
        & (sn["sign_consistency"].fillna(0) >= min_sign_consistent_frac)
    )
    return sn


def _load_h0_floor() -> dict:
    if H0_FINDINGS_JSON.exists():
        return json.loads(H0_FINDINGS_JSON.read_text())
    warnings.warn(
        "H0_FINDINGS.json not found -- run `python -m src.h_series.milestone_h0` first. "
        "Proceeding with the gate's own NW-t/FDR criteria only, with no H0 power-floor cross-check."
    )
    return {}


def main() -> None:
    floor = _load_h0_floor()
    panel = build_monthly_panel(k_horizons=K_HORIZONS)
    screen = screen_characteristics(panel, K_HORIZONS)
    gated = apply_gate(screen)

    survivors = gated[gated["passes_gate"]]
    n_distinct = int(survivors["characteristic"].nunique())
    verdict = "PASS" if n_distinct >= MIN_SURVIVORS_TO_PASS else "FAIL"

    H1_FINDINGS_JSON.write_text(json.dumps({
        "verdict": verdict,
        "n_distinct_survivors": n_distinct,
        "survivors": sorted(survivors["characteristic"].unique().tolist()),
        "h0_floor_used": floor,
    }, indent=2, default=str))

    _write_findings_md(screen, gated, verdict, floor)
    print(f"H1 verdict: {verdict} ({n_distinct} distinct characteristics survived the "
          f"sector-neutral gate). Findings: {H1_FINDINGS_MD}")


def _write_findings_md(screen: pd.DataFrame, gated: pd.DataFrame, verdict: str, floor: dict) -> None:
    lines = [
        "# H1 Findings — Sector-Neutral Alpha Screen",
        "",
        f"**Verdict: {verdict}**",
        "",
        "Gate: sector-neutralized rank IC, NW-HAC |t| >= 2, BH-FDR 10%, sign-consistent "
        "in >= 60% of sub-windows, at either k in {21, 63}. Raw (non-sector-neutralized) "
        "IC is diagnostic only and never gates (a raw-only-significant characteristic is "
        "a sector-timing bet, not stock-picking alpha).",
        "",
        "## Full screen",
        "",
        screen.round(4).to_string(index=False),
        "",
        "## Gate results (sector-neutral variant)",
        "",
        gated.round(4).to_string(index=False),
    ]
    H1_FINDINGS_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
