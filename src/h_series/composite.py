"""
composite.py — H-series Milestone H2: per-fold ridge composite over the H1
survivors, walk-forward stitched OOS score (MEDIUM_HORIZON_RESEARCH_PLAN.md
sec 3, H2).

Reuses features.build_monthly_panel() (characteristics + raw targets, already
built for H1) and spine.iter_expanding_folds() (the same expanding-window
spine as H0/H1). Ridge is intentionally the only model here (sec 1.C's
ladder: linear-on-ranks is the max complexity this dataset has earned).
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .paths import H1_FINDINGS_JSON
from .stats import rank_normalize, spearman_ic_by_group

# The 3 filing-derived survivors get a cardinal freshness x characteristic
# interaction column (sec 1.B's exp(-t/45) budget: "at most 2-3 gated
# columns, not a gated copy of everything").
FRESHNESS_INTERACTION_CHARS = ("pl", "pvp", "net_margin")

LAMBDA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
CONFIRMATION_START = pd.Timestamp("2024-03-22")  # sec 0.4's untouched-until-the-end test split
MIN_TRAIN_ROWS = 30  # a fold with fewer training rows than this is skipped, not fit on noise


def load_survivors() -> list:
    """H1's surviving characteristics -- H2 must never run against a stale
    or absent H1 PASS, and never silently falls back to the full candidate
    list."""
    if not H1_FINDINGS_JSON.exists():
        raise FileNotFoundError(
            f"{H1_FINDINGS_JSON} not found -- run `python -m src.h_series.milestone_h1` first."
        )
    data = json.loads(H1_FINDINGS_JSON.read_text())
    if data.get("verdict") != "PASS":
        raise ValueError(f"{H1_FINDINGS_JSON} verdict is {data.get('verdict')!r}, not PASS.")
    survivors = data.get("survivors")
    if not survivors:
        raise ValueError(f"{H1_FINDINGS_JSON} has no survivors.")
    return sorted(survivors)


def build_feature_matrix(panel: pd.DataFrame, survivors: list, sector_neutral: bool = False) -> tuple:
    """Rank-normalized survivor characteristics (NaN -> 0.0, cross-sectional
    median = neutral) plus freshness x characteristic interaction columns for
    the filing-derived survivors present in `survivors`. sector_neutral=True
    reads each survivor's `_sector_neutral` column instead (ablation (i)).
    Returns (X, feature_cols); X is row-aligned with panel (same index)."""
    X = pd.DataFrame(index=panel.index)
    feature_cols = []
    for char in survivors:
        src_col = f"{char}_sector_neutral" if sector_neutral else char
        rank_col = f"{char}_h2rank"
        X[rank_col] = rank_normalize(panel[src_col], panel["decision_date"]).fillna(0.0)
        feature_cols.append(rank_col)
    for char in FRESHNESS_INTERACTION_CHARS:
        if char not in survivors:
            continue
        fresh_col = f"{char}_freshness_x"
        X[fresh_col] = X[f"{char}_h2rank"] * panel["freshness_factor"].fillna(0.0)
        feature_cols.append(fresh_col)
    return X, feature_cols


def walk_forward_scores(panel: pd.DataFrame, X: pd.DataFrame, feature_cols: list,
                         target_col: str, folds: list, alpha: float) -> pd.Series:
    """Ridge(alpha) refit per expanding fold on decision_date <= train_end
    (rows with a NaN target dropped from FITTING only -- prediction still
    covers every row in the fold's OOS window regardless of target
    availability, since scores feed portfolio construction, not just IC
    measurement). Returns a score Series aligned to panel's index, NaN
    outside every fold's OOS window (before the first fold's train_end)."""
    scores = pd.Series(np.nan, index=panel.index)
    dates = panel["decision_date"]
    y = panel[target_col]
    Xv = X[feature_cols].to_numpy()

    for fold in folds:
        train_mask = ((dates <= fold.train_end) & y.notna()).to_numpy()
        oos_mask = ((dates > fold.train_end) & (dates <= fold.oos_end)).to_numpy()
        if train_mask.sum() < MIN_TRAIN_ROWS or oos_mask.sum() == 0:
            continue
        model = Ridge(alpha=alpha)
        model.fit(Xv[train_mask], y.to_numpy()[train_mask])
        scores.iloc[oos_mask] = model.predict(Xv[oos_mask])
    return scores


def select_lambda(panel: pd.DataFrame, X: pd.DataFrame, feature_cols: list,
                   target_col: str, folds: list, grid: tuple = LAMBDA_GRID,
                   confirmation_start: pd.Timestamp = CONFIRMATION_START) -> tuple:
    """Picks the lambda maximizing mean stitched-OOS Spearman IC on PRE-2024
    folds only (protocol sec 2: the 2024-03->2026-07 segment is untouched by
    any hyperparameter choice, exactly like the split's own precedent).
    Returns (best_lambda, best_rank_normalized_score_series) -- the score
    series is already stitched across ALL folds (including the final
    confirmation segment) with the frozen lambda, scored once, per plan."""
    dates = panel["decision_date"]
    pre_mask = (dates < confirmation_start).to_numpy()

    best_lambda, best_ic, best_scores = grid[0], -np.inf, None
    for alpha in grid:
        scores = walk_forward_scores(panel, X, feature_cols, target_col, folds, alpha)
        rank_scores = rank_normalize(scores, dates)
        ic = spearman_ic_by_group(rank_scores[pre_mask], panel.loc[pre_mask, target_col],
                                   dates[pre_mask])
        mean_ic = float(ic.mean()) if ic.notna().any() else -np.inf
        if mean_ic > best_ic:
            best_lambda, best_ic, best_scores = alpha, mean_ic, rank_scores

    return best_lambda, best_scores
