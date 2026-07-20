"""
composite.py — H-series Milestone H2/H3: per-fold composite over the H1
survivors, walk-forward stitched OOS score (MEDIUM_HORIZON_RESEARCH_PLAN.md
sec 3, H2; H3_PORTFOLIO_CONSTRUCTION_PLAN.md Design §a).

Reuses features.build_monthly_panel() (characteristics + raw targets, already
built for H1) and spine.iter_expanding_folds() (the same expanding-window
spine as H0/H1). select_lambda() (H2, unchanged, ridge-only -- "linear-on-
ranks is the max complexity this dataset has earned" was H2's finding, not
a permanent ceiling) is kept exactly as it ran for H2's locked verdict.
select_model() (H3) widens this into a walk-forward-CV competition between
ridge/ElasticNet/a regularization-constrained shallow GBM -- picking the
model CLASS by the same pre-2024 mean OOS Spearman IC criterion is more
data-driven than fixing it by assertion, not less.
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge

from .paths import H1_FINDINGS_JSON
from .stats import rank_normalize, spearman_ic_by_group

# The 3 filing-derived survivors get a cardinal freshness x characteristic
# interaction column (sec 1.B's exp(-t/45) budget: "at most 2-3 gated
# columns, not a gated copy of everything").
FRESHNESS_INTERACTION_CHARS = ("pl", "pvp", "net_margin")

LAMBDA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
CONFIRMATION_START = pd.Timestamp("2024-03-22")  # sec 0.4's untouched-until-the-end test split
MIN_TRAIN_ROWS = 30  # a fold with fewer training rows than this is skipped, not fit on noise

# H3 Design §a: 2 extra model-class candidates competed against ridge, kept
# small (not an open-ended search) for the same reason the original ladder
# (ridge only) was chosen -- letting walk-forward CV pick the class is more
# data-driven than fixing it by assertion, not an invitation to widen further.
ELASTICNET_ALPHA_GRID = (0.01, 0.1, 1.0)
ELASTICNET_L1_RATIO_GRID = (0.2, 0.5, 0.8)
GBM_MIN_LEAF_FRAC = 0.10  # min_samples_leaf floor as a fraction of the largest pre-2024
                          # fold's actual training-row count (see _representative_pre_train_rows)
                          # -- data-derived, not a fixed absolute copied from a larger-data default.
GBM_DISQUALIFY_GAP_MULTIPLE = 2.0  # mechanical rule (Design §a): GBM loses to ridge/ElasticNet
                                    # if it wins CV-IC but overfits its own pre-2024 folds this much harder.


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
                         target_col: str, folds: list, alpha: float,
                         model_factory=None) -> pd.Series:
    """Model refit per expanding fold on decision_date <= train_end (rows
    with a NaN target dropped from FITTING only -- prediction still covers
    every row in the fold's OOS window regardless of target availability,
    since scores feed portfolio construction, not just IC measurement).
    Returns a score Series aligned to panel's index, NaN outside every
    fold's OOS window (before the first fold's train_end).

    model_factory, if given, is a zero-arg callable returning a fresh
    unfitted estimator per fold (H3 Design §a's model-class competition,
    composite.py's select_model) -- `alpha` is then ignored. Default (None)
    reproduces H2's exact original Ridge(alpha) behavior, unchanged."""
    scores = pd.Series(np.nan, index=panel.index)
    dates = panel["decision_date"]
    y = panel[target_col]
    Xv = X[feature_cols].to_numpy()

    for fold in folds:
        train_mask = ((dates <= fold.train_end) & y.notna()).to_numpy()
        oos_mask = ((dates > fold.train_end) & (dates <= fold.oos_end)).to_numpy()
        if train_mask.sum() < MIN_TRAIN_ROWS or oos_mask.sum() == 0:
            continue
        model = model_factory() if model_factory is not None else Ridge(alpha=alpha)
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


# ---------------------------------------------------------------------------
# H3 Design §a: model-class competition (ridge / ElasticNet / shallow GBM)
# ---------------------------------------------------------------------------

def _representative_pre_train_rows(panel: pd.DataFrame, target_col: str, folds: list,
                                    confirmation_start: pd.Timestamp = CONFIRMATION_START) -> int:
    """Largest pre-2024 fold's actual training-row count -- used once to
    size the GBM candidate's min_samples_leaf floor (Design §a) before any
    OOS scoring happens, same "fixed in advance, not swept against the
    reported metric" discipline as everything else selected in this
    module. The LARGEST pre-2024 fold (not the first/smallest) is used
    because it's the most representative of the steady-state training size
    the frozen hyperparameter will actually see across most of the walk-
    forward run."""
    dates = panel["decision_date"]
    y = panel[target_col]
    counts = [int(((dates <= f.train_end) & y.notna()).sum()) for f in folds if f.train_end < confirmation_start]
    return max(counts) if counts else MIN_TRAIN_ROWS


def _gbm_factory(min_samples_leaf: int):
    return lambda: GradientBoostingRegressor(
        max_depth=3, max_features="sqrt", min_samples_leaf=min_samples_leaf, random_state=0)


def _mean_pre_ic(scores: pd.Series, panel: pd.DataFrame, target_col: str, pre_mask: np.ndarray) -> float:
    dates = panel["decision_date"]
    rank_scores = rank_normalize(scores, dates)
    ic = spearman_ic_by_group(rank_scores[pre_mask], panel.loc[pre_mask, target_col], dates[pre_mask])
    return float(ic.mean()) if ic.notna().any() else -np.inf


def _train_oos_ic_gap(panel: pd.DataFrame, X: pd.DataFrame, feature_cols: list, target_col: str,
                       folds: list, model_factory,
                       confirmation_start: pd.Timestamp = CONFIRMATION_START) -> tuple:
    """Per-pre-2024-fold train-IC vs OOS-IC, averaged across folds --
    diagnostic for the GBM overfit-disqualification rule (Design §a): a
    model that fits its own training folds far better than it predicts out
    of them is capturing noise, not signal, regardless of what its
    OOS-selection IC says. Returns (mean_train_ic, mean_oos_ic, gap)."""
    dates = panel["decision_date"]
    y = panel[target_col]
    Xv = X[feature_cols].to_numpy()
    yv = y.to_numpy()
    dv = dates.to_numpy()

    train_ics, oos_ics = [], []
    for fold in folds:
        if fold.train_end >= confirmation_start:
            continue
        train_mask = ((dates <= fold.train_end) & y.notna()).to_numpy()
        oos_mask = ((dates > fold.train_end) & (dates <= fold.oos_end) & y.notna()).to_numpy()
        if train_mask.sum() < MIN_TRAIN_ROWS or oos_mask.sum() == 0:
            continue
        model = model_factory()
        model.fit(Xv[train_mask], yv[train_mask])

        tr_rank = rank_normalize(pd.Series(model.predict(Xv[train_mask])), pd.Series(dv[train_mask]))
        tr_ic = spearman_ic_by_group(tr_rank, pd.Series(yv[train_mask]), pd.Series(dv[train_mask])).mean()

        oo_rank = rank_normalize(pd.Series(model.predict(Xv[oos_mask])), pd.Series(dv[oos_mask]))
        oo_ic = spearman_ic_by_group(oo_rank, pd.Series(yv[oos_mask]), pd.Series(dv[oos_mask])).mean()

        if np.isfinite(tr_ic) and np.isfinite(oo_ic):
            train_ics.append(tr_ic)
            oos_ics.append(oo_ic)

    if not train_ics:
        return float("nan"), float("nan"), float("nan")
    mean_train = float(np.mean(train_ics))
    mean_oos = float(np.mean(oos_ics))
    return mean_train, mean_oos, mean_train - mean_oos


def select_model(panel: pd.DataFrame, X: pd.DataFrame, feature_cols: list, target_col: str,
                  folds: list, confirmation_start: pd.Timestamp = CONFIRMATION_START) -> dict:
    """H3 Design §a: competes ridge / ElasticNet / a regularization-
    constrained shallow GBM by the same pre-2024-only mean OOS Spearman IC
    criterion select_lambda already uses -- letting walk-forward CV choose
    the model class is more data-driven than fixing it by assertion
    (this module's prior ridge-only docstring), not less.

    GBM is mechanically disqualified, not by manual review, if it wins the
    raw CV-IC selection but its pre-2024 train/OOS IC gap exceeds
    GBM_DISQUALIFY_GAP_MULTIPLE x ridge's own gap on the same folds --
    select_model then falls back to the best of ridge/ElasticNet.

    Returns {"model_class", "hyperparams", "scores" (stitched, rank-
    normalized, same shape as select_lambda's second return value),
    "diagnostics"} -- diagnostics always reports every class's IC and the
    train/OOS gap(s) actually computed, not just the winner, per Design §a's
    "must be logged per model class, not just the winner picked silently."
    """
    dates = panel["decision_date"]
    pre_mask = (dates < confirmation_start).to_numpy()

    candidates = {}  # class -> (hyperparams, mean_pre_ic, stitched_rank_scores, model_factory)

    best_lambda, ridge_scores = select_lambda(panel, X, feature_cols, target_col, folds,
                                               confirmation_start=confirmation_start)
    ridge_ic = _mean_pre_ic(ridge_scores, panel, target_col, pre_mask)
    candidates["ridge"] = ({"alpha": best_lambda}, ridge_ic, ridge_scores,
                            lambda a=best_lambda: Ridge(alpha=a))

    best_en_params, best_en_ic, best_en_scores = None, -np.inf, None
    for a in ELASTICNET_ALPHA_GRID:
        for l1 in ELASTICNET_L1_RATIO_GRID:
            factory = lambda a=a, l1=l1: ElasticNet(alpha=a, l1_ratio=l1, max_iter=10000)
            scores = walk_forward_scores(panel, X, feature_cols, target_col, folds, 0.0,
                                          model_factory=factory)
            ic = _mean_pre_ic(scores, panel, target_col, pre_mask)
            if ic > best_en_ic:
                best_en_params = {"alpha": a, "l1_ratio": l1}
                best_en_ic = ic
                best_en_scores = rank_normalize(scores, dates)
    candidates["elasticnet"] = (best_en_params, best_en_ic, best_en_scores,
                                 lambda a=best_en_params["alpha"], l1=best_en_params["l1_ratio"]:
                                 ElasticNet(alpha=a, l1_ratio=l1, max_iter=10000))

    n_rep = _representative_pre_train_rows(panel, target_col, folds, confirmation_start)
    min_leaf = max(5, round(GBM_MIN_LEAF_FRAC * n_rep))
    gbm_factory = _gbm_factory(min_leaf)
    gbm_raw_scores = walk_forward_scores(panel, X, feature_cols, target_col, folds, 0.0,
                                          model_factory=gbm_factory)
    gbm_ic = _mean_pre_ic(gbm_raw_scores, panel, target_col, pre_mask)
    candidates["gbm"] = ({"max_depth": 3, "max_features": "sqrt", "min_samples_leaf": min_leaf},
                          gbm_ic, rank_normalize(gbm_raw_scores, dates), gbm_factory)

    winner = max(candidates, key=lambda k: candidates[k][1])

    _, _, ridge_gap = _train_oos_ic_gap(panel, X, feature_cols, target_col, folds,
                                         candidates["ridge"][3], confirmation_start)
    diagnostics = {
        "ic_by_class": {k: v[1] for k, v in candidates.items()},
        "hyperparams_by_class": {k: v[0] for k, v in candidates.items()},
        "ridge_train_oos_ic_gap": ridge_gap,
    }

    if winner == "gbm":
        _, _, gbm_gap = _train_oos_ic_gap(panel, X, feature_cols, target_col, folds,
                                           candidates["gbm"][3], confirmation_start)
        diagnostics["gbm_train_oos_ic_gap"] = gbm_gap
        disqualified = not np.isfinite(ridge_gap) or gbm_gap > GBM_DISQUALIFY_GAP_MULTIPLE * ridge_gap
        diagnostics["gbm_disqualified"] = bool(disqualified)
        if disqualified:
            winner = "ridge" if candidates["ridge"][1] >= candidates["elasticnet"][1] else "elasticnet"

    hyperparams, ic, scores, _ = candidates[winner]
    return {"model_class": winner, "hyperparams": hyperparams, "mean_pre_ic": ic,
            "scores": scores, "diagnostics": diagnostics}


def _demo() -> None:
    """Runnable self-check (synthetic data, no real dataset needed): a small
    panel with one genuinely predictive feature and one pure-noise feature
    -- checks select_model() runs end to end, every candidate class's IC is
    logged (not just the winner's), and the GBM train/OOS gap diagnostic is
    present and finite whenever GBM actually wins."""
    import numpy as np

    from .spine import FoldWindow

    rng = np.random.default_rng(0)
    n_dates, n_tickers = 40, 20
    dates = pd.date_range("2015-01-31", periods=n_dates, freq="ME")
    rows = []
    for d in dates:
        signal = rng.normal(0, 1, n_tickers)
        target = 0.05 * signal + rng.normal(0, 1, n_tickers)
        for i in range(n_tickers):
            rows.append({"decision_date": d, "ticker": f"T{i}",
                         "feat_signal": signal[i], "feat_noise": rng.normal(0, 1),
                         "target_rank_k21": target[i]})
    panel = pd.DataFrame(rows)
    feature_cols = ["feat_signal", "feat_noise"]
    X = panel[feature_cols]

    folds = [FoldWindow(fold_id="fold0", train_end=dates[19], oos_end=dates[39])]
    result = select_model(panel, X, feature_cols, "target_rank_k21", folds,
                           confirmation_start=dates[30])

    assert result["model_class"] in ("ridge", "elasticnet", "gbm")
    assert set(result["diagnostics"]["ic_by_class"]) == {"ridge", "elasticnet", "gbm"}
    assert len(result["scores"]) == len(panel)
    if result["model_class"] == "gbm":
        assert np.isfinite(result["diagnostics"]["gbm_train_oos_ic_gap"])

    print("composite.select_model self-check: OK "
          f"(winner={result['model_class']}, ic_by_class={result['diagnostics']['ic_by_class']})")


if __name__ == "__main__":
    _demo()
