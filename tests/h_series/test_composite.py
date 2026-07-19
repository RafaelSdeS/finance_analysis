"""
Test: h_series/composite.py's walk-forward ridge scoring -- signal recovery
against a pure-noise feature, a no-lookahead regression (mutating a LATER
fold's data must never change an EARLIER fold's OOS predictions), and proof
that lambda selection ignores the post-2024 confirmation segment entirely.

Synthetic panel, deterministic seed. No parquet IO.

Run from project root:
    python tests/h_series/test_composite.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.h_series.composite import build_feature_matrix, select_lambda, walk_forward_scores  # noqa: E402
from src.h_series.spine import iter_expanding_folds  # noqa: E402
from src.h_series.stats import spearman_ic_by_group  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402

N_TICKERS = 30
N_DATES = 60  # monthly, 5 years -> several expanding folds


def _synthetic_panel(seed: int = 42) -> pd.DataFrame:
    """target = 2*signal_char + noise; noise_char is pure random, uncorrelated
    with target. Both characteristics are already in the [-0.5, 0.5]
    rank-normalized-ish range (uniform) so build_feature_matrix's own
    rank_normalize doesn't distort the planted relationship's sign/shape."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=N_DATES, freq="ME")
    tickers = [f"T{i}" for i in range(N_TICKERS)]
    rows = []
    for d in dates:
        signal = rng.uniform(-0.5, 0.5, N_TICKERS)
        noise_char = rng.uniform(-0.5, 0.5, N_TICKERS)
        target = 2.0 * signal + rng.normal(0, 0.05, N_TICKERS)
        for i, tk in enumerate(tickers):
            rows.append({"decision_date": d, "ticker": tk, "sector": "S1",
                         "signal_char": signal[i], "noise_char": noise_char[i],
                         "freshness_factor": 1.0, "target": target[i]})
    panel = pd.DataFrame(rows)
    panel["target_rank"] = panel.groupby("decision_date")["target"].rank(pct=True) - 0.5
    return panel


def test_ridge_recovers_signal_and_downweights_noise(passed, failed):
    panel = _synthetic_panel()
    X, feature_cols = build_feature_matrix(panel, ["signal_char", "noise_char"])
    folds = iter_expanding_folds(panel["decision_date"].max(), "2016-12-31", 12)

    scores = walk_forward_scores(panel, X, feature_cols, "target_rank", folds, alpha=1.0)
    valid = scores.notna()
    ic = spearman_ic_by_group(scores[valid], panel.loc[valid, "target_rank"], panel.loc[valid, "decision_date"])
    mean_ic = float(ic.mean())
    ic_ok = mean_ic > 0.7
    print_check("walk_forward_scores: stitched OOS score strongly recovers the planted signal",
                ic_ok, f"mean IC={mean_ic:.3f}")

    train_mask = panel["decision_date"] <= folds[-1].train_end
    model = Ridge(alpha=1.0).fit(X.loc[train_mask, feature_cols].to_numpy(), panel.loc[train_mask, "target_rank"])
    coef = dict(zip(feature_cols, model.coef_))
    downweighted = abs(coef["noise_char_h2rank"]) < abs(coef["signal_char_h2rank"]) * 0.15
    print_check("Ridge coefficient on the pure-noise characteristic is near zero relative to the signal",
                downweighted, f"signal_coef={coef['signal_char_h2rank']:.3f}, noise_coef={coef['noise_char_h2rank']:.3f}")

    ok = ic_ok and downweighted
    return passed + ok, failed + (not ok)


def test_no_lookahead_later_fold_mutation(passed, failed):
    """Mutating every row STRICTLY AFTER fold[0]'s OOS window (i.e. only data
    belonging to later folds) must leave fold[0]'s own OOS predictions
    bit-identical -- a lookahead bug (e.g. an accidental global fit, or a
    feature built with a forward-peeking window) would let later data leak
    backward into an earlier fold's fitted coefficients."""
    panel = _synthetic_panel()
    X, feature_cols = build_feature_matrix(panel, ["signal_char", "noise_char"])
    folds = iter_expanding_folds(panel["decision_date"].max(), "2016-12-31", 12)
    assert len(folds) >= 2, "fixture must produce at least 2 folds for this test to be meaningful"

    scores_before = walk_forward_scores(panel, X, feature_cols, "target_rank", folds, alpha=1.0)

    panel_mut = panel.copy()
    later_mask = panel_mut["decision_date"] > folds[0].oos_end
    rng = np.random.default_rng(999)
    panel_mut.loc[later_mask, "target_rank"] = rng.uniform(-0.5, 0.5, int(later_mask.sum()))
    panel_mut.loc[later_mask, "signal_char"] = rng.uniform(-0.5, 0.5, int(later_mask.sum()))
    X_mut, _ = build_feature_matrix(panel_mut, ["signal_char", "noise_char"])
    scores_after = walk_forward_scores(panel_mut, X_mut, feature_cols, "target_rank", folds, alpha=1.0)

    fold0_mask = (panel["decision_date"] > folds[0].train_end) & (panel["decision_date"] <= folds[0].oos_end)
    ok = bool(np.allclose(scores_before[fold0_mask].to_numpy(), scores_after[fold0_mask].to_numpy(),
                           atol=1e-12, equal_nan=True))
    print_check("walk_forward_scores: mutating LATER folds' data leaves an EARLIER fold's "
                "OOS predictions bit-identical (no lookahead)", ok)
    return passed + ok, failed + (not ok)


def test_lambda_selection_ignores_post_2024_folds(passed, failed):
    """select_lambda's IC scoring is restricted to decision_date < confirmation_start
    -- mutating ONLY post-confirmation targets (structurally uncorrelated with
    the feature, so it would visibly change the chosen lambda if it leaked in)
    must not change the chosen lambda at all."""
    dates = pd.date_range("2015-01-31", periods=130, freq="ME")  # runs well past 2024-03
    rng = np.random.default_rng(7)
    tickers = [f"T{i}" for i in range(N_TICKERS)]
    rows = []
    for d in dates:
        signal = rng.uniform(-0.5, 0.5, N_TICKERS)
        target = 2.0 * signal + rng.normal(0, 0.05, N_TICKERS)
        for i, tk in enumerate(tickers):
            rows.append({"decision_date": d, "ticker": tk, "sector": "S1",
                         "signal_char": signal[i], "freshness_factor": 1.0, "target": target[i]})
    panel = pd.DataFrame(rows)
    panel["target_rank"] = panel.groupby("decision_date")["target"].rank(pct=True) - 0.5

    X, feature_cols = build_feature_matrix(panel, ["signal_char"])
    folds = iter_expanding_folds(panel["decision_date"].max(), "2018-12-31", 12)

    lambda_before, _ = select_lambda(panel, X, feature_cols, "target_rank", folds)

    panel_mut = panel.copy()
    post_mask = panel_mut["decision_date"] >= pd.Timestamp("2024-03-22")
    assert post_mask.sum() > 0, "fixture must actually reach the confirmation segment"
    panel_mut.loc[post_mask, "target_rank"] = rng.uniform(-0.5, 0.5, int(post_mask.sum()))
    lambda_after, _ = select_lambda(panel_mut, X, feature_cols, "target_rank", folds)

    ok = lambda_before == lambda_after
    print_check("select_lambda: mutating ONLY post-2024-03-22 targets does not change the chosen lambda",
                ok, f"before={lambda_before}, after={lambda_after}")
    return passed + ok, failed + (not ok)


def main() -> int:
    print_header("h_series/composite.py (walk-forward ridge composite)")
    passed = failed = 0
    for test_fn in [
        test_ridge_recovers_signal_and_downweights_noise,
        test_no_lookahead_later_fold_mutation,
        test_lambda_selection_ignores_post_2024_folds,
    ]:
        passed, failed = test_fn(passed, failed)
    print_section_end(passed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
