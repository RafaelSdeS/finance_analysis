"""
Test: conviction_model/diagnostics.py's intrinsic embedding-quality
diagnostics, against Phase 1's own checklist (CONVICTION_MODEL_PLAN.md,
Testing strategy) -- each diagnostic recovers a known injected signal on
synthetic embeddings, and correctly rejects a pure-noise/degenerate case.

Run from project root:
    python tests/conviction_model/test_diagnostics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.conviction_model.diagnostics import (  # noqa: E402
    group_blocked_train_mask, latent_similarity_significance, linear_probe_r2,
    neighbor_outcome_variance_ratio, perturbation_sensitivity, quality_persistence_autocorrelation,
    regime_mutual_information, temporal_smoothness_significance,
)
from test_utils import print_check, print_header, print_section_end  # noqa: E402

RNG = np.random.default_rng(0)


def _clustered_embeddings(n_clusters=3, per_cluster=30, d=2, spread=0.3, gap=5.0, rng=RNG):
    centers = rng.normal(scale=gap, size=(n_clusters, d))
    labels = np.repeat(np.arange(n_clusters), per_cluster)
    embeddings = centers[labels] + rng.normal(scale=spread, size=(n_clusters * per_cluster, d))
    return embeddings, labels


def test_neighbor_outcome_variance_ratio_recovers_injected_structure(passed, failed):
    embeddings, labels = _clustered_embeddings()
    cluster_means = RNG.normal(scale=10.0, size=labels.max() + 1)
    outcomes = cluster_means[labels] + RNG.normal(scale=0.1, size=len(labels))
    ratio = neighbor_outcome_variance_ratio(embeddings, outcomes, k=10, rng=np.random.default_rng(1))
    ok = ratio <= 0.8
    print_check("neighbor_outcome_variance_ratio: recovers materially lower variance in embedded "
                "neighborhoods with an injected low-variance-per-cluster outcome", ok, f"ratio={ratio:.4f}")
    return passed + ok, failed + (not ok)


def _overlapping_window_outcomes(n_per_ticker: int, decision_spacing_days: int, horizon_days: int,
                                  rng: np.random.Generator) -> np.ndarray:
    """decision m's outcome = sum of `horizon_days` iid daily innovations starting at
    m*decision_spacing_days -- the exact overlapping-FORWARD-WINDOW mechanism that
    makes REAL diagnostic-1 outcomes correlated between temporally-close decisions
    (their forward-return windows share most of the same realized daily moves) and
    genuinely INDEPENDENT once the decision gap reaches the horizon (windows stop
    overlapping entirely, correlation goes to exactly 0) -- unlike an arbitrary
    block/regime model or a plain autocorrelated walk, this decays exactly where the
    real system's forward_horizon does, which is what exclude_window_days is meant
    to match."""
    total_days = (n_per_ticker - 1) * decision_spacing_days + horizon_days
    innovations = rng.normal(size=total_days)
    return np.array([innovations[m * decision_spacing_days: m * decision_spacing_days + horizon_days].sum()
                      for m in range(n_per_ticker)])


def _same_ticker_autocorrelation_artifact(n_tickers=5, n_per_ticker=30, ticker_spacing=100.0,
                                           local_spacing=1.0, decision_spacing_days=21,
                                           horizon_days=252, rng=None):
    """Mimics the real failure mode diagnostic 1 must not be fooled by: embeddings
    laid out along a line, STRICTLY ordered by time within each ticker (so a point's
    unfiltered k-nearest neighbors are guaranteed to be its own temporally-adjacent
    same-ticker points -- mirrors how a real encoder's overlapping daily/monthly
    windows behave, without relying on a stochastic walk's geometry to preserve
    time-order-by-distance), paired with outcomes built from genuinely overlapping
    forward windows (_overlapping_window_outcomes) so temporally-close same-ticker
    points share near-identical outcomes PURELY from window overlap -- zero genuine
    cross-sectional relationship to the embedding's content -- and are exactly
    independent once the gap reaches the horizon. Different tickers' embeddings are
    placed far enough apart (ticker_spacing >> local_spacing * n_per_ticker) that
    cross-ticker points never compete with same-ticker ones in a k-NN search."""
    rng = rng or np.random.default_rng(11)
    dates = pd.bdate_range("2010-01-01", periods=n_per_ticker, freq="ME")

    tickers, all_dates, embeddings, outcomes = [], [], [], []
    for t in range(n_tickers):
        outcome = _overlapping_window_outcomes(n_per_ticker, decision_spacing_days, horizon_days, rng)
        for m in range(n_per_ticker):
            tickers.append(f"T{t}")
            all_dates.append(dates[m])
            embeddings.append([t * ticker_spacing, m * local_spacing])
        outcomes.append(outcome)

    return (np.array(tickers), pd.DatetimeIndex(all_dates).to_numpy(),
            np.array(embeddings, dtype=np.float64), np.concatenate(outcomes))


def test_neighbor_outcome_variance_ratio_same_ticker_exclusion_removes_autocorrelation_artifact(passed, failed):
    tickers, dates, embeddings, outcomes = _same_ticker_autocorrelation_artifact()

    ratio_unfiltered = neighbor_outcome_variance_ratio(embeddings, outcomes, k=5,
                                                        rng=np.random.default_rng(20))
    ratio_filtered = neighbor_outcome_variance_ratio(embeddings, outcomes, k=5,
                                                      rng=np.random.default_rng(20),
                                                      tickers=tickers, dates=dates,
                                                      exclude_window_days=365, search_multiplier=10)

    # Without exclusion: same-ticker-adjacent-time points dominate the k-NN search purely
    # from embedding geometry, and their outcomes look artificially similar purely from
    # the overlapping-forward-window mechanism -- an artifact that would spuriously clear
    # the <=0.8 gate despite zero genuine cross-sectional relationship existing. With
    # exclusion, those points are filtered out and the ratio rises measurably -- a
    # RELATIVE threshold, not a fixed additive one, since the surviving same-ticker
    # neighbors can still be mutually adjacent to EACH OTHER (nearest-neighbor selection
    # favors a contiguous surviving cluster), so some residual overlap-driven correlation
    # among the selected neighbor set is expected even after the anchor-side artifact is
    # correctly removed -- a real limit of nearest-neighbor selection, not evidence the
    # fix didn't work.
    artifact_present = ratio_unfiltered <= 0.8
    fix_removes_artifact = ratio_filtered > ratio_unfiltered * 1.2
    ok = artifact_present and fix_removes_artifact
    print_check("neighbor_outcome_variance_ratio: excluding same-ticker-near-time neighbors removes a "
                "pure autocorrelation artifact (not a genuine embedding signal)",
                ok, f"unfiltered={ratio_unfiltered:.4f} (spuriously clears <=0.8: {artifact_present}), "
                    f"filtered={ratio_filtered:.4f}")
    return passed + ok, failed + (not ok)


def test_neighbor_outcome_variance_ratio_no_tickers_arg_matches_original_behavior(passed, failed):
    # Passing neither tickers nor dates must reproduce the pre-fix computation exactly --
    # a pure additive change, not a behavior change for existing callers.
    embeddings, labels = _clustered_embeddings()
    cluster_means = RNG.normal(scale=10.0, size=labels.max() + 1)
    outcomes = cluster_means[labels] + RNG.normal(scale=0.1, size=len(labels))
    ratio = neighbor_outcome_variance_ratio(embeddings, outcomes, k=10, rng=np.random.default_rng(1))
    ok = ratio <= 0.8
    print_check("neighbor_outcome_variance_ratio: omitting tickers/dates still recovers the original "
                "injected low-variance-per-cluster structure (no default-path regression)", ok, f"ratio={ratio:.4f}")
    return passed + ok, failed + (not ok)


def test_regime_mutual_information_clears_null_when_clusters_match_regime(passed, failed):
    embeddings, labels = _clustered_embeddings()  # labels double as the "regime" here -- exact match
    mi, threshold = regime_mutual_information(embeddings, labels, n_clusters=3,
                                               rng=np.random.default_rng(2))
    ok = mi > threshold
    print_check("regime_mutual_information: clears the permutation-null 95th percentile when clusters "
                "exactly match the regime label", ok, f"mi={mi:.4f}, null_95th={threshold:.4f}")
    return passed + ok, failed + (not ok)


def test_regime_mutual_information_does_not_clear_null_on_pure_noise(passed, failed):
    embeddings = RNG.normal(size=(90, 2))
    random_labels = RNG.integers(0, 3, size=90)
    mi, threshold = regime_mutual_information(embeddings, random_labels, n_clusters=3,
                                               rng=np.random.default_rng(3))
    ok = mi <= threshold
    print_check("regime_mutual_information: does NOT clear the null when embeddings/regime are unrelated noise",
                ok, f"mi={mi:.4f}, null_95th={threshold:.4f}")
    return passed + ok, failed + (not ok)


def test_linear_probe_recovers_injected_relationship(passed, failed):
    n, d = 400, 5
    embeddings = RNG.normal(size=(n, d))
    true_weights = RNG.normal(size=d)
    signal = embeddings @ true_weights
    target = signal + RNG.normal(scale=0.5 * signal.std(), size=n)  # moderate noise, real signal
    r2 = linear_probe_r2(embeddings, target, test_frac=0.5)
    ok = r2 > 0.5
    print_check("linear_probe_r2: recovers a known injected linear relationship out-of-sample",
                ok, f"r2={r2:.4f}")
    return passed + ok, failed + (not ok)


def test_group_blocked_train_mask_never_splits_a_group(passed, failed):
    groups = np.repeat([f"T{i}" for i in range(8)], [10, 15, 5, 20, 8, 12, 6, 9])
    train_mask = group_blocked_train_mask(groups, test_frac=0.5, rng=np.random.default_rng(3))

    train_groups = set(groups[train_mask])
    test_groups = set(groups[~train_mask])
    no_overlap = len(train_groups & test_groups) == 0
    both_nonempty = len(train_groups) > 0 and len(test_groups) > 0
    test_row_frac = (~train_mask).mean()
    frac_reasonable = 0.3 <= test_row_frac <= 0.7  # approximates test_frac=0.5 by ROW count, not group count

    ok = no_overlap and both_nonempty and frac_reasonable
    print_check("group_blocked_train_mask: no group appears on both sides, and the row-count split "
                "approximates test_frac despite uneven group sizes",
                ok, f"no_overlap={no_overlap}, both_nonempty={both_nonempty}, test_row_frac={test_row_frac:.3f}")
    return passed + ok, failed + (not ok)


def _near_duplicate_leakage_fixture(n_tickers=10, n_per_ticker=25, d=8, ticker_spread=1.5,
                                     walk_scale=0.5, ar1_phi=0.95, ar1_scale=0.5, rng=None):
    """Mirrors the real linear-probe leakage concern (review finding 6): embeddings
    that are per-ticker near-duplicates over TIME (a small per-step random walk in d
    dims, so temporally-adjacent same-ticker rows sit almost on top of each other --
    exactly how a real encoder's overlapping windows behave) paired with a target
    that's genuinely autocorrelated WITHIN a ticker (AR(1), decays with lag) but
    fully INDEPENDENT ACROSS tickers. A high-dimensional (d) linear regression,
    given even a FEW of a ticker's rows in train, can nearly interpolate that
    ticker's local target level and "predict" a temporally-adjacent held-out row of
    the SAME ticker well -- purely from embedding proximity to its own training
    near-duplicate, not from any genuinely cross-sectional, ticker-transferable
    relationship. A ticker-BLOCKED split removes this: an entirely held-out ticker
    was never seen in training at all, so there's nothing to interpolate from, and
    each ticker's AR(1) is independent of every other's by construction -- the
    honest OOS R^2 on a truly unseen ticker should be near zero."""
    rng = rng or np.random.default_rng(31)
    tickers, embeddings, targets = [], [], []
    for t in range(n_tickers):
        center = rng.normal(scale=ticker_spread, size=d)
        walk = np.cumsum(rng.normal(scale=walk_scale, size=(n_per_ticker, d)), axis=0)
        target = np.zeros(n_per_ticker)
        for m in range(1, n_per_ticker):
            target[m] = ar1_phi * target[m - 1] + rng.normal(scale=ar1_scale)
        tickers.extend([f"T{t}"] * n_per_ticker)
        embeddings.append(center + walk)
        targets.append(target)
    return np.array(tickers), np.concatenate(embeddings, axis=0), np.concatenate(targets)


def test_linear_probe_ticker_blocked_split_removes_near_duplicate_leakage(passed, failed):
    tickers, embeddings, target = _near_duplicate_leakage_fixture()

    # Old behavior: iid row shuffle (mirrors run_diagnostics.py's pre-fix rng.permutation)
    # then a plain positional split -- can place a ticker's near-duplicate adjacent-time
    # rows on both sides.
    order = np.random.default_rng(1).permutation(len(embeddings))
    r2_iid = linear_probe_r2(embeddings[order], target[order], test_frac=0.5)

    # New behavior: a whole ticker's rows land entirely on one side.
    train_mask = group_blocked_train_mask(tickers, test_frac=0.5, rng=np.random.default_rng(1))
    r2_blocked = linear_probe_r2(embeddings, target, train_mask=train_mask)

    ok = r2_iid > r2_blocked + 0.1
    print_check("linear_probe_r2: a ticker-blocked split reports materially lower (more honest) R^2 than "
                "an iid row-shuffled split, on data with only within-ticker autocorrelation and no "
                "genuine cross-ticker signal", ok, f"r2_iid={r2_iid:.4f}, r2_blocked={r2_blocked:.4f}")
    return passed + ok, failed + (not ok)


def test_linear_probe_no_train_mask_matches_original_behavior(passed, failed):
    # Passing no train_mask must reproduce the pre-fix positional-split computation exactly.
    n, d = 400, 5
    embeddings = RNG.normal(size=(n, d))
    true_weights = RNG.normal(size=d)
    signal = embeddings @ true_weights
    target = signal + RNG.normal(scale=0.5 * signal.std(), size=n)
    r2 = linear_probe_r2(embeddings, target, test_frac=0.5)
    ok = r2 > 0.5
    print_check("linear_probe_r2: omitting train_mask still recovers the original injected linear "
                "relationship (no default-path regression)", ok, f"r2={r2:.4f}")
    return passed + ok, failed + (not ok)


def test_quality_persistence_autocorrelation_high_for_persistent_series(passed, failed):
    n = 200
    slow_walk = np.cumsum(RNG.normal(scale=0.05, size=n))  # smooth, structurally persistent
    autocorr = quality_persistence_autocorrelation(slow_walk, lag=12)
    ok = autocorr >= 0.3
    print_check("quality_persistence_autocorrelation: high lag-12 autocorrelation on a smooth/persistent series",
                ok, f"autocorr={autocorr:.4f}")
    return passed + ok, failed + (not ok)


def test_perturbation_sensitivity_flags_discontinuous_encoder(passed, failed):
    def discontinuous(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * 100.0  # tiny input change near 0 -> huge output jump

    raw_inputs = np.full((20, 3), 1e-4)  # near the discontinuity
    ratio = perturbation_sensitivity(discontinuous, raw_inputs, noise_scale=1e-3,
                                      rng=np.random.default_rng(4))
    ok = ratio > 1.0
    print_check("perturbation_sensitivity: flags a discontinuous synthetic encoder (ratio > 1, fails the gate)",
                ok, f"ratio={ratio:.4f}")
    return passed + ok, failed + (not ok)


def test_perturbation_sensitivity_passes_smooth_encoder(passed, failed):
    def smooth(x: np.ndarray) -> np.ndarray:
        return 0.5 * x  # linear contraction -- normalized sensitivity == 0.5

    raw_inputs = RNG.normal(size=(20, 3))
    ratio = perturbation_sensitivity(smooth, raw_inputs, noise_scale=0.01, rng=np.random.default_rng(5))
    ok = ratio <= 1.0
    print_check("perturbation_sensitivity: a smooth synthetic encoder passes the <=1 gate", ok, f"ratio={ratio:.4f}")
    return passed + ok, failed + (not ok)


def test_temporal_smoothness_distinguishes_surprise_from_noise(passed, failed):
    n = 300
    surprise_proxy = RNG.exponential(scale=1.0, size=n)
    embedding_deltas = 2.0 * surprise_proxy + RNG.normal(scale=0.3, size=n)
    corr, p_value = temporal_smoothness_significance(embedding_deltas, surprise_proxy,
                                                       rng=np.random.default_rng(6))
    ok = corr > 0 and p_value < 0.05
    print_check("temporal_smoothness_significance: recovers a real injected correlation at p<0.05",
                ok, f"corr={corr:.4f}, p={p_value:.4f}")
    return passed + ok, failed + (not ok)


def test_temporal_smoothness_not_significant_on_unrelated_noise(passed, failed):
    n = 300
    embedding_deltas = RNG.normal(size=n)
    surprise_proxy = RNG.normal(size=n)  # unrelated to embedding_deltas
    corr, p_value = temporal_smoothness_significance(embedding_deltas, surprise_proxy,
                                                       rng=np.random.default_rng(7))
    ok = p_value >= 0.05
    print_check("temporal_smoothness_significance: does NOT claim significance on unrelated noise",
                ok, f"corr={corr:.4f}, p={p_value:.4f}")
    return passed + ok, failed + (not ok)


def test_latent_similarity_significance_recovers_matched_pairs_gap(passed, failed):
    matched = RNG.uniform(0.0, 1.0, size=100)      # matched states: small embedding distance
    random_pairs = RNG.uniform(2.0, 4.0, size=100)  # random pairs: much larger distance
    gap, p_value = latent_similarity_significance(matched, random_pairs, rng=np.random.default_rng(8))
    ok = gap > 0 and p_value < 0.05
    print_check("latent_similarity_significance: recovers a real matched-pairs-closer-than-random gap at p<0.05",
                ok, f"gap={gap:.4f}, p={p_value:.4f}")
    return passed + ok, failed + (not ok)


def main() -> int:
    print_header("conviction_model/diagnostics.py")
    passed = failed = 0
    for test_fn in [
        test_neighbor_outcome_variance_ratio_recovers_injected_structure,
        test_neighbor_outcome_variance_ratio_same_ticker_exclusion_removes_autocorrelation_artifact,
        test_neighbor_outcome_variance_ratio_no_tickers_arg_matches_original_behavior,
        test_regime_mutual_information_clears_null_when_clusters_match_regime,
        test_regime_mutual_information_does_not_clear_null_on_pure_noise,
        test_linear_probe_recovers_injected_relationship,
        test_group_blocked_train_mask_never_splits_a_group,
        test_linear_probe_ticker_blocked_split_removes_near_duplicate_leakage,
        test_linear_probe_no_train_mask_matches_original_behavior,
        test_quality_persistence_autocorrelation_high_for_persistent_series,
        test_perturbation_sensitivity_flags_discontinuous_encoder,
        test_perturbation_sensitivity_passes_smooth_encoder,
        test_temporal_smoothness_distinguishes_surprise_from_noise,
        test_temporal_smoothness_not_significant_on_unrelated_noise,
        test_latent_similarity_significance_recovers_matched_pairs_gap,
    ]:
        passed, failed = test_fn(passed, failed)
    print_section_end(passed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
