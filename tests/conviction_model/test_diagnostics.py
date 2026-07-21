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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.conviction_model.diagnostics import (  # noqa: E402
    latent_similarity_significance, linear_probe_r2, neighbor_outcome_variance_ratio,
    perturbation_sensitivity, quality_persistence_autocorrelation, regime_mutual_information,
    temporal_smoothness_significance,
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
        test_regime_mutual_information_clears_null_when_clusters_match_regime,
        test_regime_mutual_information_does_not_clear_null_on_pure_noise,
        test_linear_probe_recovers_injected_relationship,
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
