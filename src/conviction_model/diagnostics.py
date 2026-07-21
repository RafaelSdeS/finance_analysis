"""
diagnostics.py -- Phase 1 (docs/conviction_model/CONVICTION_MODEL_PLAN.md):
the 6 intrinsic embedding-quality diagnostics (+ diagnostic 7, latent
similarity) from "What a 'good' embedding means" -- none need the Phase 2
regressor, that's the point (isolates encoder quality from downstream-model
quality). Diagnostic 8 (embedding value vs. raw/PCA/autoencoder) runs in
Phase 2, not here. Reuses sklearn (KMeans, LinearRegression, mutual_info_score)
per the plan -- already a dependency, nothing new.

Each function returns the raw statistic(s); the quantitative gate (Phase 1's
table) is a separate, trivial comparison left to the caller (the Phase 1
report script, not built yet) so the pass/fail threshold stays visible and
editable in one place rather than buried inside these functions.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mutual_info_score, r2_score
from sklearn.neighbors import NearestNeighbors


def neighbor_outcome_variance_ratio(embeddings: np.ndarray, outcomes: np.ndarray,
                                     k: int = 10, rng: np.random.Generator | None = None) -> float:
    """Diagnostic 1: mean outcome variance among each point's k-nearest
    embedding neighbors, divided by mean outcome variance among k random
    points. <1 means nearby embeddings share more similar outcomes than
    chance; gate (Phase 1 table): <=0.8."""
    rng = rng or np.random.default_rng()
    n = len(embeddings)
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    _, idx = neighbors.kneighbors(embeddings)
    idx = idx[:, 1:]  # drop self-match
    neighbor_var = float(np.mean([np.var(outcomes[row]) for row in idx]))
    random_var = float(np.mean([np.var(outcomes[rng.choice(n, size=k, replace=False)]) for _ in range(n)]))
    return neighbor_var / random_var


def regime_mutual_information(embeddings: np.ndarray, regime_labels: np.ndarray,
                               n_clusters: int = 4, n_permutations: int = 200,
                               rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Diagnostic 2: mutual information between KMeans(embeddings) cluster
    assignment and a known regime indicator, vs. a permutation-null 95th
    percentile (same convention as H-series' permutation-null tests). Gate:
    observed MI > the returned threshold."""
    rng = rng or np.random.default_rng()
    clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(embeddings)
    labels = np.asarray(regime_labels)
    mi = float(mutual_info_score(clusters, labels))
    null = np.array([mutual_info_score(clusters, rng.permutation(labels)) for _ in range(n_permutations)])
    return mi, float(np.percentile(null, 95))


def linear_probe_r2(embeddings: np.ndarray, target: np.ndarray, test_frac: float = 0.5) -> float:
    """Diagnostics 3/4: fit LinearRegression(embeddings -> target) on the
    first (1-test_frac) share, score R^2 out-of-sample on the rest -- an
    out-of-sample split rather than in-sample R^2, since embeddings are
    typically high-dimensional enough that in-sample R^2 would overstate
    what the embedding actually encodes."""
    n = len(embeddings)
    split = int(n * (1 - test_frac))
    model = LinearRegression().fit(embeddings[:split], target[:split])
    pred = model.predict(embeddings[split:])
    return float(r2_score(target[split:], pred))


def valuation_vs_volatility_probe(embeddings: np.ndarray, valuation_target: np.ndarray,
                                   volatility_target: np.ndarray,
                                   test_frac: float = 0.5) -> tuple[float, float]:
    """Diagnostic 3, the direct test of "SSL might learn volatility
    clustering instead of value" (Risks). Gate: valuation R^2 > volatility
    R^2, AND valuation R^2 >= 0.05."""
    val_r2 = linear_probe_r2(embeddings, valuation_target, test_frac)
    vol_r2 = linear_probe_r2(embeddings, volatility_target, test_frac)
    return val_r2, vol_r2


def quality_persistence_autocorrelation(quality_probe_scores: np.ndarray, lag: int = 12) -> float:
    """Diagnostic 4: lag-k autocorrelation of a quality-probe score's own
    time series (should be high -- quality is structurally persistent).
    Gate: >=0.3."""
    s = np.asarray(quality_probe_scores, dtype=np.float64)
    if len(s) <= lag:
        raise ValueError(f"series length {len(s)} must exceed lag {lag}")
    return float(np.corrcoef(s[:-lag], s[lag:])[0, 1])


def perturbation_sensitivity(embed_fn, raw_inputs: np.ndarray, noise_scale: float = 0.01,
                              n_trials: int = 20, rng: np.random.Generator | None = None) -> float:
    """Diagnostic 5: mean ||embed(x+noise)-embed(x)|| / ||noise||, averaged
    over `n_trials` random perturbations and all rows of `raw_inputs`. A
    plain identity/raw-feature mapping has this ratio == 1 exactly (delta
    output == delta input) -- the gate is a parameter-free <=1: the encoder
    shouldn't amplify a small input perturbation more than raw features
    themselves would. `embed_fn`: callable, raw_inputs [N, ...] -> [N, d]."""
    rng = rng or np.random.default_rng()
    base = embed_fn(raw_inputs)
    ratios = []
    for _ in range(n_trials):
        noise = rng.normal(scale=noise_scale, size=raw_inputs.shape)
        perturbed_embed = embed_fn(raw_inputs + noise)
        delta_embed = np.linalg.norm((perturbed_embed - base).reshape(len(raw_inputs), -1), axis=-1)
        delta_input = np.linalg.norm(noise.reshape(len(raw_inputs), -1), axis=-1)
        delta_input = np.where(delta_input == 0, 1e-12, delta_input)
        ratios.append(delta_embed / delta_input)
    return float(np.mean(ratios))


def temporal_smoothness_significance(embedding_deltas: np.ndarray, surprise_proxy: np.ndarray,
                                      n_permutations: int = 200,
                                      rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Diagnostic 6: Pearson correlation between embedding-delta magnitude
    and a "surprise" proxy (e.g. |realized move|, filing-date indicator),
    with a permutation-test p-value. Gate: correlation > 0, p < 0.05."""
    rng = rng or np.random.default_rng()
    d = np.asarray(embedding_deltas, dtype=np.float64)
    s = np.asarray(surprise_proxy, dtype=np.float64)
    corr = float(np.corrcoef(d, s)[0, 1])
    null = np.array([np.corrcoef(d, rng.permutation(s))[0, 1] for _ in range(n_permutations)])
    p_value = float(np.mean(null >= corr))
    return corr, p_value


def latent_similarity_significance(matched_distances: np.ndarray, random_distances: np.ndarray,
                                    n_permutations: int = 200,
                                    rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Diagnostic 7: matched-state (ticker, date) pairs' embedding distance
    should be significantly smaller than random pairs'. Returns
    (mean(random) - mean(matched), permutation p-value) -- gate: gap > 0,
    p < 0.05."""
    rng = rng or np.random.default_rng()
    observed_gap = float(np.mean(random_distances) - np.mean(matched_distances))
    pooled = np.concatenate([matched_distances, random_distances])
    n_matched = len(matched_distances)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(pooled)
        null[i] = perm[n_matched:].mean() - perm[:n_matched].mean()
    p_value = float(np.mean(null >= observed_gap))
    return observed_gap, p_value
