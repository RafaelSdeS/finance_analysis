"""
run_diagnostics.py -- Phase 1 (docs/conviction_model/CONVICTION_MODEL_PLAN.md): the
"Phase 1 report script" diagnostics.py's own docstring refers to but doesn't build --
loads a real trained Stage 1A checkpoint, computes embeddings for every (ticker,
month-end) point in its training universe's own history, and scores them against all 7
intrinsic diagnostics + the plan's quantitative gate table. Diagnostic 8 (embedding vs.
raw/PCA/autoencoder) is Phase 2's job, not here.

One embedding per point: the encoder's 4 branch outputs are mean-pooled
(ssl_pretrain._pool_embedding, reused as-is -- the same pooling CPC's loss already
uses; diagnostics need one vector per point too, not 4 separate branch embeddings).

Sample: every (ticker, month-end) pair in each checkpoint ticker's OWN daily history
(not restricted to top-N membership dates -- matches the encoder's own "sees all
history" convention, plan's Data & universe section). No point-count cap: at ~150
tickers x ~10-15y monthly history that's ~15-20k points, cheap enough for
KNN/KMeans/linear-probe/permutation-test diagnostics to run directly.

Run from project root (does not run automatically -- see CLAUDE.md's "never execute
code you just wrote unless asked"):
    python -m src.conviction_model.run_diagnostics
    python -m src.conviction_model.run_diagnostics --checkpoint-path artifacts/checkpoints/conviction_model/stage1a-20260721-165853.pt
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from ..h_series.spine import monthly_decision_dates
from .config import SSLConfig
from .data import DAILY_FEATURES, MONTHLY_FEATURES, QUARTERLY_FEATURES, WEEKLY_FEATURES, build_frame_cache
from .diagnostics import (
    group_blocked_train_mask, latent_similarity_significance, neighbor_outcome_variance_ratio,
    perturbation_sensitivity, quality_persistence_autocorrelation, regime_mutual_information,
    temporal_smoothness_significance, valuation_vs_volatility_probe,
)
from .encoder import EncoderCNN
from .labels import (
    build_cdi_cumulative_index, compute_risk_adjusted_excess_returns, load_cdi_daily_decimal, load_prices_wide,
)
from .paths import DOCS_DIR, ROOT
from .ssl_pretrain import LazyPanelGatherer, _pool_embedding

CHECKPOINT_DIR = ROOT / "artifacts/checkpoints/conviction_model"
LOG_DIR = ROOT / "artifacts/logs/conviction_model"
SELIC_PATH = ROOT / "data/raw/macro/selic.parquet"

FORWARD_HORIZON = 252         # trading days -- diagnostic 1's outcome (12-month CDI-relative excess return)
PERTURBATION_N_POINTS = 200   # cap: diagnostic 5 re-embeds n_trials times per point, keep it cheap
QUALITY_LAG_MONTHS = 12       # diagnostic 4's autocorrelation lag
MATCHED_PAIR_GAP_DAYS = 365   # diagnostic 7's "far apart in time" threshold
MATCHED_PAIRS_PER_TICKER = 3  # diagnostic 7's cap on matched pairs per ticker
VALUATION_MATCH_TOLERANCE = 0.5  # z-score units counted as "similar" valuation reading

GATES = {
    1: lambda v: v <= 0.8,
    2: lambda mi, thresh: mi > thresh,
    3: lambda val_r2, vol_r2: val_r2 > vol_r2 and val_r2 >= 0.05,
    4: lambda v: v >= 0.3,
    5: lambda v: v <= 1.0,
    6: lambda corr, p: corr > 0 and p < 0.05,
    7: lambda gap, p: gap > 0 and p < 0.05,
}


def setup_logging(run_id: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"diagnostics-{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logfile)],
    )
    return logging.getLogger("diagnostics")


def _latest_checkpoint() -> Path:
    checkpoints = sorted(CHECKPOINT_DIR.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found in {CHECKPOINT_DIR}")
    return checkpoints[-1]


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = SSLConfig(**ckpt["config"])
    model = EncoderCNN(len(DAILY_FEATURES), len(WEEKLY_FEATURES), len(MONTHLY_FEATURES),
                        len(QUARTERLY_FEATURES), d_model=cfg.d_model, n_heads=cfg.n_heads)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt["tickers"]


def build_sample_points(tickers, frame_cache: dict) -> pd.DataFrame:
    """Every (ticker, month-end) pair across each ticker's own daily history --
    sorted per-ticker chronologically so diagnostic 6's consecutive-month deltas and
    diagnostic 7's within-ticker matched pairs can just take adjacent/nearby rows."""
    rows = []
    for t in tickers:
        daily_frame = frame_cache[t][0]
        for d in monthly_decision_dates(daily_frame.index):
            rows.append((t, d))
    points = pd.DataFrame(rows, columns=["ticker", "trade_date"])
    return points.sort_values(["ticker", "trade_date"]).reset_index(drop=True)


def compute_embeddings(model, points: pd.DataFrame, frame_cache: dict, chunk_size: int = 512):
    """Pooled [N, d] embedding per point (positionally aligned with `points`), plus the
    raw batched branch tensors (diagnostic 5's perturbation probe reuses them directly
    instead of re-gathering). One LazyPanelGatherer pass, chunked through the model to
    bound peak activation memory."""
    gatherer = LazyPanelGatherer(points, frame_cache)
    batch = gatherer.gather(np.arange(len(points)))
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(points), chunk_size):
            sl = slice(start, start + chunk_size)
            chunk = {name: t[sl] for name, t in batch.items()}
            out = model(**chunk)
            embeddings.append(_pool_embedding(out).numpy())
    return np.concatenate(embeddings, axis=0), batch


def _anchor_value(frame: pd.DataFrame, as_of: pd.Timestamp, col: str) -> float:
    """Last value of `col` at/before `as_of` -- the same anchor lookup
    data.py::window_tensor uses for its normalization anchor, reused here to pull a
    raw point-in-time feature value (not a windowed tensor) for the probe targets
    diagnostics 3/4/6/7 need."""
    hist = frame.loc[:as_of]
    if hist.empty:
        return np.nan
    return float(hist.iloc[-1][col])


# --- diagnostic 1: nearby embeddings -> similar future outcomes ------------

def diag1_neighbor_outcome(embeddings, points, rng):
    prices_wide = load_prices_wide()
    cdi_index = build_cdi_cumulative_index(load_cdi_daily_decimal(), prices_wide.index)
    decision_dates = pd.DatetimeIndex(sorted(points["trade_date"].unique()))
    universe = points.rename(columns={"trade_date": "decision_date"})[["decision_date", "ticker"]]

    labels = compute_risk_adjusted_excess_returns(prices_wide, cdi_index, decision_dates, universe,
                                                    horizons=(FORWARD_HORIZON,))
    col = f"risk_adj_excess_return_k{FORWARD_HORIZON}"
    value_map = dict(zip(zip(labels["ticker"], labels["decision_date"]), labels[col]))
    outcomes = np.array([value_map.get((t, d), np.nan) for t, d in zip(points["ticker"], points["trade_date"])])

    mask = np.isfinite(outcomes)
    ratio = neighbor_outcome_variance_ratio(
        embeddings[mask], outcomes[mask], k=10, rng=rng,
        tickers=points["ticker"].to_numpy()[mask], dates=points["trade_date"].to_numpy()[mask])
    return ratio, int(mask.sum())


# --- diagnostic 2: market regimes cluster together --------------------------

def diag2_regime_clustering(embeddings, points, rng):
    selic = pd.read_parquet(SELIC_PATH).sort_values("reference_date").set_index("reference_date")["selic"]
    unique_dates = pd.DatetimeIndex(sorted(points["trade_date"].unique()))
    # ffill against a globally date-sorted index first, THEN map back onto points'
    # ticker-sorted order -- ffill-ing directly against points' own (non-date-sorted)
    # order would propagate one ticker's selic value onto an unrelated ticker's row.
    tercile_by_date = pd.qcut(selic.reindex(unique_dates).ffill(), 3, labels=False, duplicates="drop")
    tercile_map = tercile_by_date.to_dict()
    tercile = np.array([tercile_map.get(d, np.nan) for d in points["trade_date"]], dtype=np.float64)

    mask = np.isfinite(tercile)
    mi, threshold = regime_mutual_information(embeddings[mask], tercile[mask].astype(int), rng=rng)
    return mi, threshold, int(mask.sum())


# --- diagnostic 3: valuation regimes emerge naturally ------------------------

def diag3_valuation_vs_volatility(embeddings, points, frame_cache, rng):
    val, vol = [], []
    for t, d in zip(points["ticker"], points["trade_date"]):
        daily_frame, _, _, quarterly_frame = frame_cache[t]
        val.append(_anchor_value(quarterly_frame, d, "pl_zhist_5y"))
        vol.append(_anchor_value(daily_frame, d, "volatility_ratio_20_60"))
    val, vol = np.array(val), np.array(vol)

    mask = np.isfinite(val) & np.isfinite(vol)
    # Ticker-BLOCKED split (not an iid row shuffle): a whole ticker's rows land entirely on
    # one side, so adjacent-month near-duplicate rows of the SAME ticker can't leak across
    # train/test and inflate OOS R^2 (diagnostics review finding 6).
    train_mask = group_blocked_train_mask(points["ticker"].to_numpy()[mask], test_frac=0.5, rng=rng)
    val_r2, vol_r2 = valuation_vs_volatility_probe(embeddings[mask], val[mask], vol[mask], train_mask=train_mask)
    return val_r2, vol_r2, int(mask.sum())


# --- diagnostic 4: company quality represented consistently over time -------

def diag4_quality_persistence(embeddings, points, frame_cache, rng):
    quality = np.array([_anchor_value(frame_cache[t][3], d, "roe_zhist_5y")
                         for t, d in zip(points["ticker"], points["trade_date"])])
    mask = np.isfinite(quality)
    ticker_arr = points["ticker"].to_numpy()

    # Ticker-BLOCKED OOS split: the quality probe is fit on TRAIN-ticker rows only and
    # scored (persistence autocorrelation) only on HELD-OUT test-ticker rows -- a company
    # never seen while fitting the probe. The original in-sample version (fit on
    # everything, predict on everything, with an inert rng.permutation that never actually
    # created a split -- row order doesn't affect what LinearRegression.fit learns) let an
    # overfit probe potentially memorize per-company idiosyncrasies rather than a genuinely
    # transferable quality signal.
    train_mask = group_blocked_train_mask(ticker_arr[mask], test_frac=0.5, rng=rng)
    masked_idx = np.flatnonzero(mask)  # original point positions where quality is defined
    train_idx, test_idx = masked_idx[train_mask], masked_idx[~train_mask]

    probe = LinearRegression().fit(embeddings[train_idx], quality[train_idx])
    predicted_test = probe.predict(embeddings[test_idx])  # aligned with test_idx, same order

    autocorrs = []
    test_tickers = ticker_arr[test_idx]
    for t in np.unique(test_tickers):
        # points is ticker/date-sorted and masked_idx/test_idx preserve that relative
        # order (boolean indexing never reorders), so a ticker's positions WITHIN
        # test_idx stay chronological -- lag=12 becomes "12 valid observations", a small
        # approximation of a strict calendar lag when a ticker has gaps.
        pos = np.flatnonzero(test_tickers == t)
        if len(pos) <= QUALITY_LAG_MONTHS + 1:
            continue
        autocorrs.append(quality_persistence_autocorrelation(predicted_test[pos], lag=QUALITY_LAG_MONTHS))
    return (float(np.mean(autocorrs)) if autocorrs else float("nan")), len(autocorrs)


# --- diagnostic 5: stability under small perturbations -----------------------

def diag5_perturbation(model, batch, rng):
    n = min(PERTURBATION_N_POINTS, batch["daily"].shape[0])
    idx = rng.choice(batch["daily"].shape[0], size=n, replace=False)
    daily = batch["daily"][idx].numpy()
    fixed = {name: batch[name][idx] for name in ("weekly", "monthly", "fundamentals")}

    def embed_fn(daily_arr):
        with torch.no_grad():
            out = model(torch.tensor(daily_arr, dtype=torch.float32),
                        fixed["weekly"], fixed["monthly"], fixed["fundamentals"])
        return _pool_embedding(out).numpy()

    return perturbation_sensitivity(embed_fn, daily, rng=rng), n


# --- diagnostic 6: smooth evolution absent new information -------------------

def diag6_temporal_smoothness(embeddings, points, frame_cache, rng):
    ticker_arr = points["ticker"].to_numpy()
    date_arr = points["trade_date"].to_numpy()
    deltas, surprises = [], []
    for t in points["ticker"].unique():
        idx = np.flatnonzero(ticker_arr == t)
        if len(idx) < 2:
            continue
        emb = embeddings[idx]
        daily_frame = frame_cache[t][0]
        deltas.append(np.linalg.norm(emb[1:] - emb[:-1], axis=1))
        surprises.append(np.array([abs(_anchor_value(daily_frame, d, "return_1m")) for d in date_arr[idx][1:]]))
    deltas, surprises = np.concatenate(deltas), np.concatenate(surprises)

    mask = np.isfinite(surprises)
    corr, p = temporal_smoothness_significance(deltas[mask], surprises[mask], rng=rng)
    return corr, p, int(mask.sum())


# --- diagnostic 7: latent-similarity check ------------------------------------

def diag7_latent_similarity(embeddings, points, frame_cache, rng):
    valuation = np.array([_anchor_value(frame_cache[t][3], d, "pl_zhist_5y")
                           for t, d in zip(points["ticker"], points["trade_date"])])
    ticker_arr = points["ticker"].to_numpy()
    date_arr = points["trade_date"].to_numpy()

    matched = []
    for t in points["ticker"].unique():
        idx = np.flatnonzero((ticker_arr == t) & np.isfinite(valuation))
        dates, vals = date_arr[idx], valuation[idx]
        found = 0
        # ponytail: O(n^2) per-ticker scan (n ~ months of history, capped by the
        # per-ticker MATCHED_PAIRS_PER_TICKER early break) -- fine at ~130 points/ticker;
        # revisit with a sorted-by-valuation search if the universe grows a lot.
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                if (dates[j] - dates[i]) / np.timedelta64(1, "D") < MATCHED_PAIR_GAP_DAYS:
                    continue
                if abs(vals[i] - vals[j]) > VALUATION_MATCH_TOLERANCE:
                    continue
                matched.append((idx[i], idx[j]))
                found += 1
                if found >= MATCHED_PAIRS_PER_TICKER:
                    break
            if found >= MATCHED_PAIRS_PER_TICKER:
                break

    if not matched:
        return float("nan"), float("nan"), 0
    matched = np.array(matched)
    matched_dist = np.linalg.norm(embeddings[matched[:, 0]] - embeddings[matched[:, 1]], axis=1)

    n_random = len(matched)
    a = rng.integers(0, len(embeddings), size=n_random)
    b = rng.integers(0, len(embeddings), size=n_random)
    random_dist = np.linalg.norm(embeddings[a] - embeddings[b], axis=1)

    gap, p = latent_similarity_significance(matched_dist, random_dist, rng=rng)
    return gap, p, len(matched)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = setup_logging(run_id)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else _latest_checkpoint()
    rng = np.random.default_rng(args.seed)

    log.info(f"Checkpoint: {checkpoint_path}")
    model, cfg, tickers = load_checkpoint(checkpoint_path)
    log.info(f"Universe: {len(tickers)} tickers")

    frame_cache = build_frame_cache(tickers)
    points = build_sample_points(tickers, frame_cache)
    log.info(f"Sample points: {len(points)} (ticker, month-end) pairs")

    embeddings, batch = compute_embeddings(model, points, frame_cache)
    log.info(f"Embeddings computed: {embeddings.shape}")

    results = {}

    ratio, n1 = diag1_neighbor_outcome(embeddings, points, rng)
    results["1_neighbor_outcome_variance_ratio"] = {
        "value": ratio, "n": n1, "gate": "<=0.8", "pass": bool(GATES[1](ratio))}
    log.info(f"[1] neighbor-outcome variance ratio = {ratio:.4f} (n={n1}) -- "
             f"{'PASS' if GATES[1](ratio) else 'FAIL'}")

    mi, thresh, n2 = diag2_regime_clustering(embeddings, points, rng)
    results["2_regime_mutual_information"] = {
        "mi": mi, "null_p95": thresh, "n": n2, "gate": "mi > null_p95", "pass": bool(GATES[2](mi, thresh))}
    log.info(f"[2] regime MI = {mi:.4f} vs null_p95={thresh:.4f} (n={n2}) -- "
             f"{'PASS' if GATES[2](mi, thresh) else 'FAIL'}")

    val_r2, vol_r2, n3 = diag3_valuation_vs_volatility(embeddings, points, frame_cache, rng)
    results["3_valuation_vs_volatility_probe"] = {
        "valuation_r2": val_r2, "volatility_r2": vol_r2, "n": n3,
        "gate": "val_r2 > vol_r2 and val_r2 >= 0.05", "pass": bool(GATES[3](val_r2, vol_r2))}
    log.info(f"[3] valuation R2 = {val_r2:.4f} vs volatility R2 = {vol_r2:.4f} (n={n3}) -- "
             f"{'PASS' if GATES[3](val_r2, vol_r2) else 'FAIL'}")

    persist, n4 = diag4_quality_persistence(embeddings, points, frame_cache, rng)
    gate4_pass = np.isfinite(persist) and GATES[4](persist)
    results["4_quality_persistence_autocorrelation"] = {
        "value": persist, "n_tickers": n4, "gate": ">=0.3", "pass": bool(gate4_pass)}
    log.info(f"[4] quality persistence autocorr = {persist:.4f} (n_tickers={n4}) -- "
             f"{'PASS' if gate4_pass else 'FAIL'}")

    sensitivity, n5 = diag5_perturbation(model, batch, rng)
    results["5_perturbation_sensitivity"] = {
        "value": sensitivity, "n": n5, "gate": "<=1.0", "pass": bool(GATES[5](sensitivity))}
    log.info(f"[5] perturbation sensitivity = {sensitivity:.4f} (n={n5}) -- "
             f"{'PASS' if GATES[5](sensitivity) else 'FAIL'}")

    corr, p6, n6 = diag6_temporal_smoothness(embeddings, points, frame_cache, rng)
    results["6_temporal_smoothness"] = {
        "correlation": corr, "p_value": p6, "n": n6,
        "gate": "corr > 0 and p < 0.05", "pass": bool(GATES[6](corr, p6))}
    log.info(f"[6] temporal smoothness corr = {corr:.4f}, p = {p6:.4f} (n={n6}) -- "
             f"{'PASS' if GATES[6](corr, p6) else 'FAIL'}")

    gap7, p7, n7 = diag7_latent_similarity(embeddings, points, frame_cache, rng)
    gate7_pass = np.isfinite(gap7) and GATES[7](gap7, p7)
    results["7_latent_similarity"] = {
        "gap": gap7, "p_value": p7, "n_pairs": n7,
        "gate": "gap > 0 and p < 0.05", "pass": bool(gate7_pass)}
    log.info(f"[7] latent similarity gap = {gap7:.4f}, p = {p7:.4f} (n_pairs={n7}) -- "
             f"{'PASS' if gate7_pass else 'FAIL'}")

    n_pass = sum(1 for r in results.values() if r["pass"])
    log.info(f"Diagnostics 1-7: {n_pass}/7 gates passed.")

    out_path = DOCS_DIR / f"PHASE1_DIAGNOSTICS_{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"checkpoint": str(checkpoint_path), "tickers": tickers, "n_points": len(points), "results": results},
        indent=2))
    log.info(f"Report saved: {out_path}")
    log.info(f"Log file: {LOG_DIR / f'diagnostics-{run_id}.log'}")


if __name__ == "__main__":
    main()
