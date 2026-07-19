"""
diagnostics.py — Phase 8 offline diagnostics (EIIE_DIAGNOSIS_PLAN.md "Phase
8: Evidence-first sequencing"), run AFTER training, no gradient steps:

8-D2 cross-seed consistency: do independently-seeded runs on the same
config converge to a similar allocation (a stable learned signal), or do
they diverge (seeds exploiting noise)?

8-D3 ranking quality — the master key: for each rebalance day, does the
agent's weight vector rank the realized future winners highly? If not,
that's a representation problem (features/encoder). If winners rank highly
but the policy still underallocates them, that's a policy/entropy problem
(the online phase, per Phase 7's diagnosis).

Both read the per-day full weight matrix (T, n_global) experiment.py saves
to each run's weights.npz. Runs from before that change have no on-disk
weight matrix (report.html keeps only a lossy cash+top-9+other
aggregation) -- pass --replay to reconstruct it via an inference-only
backtest against model_pretrain.pt (or model.pt if that's all the run has).

Usage:
    python -m src.rl_agent.diagnostics --runs experiments/eiie_features_frozen_*
    python -m src.rl_agent.diagnostics --runs experiments/eiie_features_2026* --replay
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .data import CASH_GIDX, PricePanel, load_price_panel
from .environment import run_backtest
from .networks import EIIECNN
from .pvm import PortfolioVectorMemory
from .train import agent_forward, load_checkpoint


def _load_run_weights(run_dir: Path, replay: bool, device: str = "cpu") -> dict:
    """Returns {"weights": (T,n_global) f32, "dates": datetime64[D] array,
    "tickers": ["cash", ...] array, "cfg": ExperimentConfig, "_panel":
    PricePanel (only set when --replay built one, so callers can reuse it)}."""
    cfg = ExperimentConfig.from_json(run_dir / "config.json")
    npz_path = run_dir / "weights.npz"
    if npz_path.exists() and not replay:
        data = np.load(npz_path)
        return {"weights": data["weights"], "dates": data["dates"],
                "tickers": data["tickers"], "cfg": cfg}

    if not replay:
        raise FileNotFoundError(
            f"{npz_path} not found (run predates Phase 8's weights.npz artifact) -- pass --replay")

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    model_path = run_dir / "model_pretrain.pt"
    if not model_path.exists():
        model_path = run_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"neither model_pretrain.pt nor model.pt found in {run_dir}")

    panel = load_price_panel(cfg.data, n_slots=cfg.model.n_assets)
    split = manifest["split"]
    if manifest["eval_split"] == "test":
        start_idx, end_idx = split["val_end_idx"] + 1, panel.end_idx
    else:
        start_idx, end_idx = split["train_end_idx"] + 1, split["val_end_idx"]

    model = EIIECNN(cfg.data.window, cfg.model.conv1_out_channels,
                     cfg.model.conv2_out_channels, len(cfg.data.features)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.l2)
    pvm = PortfolioVectorMemory(len(panel.dates), panel.n_global, slot_gidx=panel.slot_gidx,
                                 valid=panel.valid, device=device)
    load_checkpoint(str(model_path), model, optimizer, pvm, map_location=device)
    model.eval()

    def weight_fn(t, w_prev_np, w_drift_np, panel):
        return agent_forward(model, pvm, panel, t, cfg.data.features, device)

    result = run_backtest(panel, weight_fn, cfg.costs.c_sell, cfg.costs.c_buy,
                           start_idx, end_idx, cfg.costs.backtest_mu_tol)
    return {
        "weights": result.weights.astype(np.float32),
        "dates": result.dates.values.astype("datetime64[D]"),
        "tickers": np.array(["cash"] + list(panel.asset_index.tickers)),
        "cfg": cfg, "_panel": panel,
    }


def cross_seed_consistency(runs: list) -> dict:
    """8-D2. `runs`: >=2 loaded run dicts (same config, different seeds
    expected). Aligns on the intersection of dates present in every run,
    then reports mean pairwise (a) cosine similarity of daily weight
    vectors, (b) top-10-by-weight Jaccard overlap, (c) correlation of each
    run's mean-weight vector (over the common window) with every other's."""
    if len(runs) < 2:
        raise ValueError("cross_seed_consistency needs at least 2 runs")

    date_sets = [set(pd.DatetimeIndex(r["dates"])) for r in runs]
    common = pd.DatetimeIndex(sorted(set.intersection(*date_sets)))
    if len(common) == 0:
        raise ValueError("runs share no common dates -- not comparable")

    aligned = []
    for r in runs:
        idx = pd.DatetimeIndex(r["dates"])
        pos = idx.get_indexer(common)
        assert (pos >= 0).all(), "a run is missing a date in the common intersection"
        aligned.append(r["weights"][pos])

    n_runs = len(aligned)
    cos_sims, jaccards = [], []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            a, b = aligned[i], aligned[j]
            num = (a * b).sum(axis=1)
            denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            cos = np.where(denom > 0, num / np.where(denom > 0, denom, 1.0), 1.0)
            cos_sims.append(float(cos.mean()))

            top_a = np.argsort(a, axis=1)[:, -10:]
            top_b = np.argsort(b, axis=1)[:, -10:]
            jac = [len(set(top_a[t]) & set(top_b[t])) / len(set(top_a[t]) | set(top_b[t]))
                   for t in range(a.shape[0])]
            jaccards.append(float(np.mean(jac)))

    mean_weights = np.stack([a.mean(axis=0) for a in aligned])
    corr_matrix = np.corrcoef(mean_weights)
    off_diag = corr_matrix[np.triu_indices(n_runs, k=1)]

    return {
        "n_runs": n_runs, "n_common_days": len(common),
        "mean_pairwise_cosine": float(np.mean(cos_sims)),
        "mean_pairwise_top10_jaccard": float(np.mean(jaccards)),
        "mean_pairwise_meanweight_corr": float(np.mean(off_diag)),
    }


def forward_return(panel: PricePanel, t: int, k: int) -> np.ndarray:
    """Realized forward k-day return per asset in global space, t -> t+k.
    Cash uses the compounded CDI factor over (t, t+k]; there is no lookahead
    concern here since this runs strictly after training, purely as a
    diagnostic. NaN (whole vector) if t+k runs past the panel's calendar."""
    n = panel.n_global
    if t + k >= len(panel.dates):
        return np.full(n, np.nan)
    fwd = panel.close[t + k, :n] / panel.close[t, :n] - 1.0
    fwd[CASH_GIDX] = np.prod(panel.cdi_factor[t + 1: t + k + 1]) - 1.0
    return fwd


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks with tie handling (mirrors scipy.stats.rankdata's
    'average' method). Needed because exact ties are common in this data:
    any never-active global column (delisted/pre-IPO/out-of-membership,
    forward-filled flat) produces a whole block of identical values on both
    axes -- plain argsort ranks those arbitrarily, which understates or
    overstates rho depending on how the arbitrary tie-break happens to
    align (EIIE_IMPROVEMENT_PLAN.md Finding 1.2)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    n = len(a)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation via Pearson-on-average-ranks -- avoids
    adding scipy as a hard project dependency for one statistic."""
    # A constant input has no real ranking -- check the RAW values, not the
    # post-rank std, which is never zero once ranked.
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])


def _active_stock_gidx(panel: PricePanel, t: int) -> np.ndarray:
    """Global indices of the assets actually holdable (in the dynamic
    top-N universe) on day t -- excludes cash (never a slot) and any
    padding/invalid slot. Ranking/selection metrics must be scoped to this
    set, not the full ~171-wide global space: a never-active name sits at a
    permanent weight of 0 and a forward return of 0 (flat bfill), diluting
    any real signal on the ~50 names the agent could actually choose among
    (EIIE_IMPROVEMENT_PLAN.md Finding 1.1)."""
    return panel.slot_gidx[t][panel.valid[t]]


def ranking_quality(weights: np.ndarray, dates: np.ndarray, tickers: np.ndarray,
                     panel: PricePanel, k_list=(1, 5, 21), cash_frac_threshold: float = 0.5) -> dict:
    """8-D3 (fixed 2026-07-18, EIIE_IMPROVEMENT_PLAN.md Finding 1). Per k
    (trading days): mean Spearman(agent weight, realized forward k-day
    return) across backtest days, and the fraction of realized top-decile
    assets present in the agent's top-10 by weight that day -- both scoped
    to THAT DAY'S ACTIVE ~50-NAME UNIVERSE ONLY (see _active_stock_gidx),
    with tie-aware average-rank Spearman (see _rankdata). Reported over all
    days AND restricted to days the agent held below cash_frac_threshold
    cash (Phase 7 Finding 5: heavy-cash days otherwise drown out the signal
    from the minority of genuinely active days)."""
    assert list(tickers[1:]) == list(panel.asset_index.tickers), (
        "weights.npz tickers don't match this panel's asset index -- "
        "was the run built against a different membership snapshot?"
    )
    t_idx = panel.dates.get_indexer(pd.DatetimeIndex(dates))
    assert (t_idx >= 0).all(), "a weights.npz date isn't on this panel's calendar"

    out = {}
    for k in k_list:
        rows = []  # (rho, hit, is_active)
        for i, t in enumerate(t_idx):
            fwd = forward_return(panel, int(t), k)
            if np.any(np.isnan(fwd)):
                continue
            active_g = _active_stock_gidx(panel, int(t))
            if len(active_g) == 0:
                continue
            w = weights[i]
            w_active = w[active_g]
            fwd_active = fwd[active_g]

            rho = _spearman(w_active, fwd_active)
            decile_cut = np.quantile(fwd_active, 0.9)
            top_decile = set(np.where(fwd_active >= decile_cut)[0])
            top10_by_weight = set(np.argsort(w_active)[-10:])
            hit = len(top_decile & top10_by_weight) / max(len(top_decile), 1)
            rows.append((rho, hit, w[CASH_GIDX] < cash_frac_threshold))

        all_rho = [r for r, _, _ in rows]
        all_hit = [h for _, h, _ in rows]
        act_rho = [r for r, _, a in rows if a]
        act_hit = [h for _, h, a in rows if a]
        out[k] = {
            "n_days": len(rows),
            "mean_spearman": float(np.nanmean(all_rho)) if all_rho else float("nan"),
            "mean_top10_hit_rate": float(np.mean(all_hit)) if all_hit else float("nan"),
            "n_active_days": len(act_rho),
            "mean_spearman_active_days": float(np.nanmean(act_rho)) if act_rho else float("nan"),
            "mean_top10_hit_rate_active_days": float(np.mean(act_hit)) if act_hit else float("nan"),
        }
    return out


def selection_alpha(weights: np.ndarray, dates: np.ndarray, tickers: np.ndarray,
                     panel: PricePanel, k_list=(1, 5, 21), top_k_list=(1, 5),
                     n_perm: int = 1000, seed: int = 0) -> dict:
    """Forward LOG-return of the agent's top-`top_k` held names (by weight)
    minus the equal-weight mean of that day's active universe, at each
    horizon k -- the direct "is the pick better than random" measure.
    Unlike ranking_quality's full-cross-section Spearman, this has full
    statistical power even when the policy is near one-hot (observed
    effective_n ~ 1-2): the log-return training objective's gradient on
    each asset scales with that asset's own current weight (see
    EIIE_IMPROVEMENT_PLAN.md Finding 2), so it only ever disciplines the
    top of the distribution -- this metric measures exactly that.

    Permutation null per (k, top_k): for each day, redraw a random top_k
    subset of that day's active universe (same size, ignoring the agent's
    actual weights) n_perm times; the null distribution of the
    day-averaged excess log-return gives a 97.5th-percentile threshold and
    a one-sided p-value for the observed mean."""
    assert list(tickers[1:]) == list(panel.asset_index.tickers), (
        "weights.npz tickers don't match this panel's asset index -- "
        "was the run built against a different membership snapshot?"
    )
    t_idx = panel.dates.get_indexer(pd.DatetimeIndex(dates))
    assert (t_idx >= 0).all(), "a weights.npz date isn't on this panel's calendar"
    rng = np.random.default_rng(seed)

    out = {}
    for k in k_list:
        obs_alpha = {tk: [] for tk in top_k_list}
        perm_sum = {tk: np.zeros(n_perm) for tk in top_k_list}
        n_days_by_topk = {tk: 0 for tk in top_k_list}

        for i, t in enumerate(t_idx):
            t = int(t)
            if t + k >= len(panel.dates):
                continue
            active_g = _active_stock_gidx(panel, t)
            n_active = len(active_g)
            if n_active == 0:
                continue
            log_ret = np.log(panel.close[t + k, active_g]) - np.log(panel.close[t, active_g])
            universe_mean = log_ret.mean()
            w_active = weights[i][active_g]

            for tk in top_k_list:
                if n_active < tk:
                    continue
                sel = np.argsort(w_active)[-tk:]
                obs_alpha[tk].append(float(log_ret[sel].mean() - universe_mean))
                n_days_by_topk[tk] += 1

                rand_keys = rng.random((n_perm, n_active))
                perm_sel = np.argsort(rand_keys, axis=1)[:, -tk:]
                perm_sum[tk] += log_ret[perm_sel].mean(axis=1) - universe_mean

        k_out = {}
        for tk in top_k_list:
            n_days = n_days_by_topk[tk]
            if n_days == 0:
                k_out[f"top{tk}"] = {"n_days": 0, "mean_selection_alpha": float("nan"),
                                      "perm_null_975pct": float("nan"), "p_value": float("nan")}
                continue
            observed = float(np.mean(obs_alpha[tk]))
            perm_means = perm_sum[tk] / n_days
            threshold = float(np.percentile(perm_means, 97.5))
            p_value = float((np.sum(perm_means >= observed) + 1) / (n_perm + 1))
            k_out[f"top{tk}"] = {"n_days": n_days, "mean_selection_alpha": observed,
                                  "perm_null_975pct": threshold, "p_value": p_value}
        out[k] = k_out
    return out


def spearman_permutation_null(weights: np.ndarray, dates: np.ndarray, tickers: np.ndarray,
                               panel: PricePanel, k_list=(1, 5, 21), n_perm: int = 1000,
                               seed: int = 0) -> dict:
    """Null distribution of mean Spearman under 'the agent's day-t decision
    carries no information about day-t's realized forward return': shuffles
    the pairing between weight-days and forward-return-days (each day's own
    active-only, tie-aware rank vectors stay fixed; only which day's
    weight-ranks get compared against which day's return-ranks is
    permuted). Replaces the hand-set +-0.02 bar (EIIE_IMPROVEMENT_PLAN.md
    Finding 1.3, circular: it was calibrated from this same diluted
    metric) with a 97.5th-percentile threshold plus a one-sided p-value for
    the actually observed mean Spearman, per run.

    Requires a constant active-universe size across all used days (true by
    the tested invariant -- exactly 50 active members every in-window day);
    raises if that invariant doesn't hold, rather than silently degrading."""
    assert list(tickers[1:]) == list(panel.asset_index.tickers), (
        "weights.npz tickers don't match this panel's asset index -- "
        "was the run built against a different membership snapshot?"
    )
    t_idx = panel.dates.get_indexer(pd.DatetimeIndex(dates))
    assert (t_idx >= 0).all(), "a weights.npz date isn't on this panel's calendar"

    out = {}
    for k in k_list:
        z_w_rows, z_fwd_rows = [], []
        for i, t in enumerate(t_idx):
            t = int(t)
            fwd = forward_return(panel, t, k)
            if np.any(np.isnan(fwd)):
                continue
            active_g = _active_stock_gidx(panel, t)
            if len(active_g) == 0:
                continue
            w_active = weights[i][active_g]
            fwd_active = fwd[active_g]
            if np.std(w_active) == 0 or np.std(fwd_active) == 0:
                continue
            rw, rf = _rankdata(w_active), _rankdata(fwd_active)
            z_w_rows.append((rw - rw.mean()) / rw.std())
            z_fwd_rows.append((rf - rf.mean()) / rf.std())

        n_days = len(z_w_rows)
        if n_days == 0:
            out[k] = {"n_days": 0, "observed_mean_spearman": float("nan"),
                       "null_975pct": float("nan"), "p_value": float("nan")}
            continue
        lengths = {len(r) for r in z_w_rows}
        if len(lengths) != 1:
            raise ValueError(
                f"spearman_permutation_null requires a constant active-universe size "
                f"across days (k={k}); got sizes {lengths} -- the exactly-50-active-"
                f"members invariant doesn't hold for this run/window.")
        n_active = lengths.pop()
        Zw, Zf = np.stack(z_w_rows), np.stack(z_fwd_rows)
        M = (Zw @ Zf.T) / n_active  # M[i, j] = corr(day i weight-rank, day j return-rank)
        observed = float(np.mean(np.diag(M)))

        rng_k = np.random.default_rng(seed + k)
        idx = np.arange(n_days)
        null_means = np.empty(n_perm)
        for p in range(n_perm):
            perm = rng_k.permutation(n_days)
            null_means[p] = M[idx, perm].mean()

        out[k] = {
            "n_days": n_days,
            "observed_mean_spearman": observed,
            "null_975pct": float(np.percentile(null_means, 97.5)),
            "p_value": float((np.sum(null_means >= observed) + 1) / (n_perm + 1)),
        }
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8 offline diagnostics: cross-seed consistency (8-D2) + ranking quality (8-D3)")
    parser.add_argument("--runs", nargs="+", required=True,
                         help="run directories or glob patterns, e.g. experiments/eiie_features_frozen_*")
    parser.add_argument("--replay", action="store_true",
                         help="reconstruct weights via an inference-only backtest for runs with no weights.npz")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 21],
                         help="forward-return horizons in trading days for 8-D3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-perm", type=int, default=1000,
                         help="permutation draws for selection_alpha / the recalibrated Spearman null")
    parser.add_argument("--no-perm", action="store_true",
                         help="skip selection_alpha and the permutation null (faster, ranking_quality only)")
    args = parser.parse_args()

    matched = {Path(p) for pattern in args.runs for p in glob.glob(pattern)}
    matched |= {Path(p) for p in args.runs if Path(p).exists()}
    run_dirs = sorted(matched)
    if not run_dirs:
        raise SystemExit(f"no run directories matched {args.runs!r}")

    print(f"Loading {len(run_dirs)} run(s)...")
    loaded = []
    for d in run_dirs:
        try:
            loaded.append((d, _load_run_weights(d, args.replay, args.device)))
        except FileNotFoundError as e:
            print(f"  SKIP {d.name}: {e}")

    if len(loaded) >= 2:
        print("\n=== 8-D2: cross-seed consistency ===")
        for key, val in cross_seed_consistency([r for _, r in loaded]).items():
            print(f"  {key}: {val}")
    else:
        print("\n(8-D2 skipped: need >=2 loaded runs)")

    print("\n=== 8-D3: ranking quality ===")
    panel_cache = {}
    for d, r in loaded:
        cfg = r["cfg"]
        cache_key = (cfg.data.window_start, cfg.data.window_end, tuple(cfg.data.features))
        if cache_key not in panel_cache:
            panel_cache[cache_key] = r.get("_panel") or load_price_panel(cfg.data, n_slots=cfg.model.n_assets)
        panel = panel_cache[cache_key]
        quality = ranking_quality(r["weights"], r["dates"], r["tickers"], panel, tuple(args.k))
        print(f"  {d.name}:")
        for k, stats in quality.items():
            print(f"    k={k}: {stats}")

        if args.no_perm:
            continue

        print("    -- recalibrated Spearman null (replaces hand-set +-0.02 bar) --")
        null_stats = spearman_permutation_null(r["weights"], r["dates"], r["tickers"], panel,
                                                tuple(args.k), n_perm=args.n_perm)
        for k, stats in null_stats.items():
            print(f"    k={k}: {stats}")

        print("    -- selection alpha (full power at any concentration) --")
        alpha_stats = selection_alpha(r["weights"], r["dates"], r["tickers"], panel,
                                       tuple(args.k), n_perm=args.n_perm)
        for k, tk_stats in alpha_stats.items():
            print(f"    k={k}: {tk_stats}")


if __name__ == "__main__":
    main()
