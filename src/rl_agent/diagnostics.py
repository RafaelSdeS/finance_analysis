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


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation via Pearson-on-ranks -- avoids adding scipy
    as a hard project dependency for one statistic (ties are vanishingly
    unlikely in continuous returns, so no rank-averaging needed)."""
    # A constant input has no real ranking (argsort would still hand out 0..n-1
    # arbitrarily) -- check the RAW values, not the post-rank std, which is
    # never zero once ranked.
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    def rank(a):
        order = np.argsort(a)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(a))
        return r
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def ranking_quality(weights: np.ndarray, dates: np.ndarray, tickers: np.ndarray,
                     panel: PricePanel, k_list=(1, 5, 21), cash_frac_threshold: float = 0.5) -> dict:
    """8-D3. Per k (trading days): mean Spearman(agent non-cash weights,
    realized forward k-day return) across backtest days, and the fraction of
    realized top-decile assets present in the agent's top-10 by weight that
    day. Reported over all days AND restricted to days the agent held below
    cash_frac_threshold cash (Phase 7 Finding 5: heavy-cash days otherwise
    drown out the signal from the minority of genuinely active days)."""
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
            w = weights[i]
            non_cash_fwd = np.delete(fwd, CASH_GIDX)
            non_cash_w = np.delete(w, CASH_GIDX)

            rho = _spearman(non_cash_w, non_cash_fwd)
            decile_cut = np.quantile(non_cash_fwd, 0.9)
            top_decile = set(np.where(non_cash_fwd >= decile_cut)[0])
            top10_by_weight = set(np.argsort(non_cash_w)[-10:])
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


if __name__ == "__main__":
    main()
