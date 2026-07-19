"""
supervised_experiment.py — M3 supervised ranking probe experiment.

Trains a ranker on each horizon k∈{1,5,21}, computes daily IC on train & val,
and reports permutation nulls + p-values. No RL, no portfolio mechanics —
directly tests "is there extractable cross-sectional signal in these features?"
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

from .config import ExperimentConfig, DataConfig
from .data import GlobalAssetIndex, PricePanel, load_price_panel
from .supervised_probe import SupervisedRankingProbe, listwise_ranking_loss, compute_daily_ic


def compute_forward_returns(panel: PricePanel, k: int) -> np.ndarray:
    """Compute k-day forward log-returns for each asset (global space).

    Returns: [T, n_global] array, where returns[t, i] is the log-return of
    asset i from day t to day t+k (inclusive), or NaN if the window extends
    past the end of data or the asset has no price on day t or t+k.
    """
    T = panel.close.shape[0]
    n_global = panel.n_global
    fwd_returns = np.full((T, n_global), np.nan)

    for t in range(T - k):
        # Log-return from close[t] to close[t+k], global space
        c_t = panel.close[t, :n_global]
        c_tk = panel.close[t + k, :n_global]
        # ponytail: vectorized element-wise; NaN * anything = NaN as desired
        fwd_returns[t] = np.log(c_tk / c_t)

    return fwd_returns


def create_train_val_loaders(
    panel: PricePanel,
    data_config: DataConfig,
    fwd_returns: dict,  # {k: [T, n_global] array}
    train_end_idx: int,
    val_end_idx: int,
    batch_size: int = 50,
):
    """Create PyTorch data loaders for train and val splits.

    Args:
        panel: the price panel
        data_config: data configuration with features list
        fwd_returns: dict mapping horizon k to forward return array [T, n_global]
        train_end_idx: last (inclusive) index in train split
        val_end_idx: last (inclusive) index in val split
        batch_size: batch size for loader

    Yields:
        (split_name, loader) tuples for "train" and "val"
    """
    # ponytail: use torch DataLoader for batching; create custom Dataset
    class RankingDataset(torch.utils.data.Dataset):
        def __init__(self, panel, data_config, fwd_returns, start_idx, end_idx, k_list):
            self.panel = panel
            self.features = tuple(data_config.features)
            self.fwd_returns = fwd_returns
            self.start_idx = start_idx
            self.end_idx = end_idx
            self.k_list = k_list

        def __len__(self):
            return self.end_idx - self.start_idx + 1

        def __getitem__(self, idx):
            t = self.start_idx + idx
            # X: [n_features, m, window] as numpy
            X_np = self.panel.window_tensor(t, features=self.features)
            X = torch.from_numpy(X_np).float()
            mask = torch.from_numpy(self.panel.valid[t]).bool()
            # Returns for each horizon: {k: [n_global]} as torch
            returns_t = {k: torch.from_numpy(self.fwd_returns[k][t]).float() for k in self.k_list}
            return X, mask, returns_t, t

    k_list = sorted(fwd_returns.keys())

    train_dataset = RankingDataset(panel, data_config, fwd_returns, 0, train_end_idx, k_list)
    val_dataset = RankingDataset(panel, data_config, fwd_returns, train_end_idx + 1, val_end_idx, k_list)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    yield "train", train_loader
    yield "val", val_loader


def train_probe(
    probe: SupervisedRankingProbe,
    train_loader: torch.utils.data.DataLoader,
    k: int,
    epochs: int = 10,
    device: str = "cuda",
):
    """Train the ranking probe on a single horizon k."""
    probe.to(device)
    optimizer = optim.Adam(probe.parameters(), lr=1e-4)

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            X, mask, returns_t, _ = batch
            X = X.to(device)
            mask = mask.to(device)
            fwd_returns_k = returns_t[k].to(device)

            scores = probe(X, mask)
            loss = listwise_ranking_loss(scores, fwd_returns_k, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, loss={avg_loss:.6f}")

    return probe


def evaluate_probe(
    probe: SupervisedRankingProbe,
    loader: torch.utils.data.DataLoader,
    k: int,
    device: str = "cuda",
    n_perm: int = 1000,
) -> dict:
    """Evaluate probe on a loader, compute IC with permutation null.

    Returns: {
        'daily_ic': float,  # mean Spearman
        'perm_null_975pct': float,
        'p_value': float,
        'n_days': int
    }
    """
    probe.eval()
    probe.to(device)

    all_scores = []
    all_returns = []
    all_masks = []

    with torch.no_grad():
        for batch in loader:
            X, mask, returns_t, _ = batch
            X = X.to(device)
            mask = mask.to(device)
            scores = probe(X, mask)
            fwd_returns_k = returns_t[k]

            all_scores.append(scores.cpu().numpy())
            all_returns.append(fwd_returns_k.numpy())
            all_masks.append(mask.cpu().numpy())

    scores = np.concatenate(all_scores, axis=0)  # [B_total, m]
    returns = np.concatenate(all_returns, axis=0)  # [B_total, n_global]
    masks = np.concatenate(all_masks, axis=0)  # [B_total, m]

    # Compute daily IC (active-only Spearman)
    daily_ic = compute_daily_ic(torch.from_numpy(scores), torch.from_numpy(returns), torch.from_numpy(masks))

    # Permutation null: shuffle weight-return pairs (within each run/loader)
    # ponytail: simplified null — just shuffle returns for each day independently
    ics_null = []
    np.random.seed(0)
    for _ in range(n_perm):
        returns_perm = returns.copy()
        for t in range(len(returns_perm)):
            np.random.shuffle(returns_perm[t])
        ic_perm = compute_daily_ic(
            torch.from_numpy(scores),
            torch.from_numpy(returns_perm),
            torch.from_numpy(masks),
        )
        ics_null.append(ic_perm)

    perm_null_975 = np.percentile(ics_null, 97.5)
    p_value = np.mean(np.array(ics_null) >= daily_ic)  # one-sided: null >= observed

    return {
        "daily_ic": daily_ic,
        "perm_null_975pct": perm_null_975,
        "p_value": p_value,
        "n_days": len(scores),
    }


def run_supervised_experiment(config: ExperimentConfig, out_dir: Path) -> dict:
    """Run M3 supervised ranking probe experiment.

    Returns a dict with results for each horizon k.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading price panel...")
    panel = load_price_panel(config.data)
    print(f"  Panel: {panel.close.shape[0]} days, {panel.n_global} global assets")

    # Compute forward returns for each horizon
    print("Computing forward returns...")
    k_list = [1, 5, 21]
    fwd_returns = {k: compute_forward_returns(panel, k) for k in k_list}

    # Get train/val split indices from split_config.json
    from ..build_dataset.paths import DATA_DIR

    split_config_path = DATA_DIR / "split_config.json"
    if split_config_path.exists():
        import json
        with open(split_config_path) as f:
            split_config = json.load(f)
        from ..build_dataset.manifest import iter_fit_windows
        fit_windows = list(iter_fit_windows(split_config))
    else:
        raise FileNotFoundError(f"split_config.json not found at {split_config_path}")
    if not fit_windows:
        raise ValueError("No fit windows found in split_config.json")

    fit_window = fit_windows[0]  # Use the first (and usually only) window
    train_end_idx = fit_window.train_end_idx
    val_end_idx = fit_window.val_end_idx

    print(f"  Train: [0, {train_end_idx}] ({train_end_idx + 1} days)")
    print(f"  Val:   [{train_end_idx + 1}, {val_end_idx}] ({val_end_idx - train_end_idx} days)")

    # Results for each horizon
    results = {}

    for k in k_list:
        print(f"\nHorizon k={k}")

        # Create probe
        probe = SupervisedRankingProbe(
            window=config.data.window,
            conv1_out_channels=2,
            conv2_out_channels=20,
            n_features=len(config.data.features),
        )

        # Create loaders
        print(f"  Creating data loaders...")
        loaders = list(
            create_train_val_loaders(
                panel,
                config.data,
                fwd_returns,
                train_end_idx,
                val_end_idx,
                batch_size=50,
            )
        )

        train_loader = dict(loaders)["train"]
        val_loader = dict(loaders)["val"]

        # Train
        print(f"  Training probe on {k}-day horizon...")
        probe = train_probe(probe, train_loader, k, epochs=20, device=config.train.device)

        # Evaluate
        print(f"  Evaluating on train split...")
        train_result = evaluate_probe(probe, train_loader, k, device=config.train.device)
        print(f"    IC={train_result['daily_ic']:.4f}, p={train_result['p_value']:.3f}")

        print(f"  Evaluating on val split...")
        val_result = evaluate_probe(probe, val_loader, k, device=config.train.device)
        print(f"    IC={val_result['daily_ic']:.4f}, p={val_result['p_value']:.3f}")

        results[k] = {
            "train": train_result,
            "val": val_result,
            "signal": val_result["daily_ic"] > val_result["perm_null_975pct"],
        }

        # Save checkpoint
        ckpt_path = out_dir / f"probe_k{k}.pt"
        torch.save(probe.state_dict(), ckpt_path)
        print(f"  Saved checkpoint to {ckpt_path.name}")

    # Write results
    results_path = out_dir / "supervised_results.json"
    with open(results_path, "w") as f:
        # Convert numpy types to Python for JSON serialization
        results_serializable = {}
        for k, res in results.items():
            results_serializable[str(k)] = {
                "train": {kk: float(vv) for kk, vv in res["train"].items()},
                "val": {kk: float(vv) for kk, vv in res["val"].items()},
                "signal": bool(res["signal"]),
            }
        json.dump(results_serializable, f, indent=2)

    print(f"\nResults saved to {results_path}")
    return results
