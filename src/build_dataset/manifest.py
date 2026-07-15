"""
manifest.py — reproducibility manifest, walk-forward split config, and
immutable dataset_v{N} snapshots. Written once per build, after the
parquet itself is on disk.
"""

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .paths import OUTPUT_PATH, ROOT, SCALER_DIR, SPLIT_CONFIG_PATH


# =============================================================================
# BUILD MANIFEST
# =============================================================================

def write_manifest(dataset):
    """Reproducibility record + per-column distribution snapshot, one per build.

    Written next to the parquet as ml_dataset.manifest.json. Comparing two
    manifests (e.g. before/after a code change) surfaces silent distribution
    drift that passes every schema check.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=ROOT,
        ).stdout.strip() or "unknown"
    except OSError:
        commit = "unknown"

    def _f(x):
        return None if pd.isna(x) else round(float(x), 6)

    numeric = dataset.select_dtypes(include="number")
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "rows": len(dataset),
        "tickers": int(dataset["ticker"].nunique()),
        "date_min": str(dataset["trade_date"].min().date()),
        "date_max": str(dataset["trade_date"].max().date()),
        "columns": list(dataset.columns),
        "column_stats": {
            c: {
                "nan_pct": round(float(numeric[c].isna().mean()) * 100, 2),
                "mean": _f(numeric[c].mean()),
                "std": _f(numeric[c].std()),
                "p1": _f(numeric[c].quantile(0.01)),
                "p50": _f(numeric[c].quantile(0.50)),
                "p99": _f(numeric[c].quantile(0.99)),
            }
            for c in numeric.columns
        },
    }
    path = OUTPUT_PATH.with_suffix(".manifest.json")
    path.write_text(json.dumps(manifest, indent=1))
    print(f"Manifest saved to: {path}")
    return manifest


# =============================================================================
# SPLIT CONFIG
# =============================================================================

def compute_split_dates(dataset, train_frac=0.7, val_frac=0.15):
    """Walk-forward train/val/test cutoffs, one pair of dates for the whole dataset.

    Split over unique trade_date, not row count: tickers have different history
    lengths, so a row-count split would let long-history tickers drag the
    boundary later than a short-history ticker would. Splitting on the calendar
    keeps the cutoff dates identical regardless of which tickers are in scope.
    """
    dates = np.sort(dataset["trade_date"].unique())
    train_end = pd.Timestamp(dates[int(len(dates) * train_frac) - 1])
    val_end = pd.Timestamp(dates[int(len(dates) * (train_frac + val_frac)) - 1])
    return train_end, val_end


def write_split_config(dataset, train_frac=0.7, val_frac=0.15):
    """Leak-safe split boundaries as a small json, not materialized parquet copies.

    Filter ml_dataset.parquet by trade_date against these cutoffs at load time
    (train: <= train_end, val: train_end < d <= val_end, test: > val_end)
    instead of keeping three separate parquet files in sync with the source.
    """
    train_end, val_end = compute_split_dates(dataset, train_frac, val_frac)
    is_train = dataset["trade_date"] <= train_end
    is_val = (dataset["trade_date"] > train_end) & (dataset["trade_date"] <= val_end)
    is_test = dataset["trade_date"] > val_end

    config = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_frac": train_frac,
        "val_frac": val_frac,
        "train_end": str(train_end.date()),
        "val_end": str(val_end.date()),
        "rows": {
            "train": int(is_train.sum()),
            "val": int(is_val.sum()),
            "test": int(is_test.sum()),
        },
    }
    SPLIT_CONFIG_PATH.write_text(json.dumps(config, indent=1))
    print(f"Split config saved to: {SPLIT_CONFIG_PATH}")


# =============================================================================
# FIT WINDOWS (evaluation-methodology-agnostic scaler boundaries)
# =============================================================================

@dataclass(frozen=True)
class FitWindow:
    """A boundary a fitted scaler should train on: rows with
    fit_start < trade_date <= fit_end (fit_start=None means "from the
    beginning of history" -- expanding). fold_id names the artifact
    directory when a config resolves to more than one window.
    """
    fold_id: str
    fit_start: Optional[pd.Timestamp]
    fit_end: pd.Timestamp


def iter_fit_windows(split_config: dict) -> list[FitWindow]:
    """Resolve the fit window(s) a scaler should train on, from the active
    split configuration -- the one seam between evaluation methodology and
    scaler fitting (docs/PER_TICKER_SCALING_PLAN.md §3.5). Callers (R3/R4
    fitting code) never read train_end directly; they iterate the windows
    this returns, so a future change to the split config format (rolling
    windows, expanding folds, multiple folds) only ever touches this function.

    Today's split_config.json (train_end/val_end) is the single-window,
    expanding-from-start case. A future multi-fold/rolling format would add
    a branch here (e.g. split_config["folds"]) and nowhere else.
    """
    train_end = pd.Timestamp(split_config["train_end"])
    return [FitWindow(fold_id="full", fit_start=None, fit_end=train_end)]


# =============================================================================
# DATASET VERSIONING
# =============================================================================

def _manifest_fingerprint(manifest):
    """Manifest fields that reflect actual output content (excludes build_at/git_commit)."""
    return {k: manifest[k] for k in ("rows", "tickers", "date_min", "date_max", "columns", "column_stats")}


def nan_regressions(prev_manifest, manifest, threshold=2.0):
    """Report columns whose NaN% rose by >threshold percentage points.

    Args:
        prev_manifest: previous build's manifest dict
        manifest: current build's manifest dict
        threshold: pp rise to flag (default 2.0)

    Returns:
        list of column names that regressed, or empty list.
    """
    regressions = []
    prev_stats = {col: stats.get("nan_pct", 0) for col, stats in prev_manifest.get("column_stats", {}).items()}
    curr_stats = {col: stats.get("nan_pct", 0) for col, stats in manifest.get("column_stats", {}).items()}

    for col in prev_stats:
        if col in curr_stats:
            rise = curr_stats[col] - prev_stats[col]
            if rise > threshold:
                regressions.append(f"{col} ({prev_stats[col]:.2f}% → {curr_stats[col]:.2f}%)")

    return regressions


def sync_dataset_version(manifest):
    """Snapshot the current build into data/processed/dataset_v{N}/, skipping no-op reruns.

    Copies (doesn't re-serialize) ml_dataset.parquet + its manifest + split_config
    + the scalers/ directory (if any were fit for this build) into an immutable,
    incrementing folder so an experiment can cite dataset_v{N} by name -- including
    exactly which fitted scaler it used, not just which parquet. N only bumps when
    the manifest's content fingerprint actually changed vs. the latest existing
    version -- an unchanged rerun is skipped rather than piling up another 250MB+
    copy.
    """
    existing = sorted(
        (int(match.group(1)), path)
        for path in OUTPUT_PATH.parent.glob("dataset_v*")
        if (match := re.fullmatch(r"dataset_v(\d+)", path.name))
    )
    latest_n, latest_dir = existing[-1] if existing else (0, None)

    if latest_dir is not None:
        prev_manifest = json.loads((latest_dir / "ml_dataset.manifest.json").read_text())
        if _manifest_fingerprint(prev_manifest) == _manifest_fingerprint(manifest):
            print(f"No content change vs {latest_dir.name} -- skipping new version.")
            return
        regressions = nan_regressions(prev_manifest, manifest)
        if regressions:
            print("WARNING: NaN% regression vs previous version:")
            for col in regressions:
                print(f"  • {col}")

    version_dir = OUTPUT_PATH.parent / f"dataset_v{latest_n + 1}"
    version_dir.mkdir()
    shutil.copy2(OUTPUT_PATH, version_dir / OUTPUT_PATH.name)
    shutil.copy2(OUTPUT_PATH.with_suffix(".manifest.json"), version_dir / "ml_dataset.manifest.json")
    shutil.copy2(SPLIT_CONFIG_PATH, version_dir / SPLIT_CONFIG_PATH.name)
    if SCALER_DIR.exists():
        shutil.copytree(SCALER_DIR, version_dir / SCALER_DIR.name)
    print(f"Versioned snapshot saved to: {version_dir}")
