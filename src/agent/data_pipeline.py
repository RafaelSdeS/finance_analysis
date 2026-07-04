"""
Data Pipeline for PortfolioEnv

Converts the long-format ml_dataset_training.parquet into dense numpy tensors
the environment can step through quickly (no pandas in the RL hot loop):

  features: [n_dates, n_tickers, n_features]  raw (unnormalized) state features
  returns:  [n_dates, n_tickers]              daily log returns (NaN if inactive)
  mask:     [n_dates, n_tickers]              True where ticker traded that day

Also fits a StandardScaler on TRAIN dates only (no lookahead) and saves it
for reuse by validation, test, and live inference.

Run once (or whenever the dataset changes):
    python src/agent/data_pipeline.py
"""

import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.agent.config import AgentConfig, DEFAULT_CONFIG, generate_windows, window_to_config

logger = logging.getLogger(__name__)

TENSORS_PATH = "data/processed/agent_tensors.npz"
SCALER_PATH = "data/models/feature_scaler.pkl"
MIN_ROWS_PER_TICKER = 252  # ~1 trading year; drops stub listings


def build_tensors(config: AgentConfig = DEFAULT_CONFIG) -> dict:
    """Pivot long-format dataset into dense [dates, tickers, features] tensors."""
    features = config.state_features
    cols = ["ticker", "trade_date"] + features
    logger.info("Loading %s (%d columns)", config.dataset_path, len(cols))
    df = pd.read_parquet(config.dataset_path, columns=cols)

    # Universe: every ticker with at least one trading year of data
    ticker_counts = df.groupby("ticker").size()
    universe = sorted(ticker_counts[ticker_counts >= MIN_ROWS_PER_TICKER].index)
    dropped = ticker_counts[ticker_counts < MIN_ROWS_PER_TICKER]
    logger.info(
        "Universe: %d tickers (dropped %d with <%d rows: %s)",
        len(universe), len(dropped), MIN_ROWS_PER_TICKER, list(dropped.index),
    )
    df = df[df["ticker"].isin(universe)]

    # Trading calendar: union of all dates in the filtered data
    dates = np.sort(df["trade_date"].unique())
    date_idx = {d: i for i, d in enumerate(dates)}
    tick_idx = {t: i for i, t in enumerate(universe)}
    n_dates, n_tickers, n_features = len(dates), len(universe), len(features)
    logger.info("Tensor shape: [%d dates, %d tickers, %d features]", n_dates, n_tickers, n_features)

    # Fill dense tensors (vectorized scatter via integer indices)
    rows = df["trade_date"].map(date_idx).to_numpy()
    cols_ = df["ticker"].map(tick_idx).to_numpy()

    feat_tensor = np.full((n_dates, n_tickers, n_features), np.nan, dtype=np.float32)
    feat_tensor[rows, cols_, :] = df[features].to_numpy(dtype=np.float32)

    mask = np.zeros((n_dates, n_tickers), dtype=bool)
    mask[rows, cols_] = True

    returns_idx = features.index("returns")
    returns = feat_tensor[:, :, returns_idx].copy()  # [n_dates, n_tickers], NaN if inactive

    return {
        "features": feat_tensor,
        "returns": returns,
        "mask": mask,
        "dates": dates,
        "tickers": np.array(universe),
    }


def fit_train_scaler(tensors: dict, cutoff: str) -> StandardScaler:
    """Fit StandardScaler on active train-date cells up to cutoff (no lookahead)."""
    dates = pd.to_datetime(tensors["dates"])
    train_slice = dates <= pd.Timestamp(cutoff)

    train_feats = tensors["features"][train_slice]        # [train_dates, tickers, features]
    train_mask = tensors["mask"][train_slice]             # [train_dates, tickers]
    active_rows = train_feats[train_mask]                 # [active_cells, features]

    scaler = StandardScaler()
    # NaN-tolerant fit: StandardScaler ignores NaN via manual masking per feature
    scaler.mean_ = np.nanmean(active_rows, axis=0)
    scaler.scale_ = np.nanstd(active_rows, axis=0)
    scaler.scale_[scaler.scale_ == 0] = 1.0  # constant features: avoid div-by-zero
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = active_rows.shape[1]

    logger.info(
        "Scaler fitted on %s active train cells (cutoff=%s)",
        f"{len(active_rows):,}", cutoff,
    )
    return scaler


def run_pipeline(config: AgentConfig = DEFAULT_CONFIG) -> None:
    """Build tensors, fit one scaler per rolling window, save both to disk."""
    tensors = build_tensors(config)

    # Fit one scaler per window: each uses that window's carved train_end as cutoff (no lookahead per window)
    windows = generate_windows(
        config.dataset_start, config.dataset_end,
        config.window_train_years, config.window_test_years,
    )
    scalers = {}
    for window in windows:
        window_config = window_to_config(window, config)
        cutoff = window_config.train_end  # train_end after val carving: the actual train boundary for this window
        scaler = fit_train_scaler(tensors, cutoff)
        scalers[window.window_id] = scaler
        logger.info("  Window %d: scaler fitted (cutoff=%s)", window.window_id, cutoff)

    np.savez_compressed(TENSORS_PATH, **tensors)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scalers, f)

    logger.info("Saved tensors → %s", TENSORS_PATH)
    logger.info("Saved %d scalers (one per window) → %s", len(scalers), SCALER_PATH)

    # Summary for the console
    dates = pd.to_datetime(tensors["dates"])
    print(f"\n{'=' * 60}")
    print("DATA PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Tensors:  {tensors['features'].shape} → {TENSORS_PATH}")
    print(f"Universe: {len(tensors['tickers'])} tickers")
    print(f"Calendar: {dates.min().date()} → {dates.max().date()} ({len(dates)} days)")
    print(f"Active cells: {tensors['mask'].sum():,} / {tensors['mask'].size:,} "
          f"({tensors['mask'].mean():.1%})")
    print(f"Scalers:  {len(scalers)} (one per rolling window) → {SCALER_PATH}")

    # Self-check: verify monotonic window_ids and cutoffs
    cutoffs = [scalers[w.window_id] for w in windows]
    assert all(scalers[w.window_id] is not None for w in windows), "Missing scaler for some window"
    print(f"✓ Verified: all {len(windows)} windows have scalers")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_pipeline()
