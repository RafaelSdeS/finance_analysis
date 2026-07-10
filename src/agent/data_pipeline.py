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

import argparse
import dataclasses
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.agent.config import AgentConfig, DEFAULT_CONFIG, generate_windows, window_to_config

logger = logging.getLogger(__name__)

TENSORS_PATH = "data/processed/agent_tensors.npz"
SCALER_PATH = "artifacts/models/feature_scaler.pkl"
MIN_ROWS_PER_TICKER = 252  # ~1 trading year; drops stub listings


def select_universe(caps: pd.DataFrame, ticker_pool: list[str], universe_size: int,
                     cutoff: str | None = None) -> set[str]:
    """Top-N tickers from ticker_pool by mean market cap.

    If cutoff is given, only rows with trade_date <= cutoff are used — the
    selection is then point-in-time (no lookahead into future market caps).
    cutoff=None (default) reproduces the original full-sample-mean behavior.
    """
    rows = caps[caps["ticker"].isin(ticker_pool)]
    if cutoff is not None:
        rows = rows[rows["trade_date"] <= pd.Timestamp(cutoff)]
    top = rows.groupby("ticker")["market_cap"].mean().nlargest(universe_size)
    return set(top.index)


def cross_sectional_transform(feat_tensor: np.ndarray, mask: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Per-date cross-sectional rank or z-score of each feature, computed ONLY over active
    names that day (DIAGNOSIS_PLAN.md Phase 2.3).

    Unlike the existing per-window GLOBAL scaler (fit once on a 10+ year train span and
    applied unchanged to every date), this comparison is always relative to that SAME
    day's peers, so decades of regime drift in a feature's typical level/scale can't swamp
    the cross-sectional signal the policy actually needs (rank stocks against TODAY's
    market, not a multi-year historical average).

    Args:
        feat_tensor: [D, N, F] raw feature values, NaN where inactive/missing.
        mask:        [D, N] bool, True where that (date, ticker) is active.
        method: "zscore" (per-date mean-0/std-1) or "rank" (per-date percentile rank in
            [-1, 1], 0 = cross-sectional median; ties broken arbitrarily by argsort order,
            not averaged -- ponytail: fine for continuous financial features, exact ties
            are rare, and this avoids an O(D*F) scipy.stats.rankdata loop).

    Returns:
        [D, N, F] array, same shape as feat_tensor. Inactive or single-name-active cells
        are 0 (matching this pipeline's existing post-normalization NaN->0 convention;
        cross-sectional stats are undefined with <2 active names).
    """
    if method not in ("zscore", "rank"):
        raise ValueError(f"method must be 'zscore' or 'rank', got {method!r}")

    D, N, F = feat_tensor.shape
    valid = mask[:, :, None] & ~np.isnan(feat_tensor)  # [D, N, F]
    n_valid = valid.sum(axis=1)  # [D, F]

    if method == "zscore":
        filled = np.where(valid, feat_tensor, 0.0)
        count = np.maximum(n_valid, 1)[:, None, :]  # [D, 1, F]
        mean = filled.sum(axis=1, keepdims=True) / count
        var = np.where(valid, (feat_tensor - mean) ** 2, 0.0).sum(axis=1, keepdims=True) / count
        std = np.sqrt(var)
        std = np.where(std < 1e-8, 1.0, std)
        out = (feat_tensor - mean) / std
    else:  # rank
        filled = np.where(valid, feat_tensor, np.inf)  # NaN/inactive sorts last, excluded via n_valid below
        order = np.argsort(filled, axis=1)
        ranks = np.argsort(order, axis=1).astype(np.float64)  # 0-indexed rank per (date, feature)
        denom = np.maximum(n_valid - 1, 1)[:, None, :]
        out = (ranks / denom) * 2.0 - 1.0  # [-1, 1]

    out = np.where(valid & (n_valid[:, None, :] >= 2), out, 0.0)
    return out.astype(np.float32)


def build_tensors(config: AgentConfig = DEFAULT_CONFIG) -> dict:
    """Pivot long-format dataset into dense [dates, tickers, features] tensors."""
    features = config.base_state_features
    cols = ["ticker", "trade_date"] + features
    logger.info("Loading %s (%d columns)", config.dataset_path, len(cols))
    df = pd.read_parquet(config.dataset_path, columns=cols)

    # Universe: every ticker with at least one trading year of data
    ticker_counts = df.groupby("ticker").size()
    base_universe = sorted(ticker_counts[ticker_counts >= MIN_ROWS_PER_TICKER].index)
    dropped = ticker_counts[ticker_counts < MIN_ROWS_PER_TICKER]
    logger.info(
        "Universe: %d tickers (dropped %d with <%d rows: %s)",
        len(base_universe), len(dropped), MIN_ROWS_PER_TICKER, list(dropped.index),
    )

    pit_membership: dict[str, set[str]] | None = None
    if config.universe_size is not None:
        caps = pd.read_parquet(config.dataset_path, columns=["ticker", "trade_date", "market_cap"])
        if config.pit_universe:
            # One selection per rolling window, using only data <= that window's train_end
            # (mirrors the per-window scaler discipline in run_pipeline/fit_train_scaler).
            windows = generate_windows(
                config.dataset_start, config.dataset_end,
                config.window_train_years, config.window_test_years,
            )
            cutoffs = sorted({window_to_config(w, config).train_end for w in windows})
            pit_membership = {
                cutoff: select_universe(caps, base_universe, config.universe_size, cutoff=cutoff)
                for cutoff in cutoffs
            }
            universe = sorted(set().union(*pit_membership.values()))
            if "CASH" in base_universe:
                universe = sorted(set(universe) | {"CASH"})
            logger.info(
                "Universe: point-in-time top %d per window (%d cutoffs), %d tickers in union (+CASH)",
                config.universe_size, len(cutoffs), len(universe),
            )
        else:
            selected = select_universe(caps, base_universe, config.universe_size, cutoff=None)
            if "CASH" in base_universe:
                selected.add("CASH")  # synthetic asset has no market_cap; exempt from ranking, always kept
            universe = sorted(selected)
            logger.info("Universe filtered to top %d tickers by mean market cap (+CASH)", config.universe_size)
    else:
        universe = base_universe

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

    if config.cross_sectional_features:
        cs_source = config.price_features + config.fundamental_features  # macro excluded: degenerate cross-section
        cs_idx = [features.index(f) for f in cs_source]
        cs_tensor = cross_sectional_transform(feat_tensor[:, :, cs_idx], mask, method=config.cross_sectional_method)
        feat_tensor = np.concatenate([feat_tensor, cs_tensor], axis=2)
        logger.info(
            "Cross-sectional features: added %d channels (method=%s) -> %d total features",
            cs_tensor.shape[2], config.cross_sectional_method, feat_tensor.shape[2],
        )

    result = {
        "features": feat_tensor,
        "returns": returns,
        "mask": mask,
        "dates": dates,
        "tickers": np.array(universe),
    }

    if pit_membership is not None:
        # One boolean row per window cutoff, True where that ticker was in THAT window's
        # PIT top-N (+ CASH, always active regardless of ranking). env.py ANDs the row
        # matching its config.train_end into the time-varying activity mask, so a stock
        # only ever appears in the state/action space for windows where it was actually
        # a top-N name at decision time.
        cutoffs_sorted = sorted(pit_membership.keys())
        membership_mask = np.zeros((len(cutoffs_sorted), n_tickers), dtype=bool)
        for ci, cutoff in enumerate(cutoffs_sorted):
            selected = pit_membership[cutoff]
            for t in selected:
                membership_mask[ci, tick_idx[t]] = True
            if "CASH" in tick_idx:
                membership_mask[ci, tick_idx["CASH"]] = True
        result["pit_cutoffs"] = np.array(cutoffs_sorted)
        result["pit_membership_mask"] = membership_mask

    return result


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
    # Keyed by the actual train_end cutoff (not window.window_id): window_id is a positional index
    # that depends on (window_train_years, window_test_years), so it's unstable across different CLI
    # overrides (e.g. --train-years 2 --test-years 1 for a smoke run generates a different window_id
    # → date mapping than the default config used here). The cutoff date is what the scaler actually
    # was fit on, so it's a stable, self-describing key regardless of which window scheme produced it.
    pit_cutoffs = list(tensors["pit_cutoffs"]) if "pit_membership_mask" in tensors else None

    scalers = {}
    for window in windows:
        window_config = window_to_config(window, config)
        cutoff = window_config.train_end  # train_end after val carving: the actual train boundary for this window
        scaler_tensors = tensors
        if pit_cutoffs is not None:
            # Fit each window's scaler only on cells that were actually in THAT window's
            # PIT universe, so precomputed scaler stats match what env.py restricts to live
            # (see env.py's _load_normalized_tensors PIT masking).
            if cutoff in pit_cutoffs:
                row = tensors["pit_membership_mask"][pit_cutoffs.index(cutoff)]
                scaler_tensors = {**tensors, "mask": tensors["mask"] & row[np.newaxis, :]}
            else:
                logger.warning("No PIT membership row for cutoff=%s — scaler fit on full universe", cutoff)
        scaler = fit_train_scaler(scaler_tensors, cutoff)
        scalers[cutoff] = scaler
        logger.info("  Window %d: scaler fitted (cutoff=%s)", window.window_id, cutoff)

    # Stamp the feature list so env.py can fail loudly on a stale tensor file
    # (a feature-list change that keeps the same count is otherwise silent).
    tensors["feature_names"] = np.array(config.state_features)
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

    # Self-check: verify every window got a scaler
    assert all(window_to_config(w, config).train_end in scalers for w in windows), "Missing scaler for some window"
    print(f"✓ Verified: all {len(windows)} windows have scalers")


def _selfcheck_universe_filter() -> None:
    """Verify universe_size caps the ticker count and stays a subset of the unfiltered universe."""
    config = dataclasses.replace(DEFAULT_CONFIG, universe_size=5)
    tensors = build_tensors(config)
    full_universe = set(build_tensors(DEFAULT_CONFIG)["tickers"])

    # +1 allowance: CASH is always added on top of universe_size (exempt from ranking).
    assert len(tensors["tickers"]) <= 6, f"expected <=5 stocks +CASH, got {len(tensors['tickers'])}"
    assert set(tensors["tickers"]) <= full_universe, "filtered universe must be a subset of the unfiltered one"
    print(f"✓ Universe filter self-check passed: {len(tensors['tickers'])} tickers (requested <=5 +CASH)")


def _selfcheck_pit_universe() -> None:
    """Verify pit_universe=True produces per-window membership rows that stay within the
    non-PIT top-N union (PIT is strictly a per-window restriction of the same candidate pool)."""
    config = dataclasses.replace(DEFAULT_CONFIG, universe_size=10, pit_universe=True)
    tensors = build_tensors(config)
    assert "pit_membership_mask" in tensors, "pit_universe=True must produce pit_membership_mask"
    assert "pit_cutoffs" in tensors, "pit_universe=True must produce pit_cutoffs"

    mask, cutoffs, tickers = tensors["pit_membership_mask"], tensors["pit_cutoffs"], tensors["tickers"]
    assert mask.shape == (len(cutoffs), len(tickers)), f"shape mismatch: {mask.shape}"
    # Every window selects at most universe_size tickers (CASH exempt from the cap)
    is_cash = tickers == "CASH"
    for row in mask:
        n_selected = int((row & ~is_cash).sum())
        assert n_selected <= config.universe_size, f"window selected {n_selected} > universe_size"
    # CASH (if present) is active in every window
    if is_cash.any():
        assert mask[:, is_cash].all(), "CASH must be active in every PIT window"
    print(f"✓ PIT universe self-check passed: {len(cutoffs)} windows, {len(tickers)} tickers in union")


def _selfcheck_cross_sectional_features() -> None:
    """Verify build_tensors(cross_sectional_features=True) against the REAL dataset (no
    disk writes -- build_tensors never touches TENSORS_PATH/SCALER_PATH, only run_pipeline
    does): tensor shape matches the extended state_features count, and the cross-sectional
    channels carry real (non-degenerate) signal, not silently all-zero."""
    config = dataclasses.replace(DEFAULT_CONFIG, universe_size=10, cross_sectional_features=True)
    tensors = build_tensors(config)

    n_base = len(config.base_state_features)
    n_expected = len(config.state_features)
    assert tensors["features"].shape[2] == n_expected, (
        f"tensor has {tensors['features'].shape[2]} feature channels, expected {n_expected}"
    )
    assert n_expected > n_base, "cross_sectional_features=True must add channels beyond the base set"

    cs_channels = tensors["features"][:, :, n_base:]
    active_std = np.nanstd(cs_channels[tensors["mask"]])
    assert active_std > 0.1, (
        f"cross-sectional channels look degenerate (std={active_std:.4f} on active cells) -- "
        f"expected real, non-trivial spread from a zscore transform"
    )
    print(f"✓ Cross-sectional features self-check passed: {n_base} base -> {n_expected} total "
          f"channels, active-cell std={active_std:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build agent tensors + fit rolling-window scalers")
    parser.add_argument("--universe-size", type=int, default=None,
                         help="Filter to top-N tickers by mean market cap (default: no filtering)")
    parser.add_argument("--pit-universe", action="store_true",
                         help="Select universe per-window using only data <= that window's train_end "
                              "(no lookahead into future market caps); requires --universe-size")
    parser.add_argument("--cross-sectional-features", action="store_true",
                         help="Add per-date rank/z-score channels of every price+fundamental "
                              "feature, computed only over active names that day (DIAGNOSIS_PLAN.md "
                              "Phase 2.3) -- doubles state_features' price+fundamental count")
    parser.add_argument("--cross-sectional-method", type=str, default="zscore", choices=["zscore", "rank"],
                         help="Method for --cross-sectional-features (default: zscore)")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    if args.universe_size is not None:
        config = dataclasses.replace(DEFAULT_CONFIG, universe_size=args.universe_size)
    if args.pit_universe:
        if args.universe_size is None:
            raise SystemExit("--pit-universe requires --universe-size")
        config = dataclasses.replace(config, pit_universe=True)
    if args.cross_sectional_features:
        config = dataclasses.replace(config, cross_sectional_features=True, cross_sectional_method=args.cross_sectional_method)

    run_pipeline(config)
