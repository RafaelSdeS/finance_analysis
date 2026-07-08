"""
Information Coefficient (IC) signal analysis for feature ranking.

Computes daily cross-sectional Spearman rank IC of each feature vs forward returns.
Identifies which features have exploitable signal at different horizons (5-day, 21-day).
Output: ranked table (stdout) + JSON results for later reference.

Usage:
  python -m src.agent.ic_analysis [--dataset <path>] [--horizons 5 21] [--out <path>]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Features to skip (identifiers, raw prices, balance-sheet levels, etc.)
SKIP_COLS = {
    "ticker", "trade_date", "open", "high", "low", "close", "adj_close",
    "volume", "volume_adjusted", "traded_amount", "num_trades",
    "reference_date", "shares_outstanding", "corporate_name", "trade_name",
    "cvm_code", "cnpj", "sector", "status",
    "net_income", "equity", "net_revenue", "total_debt", "ebitda", "ebit",
    "net_debt", "cash", "total_assets", "current_assets", "current_liabilities",
    "market_cap", "log_return",
}

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_panel(dataset_path: Path, horizons: list[int] = None):
    """Load panel data and compute forward returns.

    Returns:
        dict[str, pd.DataFrame]: feature_name -> [dates x tickers] matrix
        dict[int, pd.DataFrame]: horizon -> [dates x tickers] forward returns
    """
    if horizons is None:
        horizons = [5, 21]

    df = pd.read_parquet(dataset_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {dataset_path.name}")

    # Drop CASH if present (synthetic row from training file)
    if "ticker" in df.columns:
        df = df[df["ticker"] != "CASH"].reset_index(drop=True)

    # Identify candidate features (float columns, not in skip list)
    candidates = [c for c in df.columns if df[c].dtype.kind == "f" and c not in SKIP_COLS]
    logger.info(f"Identified {len(candidates)} candidate features for IC analysis")

    # Build feature matrices [dates x tickers]
    features = {}
    for feat in candidates:
        feat_df = df.pivot_table(index="trade_date", columns="ticker", values=feat, aggfunc="first")
        if len(feat_df) > 0:
            features[feat] = feat_df

    # Build forward-return matrices per horizon
    price_df = df.pivot_table(index="trade_date", columns="ticker", values="adj_close", aggfunc="first")
    fwd_rets = {}
    for h in horizons:
        fwd = np.log(price_df.shift(-h) / price_df)
        fwd_rets[h] = fwd
        logger.info(f"Forward {h}d returns: {fwd.notna().sum().mean():.0f} names per date on avg")

    return features, fwd_rets


def daily_rank_ic(feat_mat: pd.DataFrame, fwd_mat: pd.DataFrame, min_names: int = 10) -> pd.Series:
    """Vectorized Spearman rank IC: row-wise rank correlation.

    Each row (date) is independently ranked and correlated.
    Days with <min_names joint-valid observations are excluded.
    """
    # Rank each row (NaN-aware)
    feat_rank = feat_mat.rank(axis=1, na_option="keep", method="average")
    fwd_rank = fwd_mat.rank(axis=1, na_option="keep", method="average")

    # Row-wise Pearson of ranks with joint-NaN masking
    ic_series = []
    for date in feat_rank.index:
        f_r = feat_rank.loc[date].values
        r_r = fwd_rank.loc[date].values

        valid = ~(np.isnan(f_r) | np.isnan(r_r))
        n_valid = valid.sum()

        if n_valid < min_names:
            ic_series.append(np.nan)
        else:
            f_r_clean = f_r[valid]
            r_r_clean = r_r[valid]
            # Pearson of ranks = Spearman
            corr = np.corrcoef(f_r_clean, r_r_clean)[0, 1]
            ic_series.append(corr)

    return pd.Series(ic_series, index=feat_rank.index)


def summarize(ic_series: pd.Series, horizon: int, feature_name: str) -> dict:
    """Summarize IC series: mean, std, naive t-stat, non-overlap t-stat, by-period."""
    valid_ic = ic_series.dropna()
    if len(valid_ic) == 0:
        return {"feature": feature_name, "horizon": horizon, "n_days": 0}

    mean_ic = float(valid_ic.mean())
    std_ic = float(valid_ic.std())
    n = len(valid_ic)

    # Naive t-stat (inflated for overlapping windows)
    naive_t = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else 0.0

    # Non-overlap t-stat: subsample every h-th observation
    non_overlap_ic = valid_ic.iloc[::horizon]
    non_overlap_t = (
        float(non_overlap_ic.mean() / (non_overlap_ic.std() / np.sqrt(len(non_overlap_ic))))
        if len(non_overlap_ic) > 1 and non_overlap_ic.std() > 0
        else 0.0
    )

    # By 5-year period
    period_means = {}
    for year_bucket in range(2000, 2030, 5):
        mask = (valid_ic.index.year >= year_bucket) & (valid_ic.index.year < year_bucket + 5)
        period_data = valid_ic[mask]
        if len(period_data) > 0:
            period_means[f"{year_bucket}-{year_bucket+4}"] = float(period_data.mean())

    return {
        "feature": feature_name,
        "horizon": horizon,
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "n_days": n,
        "naive_t_stat": naive_t,
        "non_overlap_t_stat": non_overlap_t,
        "period_means": period_means,
    }


def main():
    parser = argparse.ArgumentParser(description="IC signal analysis")
    parser.add_argument("--dataset", type=Path, default=_PROJECT_ROOT / "data/processed/ml_dataset.parquet",
                        help="Path to ml_dataset.parquet (not ml_dataset_training.parquet)")
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 21],
                        help="Forward return horizons in trading days")
    parser.add_argument("--min-names", type=int, default=10,
                        help="Minimum valid names per date to include")
    parser.add_argument("--out", type=Path, default=_PROJECT_ROOT / "data/backtest/ic_analysis.json",
                        help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Load data
    features, fwd_rets = load_panel(args.dataset, args.horizons)

    # Compute IC per feature and horizon
    results = {}
    for horizon in sorted(args.horizons):
        logger.info(f"\n=== Horizon {horizon}d ===")
        results[horizon] = []

        fwd_mat = fwd_rets[horizon]

        for feat_name, feat_mat in sorted(features.items()):
            # Align to same dates and tickers
            shared_dates = fwd_mat.index.intersection(feat_mat.index)
            shared_tickers = fwd_mat.columns.intersection(feat_mat.columns)
            if len(shared_dates) < 20 or len(shared_tickers) < 10:
                continue

            feat_aligned = feat_mat.loc[shared_dates, shared_tickers]
            fwd_aligned = fwd_mat.loc[shared_dates, shared_tickers]

            ic = daily_rank_ic(feat_aligned, fwd_aligned, args.min_names)
            summary = summarize(ic, horizon, feat_name)

            if summary["n_days"] > 0:
                results[horizon].append(summary)

        # Sort by |mean_ic|
        results[horizon].sort(key=lambda x: abs(x.get("mean_ic", 0)), reverse=True)

        # Print table
        print(f"\n{'Feature':<40} {'Mean IC':>10} {'Naive t':>10} {'NoOverlap t':>12} {'N Days':>8}")
        print("-" * 90)
        for r in results[horizon][:30]:  # Top 30
            print(f"{r['feature']:<40} {r.get('mean_ic', 0):>10.4f} {r.get('naive_t_stat', 0):>10.3f} "
                  f"{r.get('non_overlap_t_stat', 0):>12.3f} {r.get('n_days', 0):>8.0f}")

    # Write JSON
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved → {args.out}")


if __name__ == "__main__":
    main()
