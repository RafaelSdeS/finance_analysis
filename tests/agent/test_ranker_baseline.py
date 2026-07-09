"""
Unit test for ranker_baseline.compute_ranker_ic (synthetic, no data files).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.ranker_baseline import compute_ranker_ic


def _synthetic_df(n_dates: int = 80, n_tickers: int = 20, seed: int = 42) -> pd.DataFrame:
    """Build a panel with a cross-sectional, ticker-invariant signal->return mapping.

    ranker_baseline's 50/50 split is a row-index split on data pre-sorted by
    [ticker, trade_date] (matching ml_dataset.parquet's sort order) — i.e. a
    held-out-tickers split, not a temporal split. A per-ticker "quality" level
    driving both `signal` and forward returns is the relationship a
    cross-sectional ranker actually needs to generalize across an unseen
    ticker, and is what this codebase's real fundamental features look like
    (e.g. ROA is roughly stable per company, not a time-series predictor).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    tickers = [f"T{i}" for i in range(n_tickers)]
    ticker_quality = rng.normal(size=n_tickers)

    rows = []
    for ticker, q in zip(tickers, ticker_quality):
        signal = q + rng.normal(scale=0.1, size=len(dates))
        daily_log_ret = q / 21 + rng.normal(scale=0.001, size=len(dates))
        price = 100 * np.exp(np.cumsum(daily_log_ret))
        for i, d in enumerate(dates):
            rows.append({
                "ticker": ticker,
                "trade_date": d,
                "adj_close": price[i],
                "signal": signal[i],
                "noise": rng.normal(),
            })
    return pd.DataFrame(rows).sort_values(["ticker", "trade_date"]).reset_index(drop=True)


def main():
    """Synthetic compute_ranker_ic tests."""
    print("✓ Test 1: Ranker extracts signal from a predictive feature")
    df = _synthetic_df()
    result = compute_ranker_ic(df, horizon=21)
    assert result is not None, "compute_ranker_ic returned None on valid synthetic data"
    assert "mean_ic" in result and "t_stat" in result, "missing expected keys in result"
    print(f"  mean_ic={result['mean_ic']:.4f}, t_stat={result['t_stat']:.2f}, n_days={result['n_days']}")
    assert result["mean_ic"] > 0.1, f"expected strong positive IC on synthetic signal, got {result['mean_ic']:.4f}"

    print("\n✓ Test 2: Handles too-few tickers/dates without raising")
    tiny_df = _synthetic_df(n_dates=5, n_tickers=3)
    tiny_result = compute_ranker_ic(tiny_df, horizon=21)
    assert tiny_result is None or "mean_ic" in tiny_result

    print("\n✓ All ranker_baseline tests passed")


if __name__ == "__main__":
    main()
