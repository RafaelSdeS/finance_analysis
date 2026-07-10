"""
Unit tests for ic_analysis module (synthetic panel, no data files).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.ic_analysis import daily_rank_ic, load_panel, summarize


def _make_universe_signal_panel() -> pd.DataFrame:
    """15 trading days x 12 tickers. `myfeat` perfectly ranks SMALL tickers' next-day
    returns (signal exists there); BIG tickers' returns are a fixed derangement of their
    own feat order (near-zero rank relationship). Daily returns are constant per ticker,
    so every date's cross-section reproduces the identical designed relationship.

    DIAGNOSIS_PLAN.md F2: the agent trades a top-N-by-market-cap ("BIG") universe, but IC
    has historically been measured on the full listed universe (which includes "SMALL").
    This panel demonstrates the resulting bias: full-universe IC can look strong while the
    subset the agent can actually trade carries none of it.
    """
    dates = pd.bdate_range("2020-01-01", periods=16)
    small = [f"SMALL{i}" for i in range(1, 7)]
    big = [f"BIG{i}" for i in range(1, 7)]
    tickers = small + big

    myfeat = {t: i + 1 for i, t in enumerate(tickers)}  # SMALL1..6 -> 1..6, BIG1..6 -> 7..12
    big_perm = [3, 6, 1, 5, 2, 4]  # derangement of BIG's own feat order -> near-zero rank correlation
    daily_ret = {}
    for i, t in enumerate(small):
        daily_ret[t] = (i + 1) * 0.01  # return rank == feat rank within SMALL (range 0.01-0.06)
    for i, t in enumerate(big):
        # +6 offset keeps BIG's return magnitudes strictly above SMALL's (range 0.07-0.12), so in the
        # COMBINED 12-name ranking SMALL/BIG don't interleave -- isolates the within-BIG derangement
        # as the only source of full-universe rank error, instead of also picking up spurious
        # cross-block noise from overlapping return magnitudes.
        daily_ret[t] = (6 + big_perm[i]) * 0.01  # return rank scrambled vs feat rank within BIG

    rows, price = [], {t: 100.0 for t in tickers}
    for d in dates:
        for t in tickers:
            rows.append({"ticker": t, "trade_date": d, "adj_close": price[t], "myfeat": float(myfeat[t])})
        for t in tickers:
            price[t] *= 1.0 + daily_ret[t]
    return pd.DataFrame(rows), set(small), set(big)


def main():
    """Synthetic IC tests: perfect signal, inverted, random, min_names."""
    print("✓ Test 1: Perfect positive IC (feature ranks match forward returns)")
    np.random.seed(42)
    n_dates, n_names = 50, 20
    feat = pd.DataFrame(
        np.random.randn(n_dates, n_names),
        columns=[f"t{i}" for i in range(n_names)],
        index=pd.date_range("2020-01-01", periods=n_dates),
    )
    fwd = feat.copy()  # Identical ranks → IC ≈ 1
    ic = daily_rank_ic(feat, fwd, min_names=10)
    assert ic.mean() > 0.99, f"Expected IC ≈ 1, got {ic.mean()}"
    print(f"  Mean IC: {ic.mean():.4f}")

    print("\n✓ Test 2: Perfect negative IC")
    fwd_inv = -feat  # Inverted ranks → IC ≈ -1
    ic_inv = daily_rank_ic(feat, fwd_inv, min_names=10)
    assert ic_inv.mean() < -0.99, f"Expected IC ≈ -1, got {ic_inv.mean()}"
    print(f"  Mean IC: {ic_inv.mean():.4f}")

    print("\n✓ Test 3: Random IC (should be small)")
    fwd_rand = pd.DataFrame(
        np.random.randn(n_dates, n_names),
        columns=[f"t{i}" for i in range(n_names)],
        index=feat.index,
    )
    ic_rand = daily_rank_ic(feat, fwd_rand, min_names=10)
    assert abs(ic_rand.mean()) < 0.1, f"Expected small IC, got {ic_rand.mean()}"
    print(f"  Mean IC: {ic_rand.mean():.4f}")

    print("\n✓ Test 4: Min names exclusion")
    feat_sparse = feat.copy()
    feat_sparse.iloc[:5] = np.nan  # First 5 dates have <min_names valid
    ic_sparse = daily_rank_ic(feat_sparse, fwd, min_names=10)
    assert ic_sparse.iloc[:5].isna().all(), "Days with <min_names should be NaN"
    print(f"  First 5 dates NaN: {ic_sparse.iloc[:5].isna().sum()}/5")

    print("\n✓ Test 5: Summarize IC")
    summary = summarize(ic, 5, "test_feature")
    assert summary["horizon"] == 5
    assert summary["n_days"] > 0
    assert "mean_ic" in summary
    assert "non_overlap_t_stat" in summary
    print(f"  Summary keys: {list(summary.keys())}")

    print("\n✓ Test 6 (problem): full-universe IC hides that the tradable subset has no signal")
    panel, small_tickers, big_tickers = _make_universe_signal_panel()
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "panel.parquet"
        panel.to_parquet(dataset_path)

        features_full, fwd_full = load_panel(dataset_path, horizons=[1])
        ic_full = daily_rank_ic(features_full["myfeat"], fwd_full[1], min_names=5)
        mean_ic_full = ic_full.mean()
        assert mean_ic_full > 0.7, f"expected strong full-universe IC (SMALL group drives it), got {mean_ic_full}"
        print(f"  Full-universe (12 tickers) mean IC: {mean_ic_full:.4f}")

        print("\n✓ Test 7 (solution): filtering to the tradable (BIG-only) universe collapses the IC")
        features_big, fwd_big = load_panel(dataset_path, horizons=[1], tickers=big_tickers)
        ic_big = daily_rank_ic(features_big["myfeat"], fwd_big[1], min_names=5)
        mean_ic_big = ic_big.mean()
        assert abs(mean_ic_big) < 0.2, f"expected near-zero IC on the signal-free BIG subset, got {mean_ic_big}"
        print(f"  Tradable-universe (6 BIG tickers) mean IC: {mean_ic_big:.4f}")
        assert mean_ic_full - mean_ic_big > 0.3, "filtering must materially change the measured IC"

        print("\n✓ Test 8 (solution): filter is honored exactly — only supplied tickers enter the cross-sections")
        assert set(features_big["myfeat"].columns) == big_tickers, (
            f"filtered panel leaked tickers outside the requested set: {set(features_big['myfeat'].columns)}"
        )
        assert set(fwd_big[1].columns) == big_tickers, "filtered forward-return matrix leaked extra tickers"
        # min_names respected on the filtered (smaller) set: asking for more names than exist -> all-NaN
        ic_over_strict = daily_rank_ic(features_big["myfeat"], fwd_big[1], min_names=10)
        assert ic_over_strict.isna().all(), "min_names > available filtered tickers should yield no valid days"

    print("\n✓ All ic_analysis tests passed")


if __name__ == "__main__":
    main()
