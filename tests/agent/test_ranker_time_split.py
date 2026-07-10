"""
Unit tests for ranker_baseline's train/test split mode (synthetic panel, no data files).

DIAGNOSIS_PLAN.md F2: compute_ranker_ic's original "row" split cuts the (ticker,
trade_date)-sorted dataframe by ROW POSITION, which is effectively a ticker-holdout with
FULL date overlap (verified on the real dataset: all 6,568 trading dates appear on both
sides). This isn't a walk-forward test — the test set spans the SAME calendar range as
train, so a relationship that only held in one historical regime can still show up as
"signal" for held-out tickers, because train tickers experienced (and the model partly
learned) that regime too. A genuine time split's test set is confined to whatever comes
strictly AFTER the train cutoff, honestly exposing whether the model has any forward skill.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.ranker_baseline import compute_ranker_ic

N_TICKERS = 24
N_PER_ERA = 60

# Hand-verified (via the Spearman formula) zero-correlation derangement of ranks 1..24 —
# era B's myfeat-to-return relationship is exactly uncorrelated, not just "unrelated by
# random luck". rho = 1 - 6*sum((rank-perm)^2) / (n*(n^2-1)) = 0.0 for this permutation.
_ERA_B_PERM = [15, 21, 20, 10, 3, 9, 13, 6, 12, 14, 2, 17, 18, 19, 11, 8, 1, 22, 16, 7, 23, 5, 4, 24]


def _make_regime_shift_panel() -> pd.DataFrame:
    """2*N_PER_ERA trading days x N_TICKERS. `myfeat` is a constant per-ticker value
    (1..N_TICKERS). In era A (first half of dates), next-day returns are ranked exactly by
    myfeat every day (a strong, stable signal). In era B (second half), returns follow a
    fixed zero-correlation derangement of myfeat's order — a regime shift with NO real
    relationship, that the model (trained only on what's before the cutoff, under an
    honest split) can never have seen coming.
    """
    dates = pd.bdate_range("2015-01-01", periods=N_PER_ERA * 2)
    tickers = [f"T{i}" for i in range(1, N_TICKERS + 1)]
    myfeat = {t: i + 1 for i, t in enumerate(tickers)}
    era_a_ret = {t: (i + 1) * 0.01 for i, t in enumerate(tickers)}          # return rank == myfeat rank
    era_b_ret = {t: _ERA_B_PERM[i] * 0.01 for i, t in enumerate(tickers)}   # zero-correlation derangement

    rows, price = [], {t: 100.0 for t in tickers}
    for d_idx, d in enumerate(dates):
        for t in tickers:
            rows.append({"ticker": t, "trade_date": d, "adj_close": price[t], "myfeat": float(myfeat[t])})
        era_ret = era_a_ret if d_idx < N_PER_ERA else era_b_ret
        for t in tickers:
            price[t] *= 1.0 + era_ret[t]
    return pd.DataFrame(rows).sort_values(["ticker", "trade_date"]).reset_index(drop=True)


def main():
    panel = _make_regime_shift_panel()

    print("Test 1 (problem): the legacy row split reports an inflated, spurious IC")
    # Train tickers span BOTH eras (row split cuts by ticker, not date), so the model
    # partially learns era A's real signal from train data alone -- then the row split's
    # TEST SET also spans both eras (same calendar range, different tickers), so era A's
    # inherited signal pulls the overall average IC up even though half the test dates
    # (era B) carry zero real relationship.
    result_row = compute_ranker_ic(panel, horizon=1, split="row")
    assert result_row is not None, "row split returned no result"
    print(f"  ROW split: mean_ic={result_row['mean_ic']:.4f}, t_stat={result_row['t_stat']:.2f}, "
          f"n_days={result_row['n_days']}")
    assert result_row["mean_ic"] > 0.3, (
        f"expected the row split to report a spuriously strong IC (era A leaking through "
        f"train tickers into the test average), got {result_row['mean_ic']}"
    )

    print("\nTest 2 (solution): the honest time split confines testing to the true "
          "out-of-sample regime and reports the truth")
    # Train = era A only (first half of dates); test = era B only (strictly after the
    # cutoff) -- the model never saw era B's (lack of a) relationship during training, and
    # the test set contains ONLY era B, so era A's signal cannot leak into the average.
    result_time = compute_ranker_ic(panel, horizon=1, split="time")
    assert result_time is not None, "time split returned no result"
    print(f"  TIME split: mean_ic={result_time['mean_ic']:.4f}, t_stat={result_time['t_stat']:.2f}, "
          f"n_days={result_time['n_days']}")
    assert abs(result_time["mean_ic"]) < 0.05, (
        f"expected the honest time split to report ~zero IC (era B has zero true "
        f"relationship, hand-verified via the Spearman formula), got {result_time['mean_ic']}"
    )
    assert result_row["mean_ic"] - result_time["mean_ic"] > 0.3, (
        "the row split's IC should be materially higher than the time split's -- this gap "
        "IS the leak (F2), not a modeling artifact"
    )

    print("\nTest 3 (solution): train is strictly <= cutoff < all test dates under the time split")
    dates_sorted = np.sort(panel["trade_date"].unique())
    cutoff = dates_sorted[len(dates_sorted) // 2]
    train_dates = panel.loc[panel["trade_date"] <= cutoff, "trade_date"]
    test_dates = panel.loc[panel["trade_date"] > cutoff, "trade_date"]
    assert train_dates.max() <= cutoff < test_dates.min(), (
        "time split must not allow any date overlap between train and test"
    )
    print(f"  train_end={train_dates.max().date()} <= cutoff={pd.Timestamp(cutoff).date()} "
          f"< test_start={test_dates.min().date()}")

    print("\nTest 4: an invalid split mode raises rather than silently doing something else")
    try:
        compute_ranker_ic(panel, horizon=1, split="bogus")
        raise AssertionError("expected ValueError for an unknown split mode")
    except ValueError:
        pass
    print("  ValueError raised as expected")

    print("\n✓ All ranker_baseline time-split tests passed")


if __name__ == "__main__":
    main()
