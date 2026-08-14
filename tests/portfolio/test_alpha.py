"""
test_alpha.py -- checks for src/portfolio/alpha.py:
  1. _label_close_dates: exact date arithmetic.
  2. _purge_embargo_mask: exact boundary correctness (the anti-lookahead
     guard's whole job is this boundary, so it's checked directly, not just
     indirectly through a trained model's behavior).
  3. monotone_constraints is actually wired to the right feature.
  4. walk_forward_predict / rank_ic basic correctness.

Fast group (synthetic only). Run: python tests/portfolio/test_alpha.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.portfolio.alpha import (  # noqa: E402
    MONOTONE_FEATURE, _global_trading_dates, _label_close_dates, _purge_embargo_mask, fit,
    predict, rank_ic, shrink_alpha, walk_forward_predict,
)
from src.portfolio.features import feature_columns  # noqa: E402
from tests.test_utils import print_check, print_header, print_section_end  # noqa: E402


def _synthetic_df(n_rows=1000, seed=0, n_tickers=1):
    """n_tickers=1 gives one row per date (fine for purge/monotone checks,
    which don't need cross-sectional variation). rank_ic is inherently
    cross-sectional -- it needs >1 ticker per date to correlate anything."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_rows)
    cols = feature_columns(include_sector=False)
    tickers = [f"T{i}" for i in range(n_tickers)]
    grid = pd.MultiIndex.from_product([dates, tickers], names=["trade_date", "ticker"])
    index_df = pd.DataFrame(index=grid).reset_index()
    n = len(index_df)
    features_df = pd.DataFrame({c: rng.normal(0, 1, n) for c in cols})
    return pd.concat([index_df, features_df], axis=1)


def test_label_close_dates():
    passed = failed = 0
    global_dates = pd.bdate_range("2020-01-01", periods=20)
    trade_dates = pd.Series(global_dates)
    closes = _label_close_dates(trade_dates, global_dates, horizon_td=5)

    exact_ok = all(closes.iloc[i] == global_dates[i + 5] for i in range(15))
    print_check("close date is exactly `horizon_td` positions ahead in the global calendar",
                bool(exact_ok))
    passed, failed = passed + exact_ok, failed + (not exact_ok)

    tail_nat_ok = closes.iloc[15:].isna().all()
    print_check("rows too near the end of history get NaT (no valid window yet)",
                bool(tail_nat_ok))
    passed, failed = passed + tail_nat_ok, failed + (not tail_nat_ok)
    return passed, failed


def test_purge_embargo_mask():
    passed = failed = 0
    dates = pd.bdate_range("2020-01-01", periods=300)
    df = pd.DataFrame({"ticker": "AAA", "trade_date": dates, "label": 0.01})
    horizon_td, embargo_days = 10, 5
    as_of = dates[200]

    mask = _purge_embargo_mask(df, as_of, horizon_td, embargo_days)

    global_dates = dates
    cutoff = as_of - pd.Timedelta(days=embargo_days)
    expected = [
        (i + horizon_td < len(dates)) and (global_dates[i + horizon_td] <= cutoff)
        for i in range(len(dates))
    ]
    matches_ok = list(mask) == expected
    print_check("mask matches an independently hand-computed boundary row-by-row",
                bool(matches_ok), f"mismatches: {sum(m != e for m, e in zip(mask, expected))}")
    passed, failed = passed + matches_ok, failed + (not matches_ok)

    # A row whose label window closes strictly AFTER as_of must never be
    # trainable on -- the core anti-lookahead guarantee.
    still_open = [i for i in range(len(dates)) if i + horizon_td < len(dates)
                  and global_dates[i + horizon_td] > as_of]
    no_lookahead_ok = not any(mask.iloc[i] for i in still_open)
    print_check("no row whose label window closes after as_of is ever included",
                bool(no_lookahead_ok))
    passed, failed = passed + no_lookahead_ok, failed + (not no_lookahead_ok)
    return passed, failed


def test_precomputed_close_dates_matches_recomputed_every_call():
    """walk_forward_predict now computes _label_close_dates ONCE outside its
    retrain loop and threads it through fit -> _purge_embargo_mask, instead
    of _purge_embargo_mask recomputing it fresh at every rebalance date. Pure
    perf hoist -- must produce a BYTE-IDENTICAL mask at several different
    as_of values, not just a similar one."""
    passed = failed = 0
    dates = pd.bdate_range("2020-01-01", periods=300)
    df = pd.DataFrame({"ticker": "AAA", "trade_date": dates, "label": 0.01})
    horizon_td, embargo_days = 10, 5

    global_dates = _global_trading_dates(df)
    close_dates = _label_close_dates(df["trade_date"], global_dates, horizon_td)

    all_match = True
    for as_of in (dates[50], dates[150], dates[200], dates[290]):
        recomputed = _purge_embargo_mask(df, as_of, horizon_td, embargo_days)
        precomputed = _purge_embargo_mask(df, as_of, horizon_td, embargo_days, close_dates=close_dates)
        if list(recomputed) != list(precomputed):
            all_match = False
    print_check("precomputed close_dates gives an identical mask to recomputing it fresh, "
                "at every as_of tested", bool(all_match))
    passed, failed = passed + all_match, failed + (not all_match)
    return passed, failed


def test_monotone_constraint():
    passed = failed = 0
    df = _synthetic_df(n_rows=1500, seed=1)
    # a real, mildly noisy positive relationship on the constrained feature
    rng = np.random.default_rng(2)
    df["label"] = 0.5 * df[MONOTONE_FEATURE] + rng.normal(0, 0.05, len(df))
    as_of = df["trade_date"].iloc[-1] + pd.Timedelta(days=30)  # predict just past all history

    model = fit(df, as_of, horizon_td=1, embargo_days=0, min_train_rows=100, n_estimators=50)
    fit_ok = model is not None
    print_check("model fits on the synthetic panel", bool(fit_ok))
    passed, failed = passed + fit_ok, failed + (not fit_ok)
    if not fit_ok:
        print_section_end(passed, failed)
        return passed, failed

    cols = feature_columns(include_sector=False)
    base_row = df[cols].median().to_frame().T
    sweep = pd.concat([base_row] * 20, ignore_index=True)
    sweep[MONOTONE_FEATURE] = np.linspace(-3, 3, 20)
    preds = model.predict(sweep)

    monotone_ok = bool(np.all(np.diff(preds) >= -1e-9))
    print_check(f"predictions are non-decreasing in {MONOTONE_FEATURE} (monotone_constraints applied)",
                monotone_ok)
    passed, failed = passed + monotone_ok, failed + (not monotone_ok)
    return passed, failed


def test_predict_empty_date():
    """A rebalance date with no matching row (e.g. after universe
    restriction, a ticker qualifies for a period without trading on that
    exact calendar date) must return an empty prediction, not crash."""
    passed = failed = 0
    df = _synthetic_df(n_rows=300, seed=5, n_tickers=5)
    df["label"] = 0.01
    as_of = df["trade_date"].max() - pd.Timedelta(days=200)
    model = fit(df, as_of, horizon_td=1, embargo_days=0, min_train_rows=50, n_estimators=20)

    missing_date = pd.Timestamp("2099-01-01")  # guaranteed no row at this date
    result = predict(model, df, missing_date)
    ok = isinstance(result, pd.Series) and result.empty
    print_check("predict() on a date with zero matching rows returns an empty Series, not a crash",
                bool(ok))
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_shrink_alpha():
    passed = failed = 0
    a = pd.Series({"A": 0.10, "B": -0.05, "C": 0.0})

    zero_ok = shrink_alpha(a, 0.0).equals(a)
    print_check("factor=0 leaves alpha untouched", bool(zero_ok))
    passed, failed = passed + zero_ok, failed + (not zero_ok)

    full_ok = np.allclose(shrink_alpha(a, 1.0).to_numpy(), 0.0)
    print_check("factor=1 flattens alpha to exactly 0", bool(full_ok))
    passed, failed = passed + full_ok, failed + (not full_ok)

    half = shrink_alpha(a, 0.5)
    half_ok = np.allclose(half.to_numpy(), (a * 0.5).to_numpy())
    print_check("factor=0.5 halves every value, sign preserved", bool(half_ok),
                f"got {half.to_dict()}")
    passed, failed = passed + half_ok, failed + (not half_ok)
    return passed, failed


def test_walk_forward_and_rank_ic():
    passed = failed = 0
    df = _synthetic_df(n_rows=300, seed=3, n_tickers=15)
    rng = np.random.default_rng(4)
    signal_col = feature_columns(include_sector=False)[0]
    df["label"] = 0.3 * df[signal_col] + rng.normal(0, 0.2, len(df))

    reb_quarter_dates = df["trade_date"].drop_duplicates().iloc[::63].iloc[:8]
    reb_dates = pd.DatetimeIndex(reb_quarter_dates)  # sparse, quarterly-ish spacing
    preds = walk_forward_predict(df, reb_dates, horizon_td=1, embargo_days=0,
                                  min_train_rows=200, n_estimators=30)

    shape_ok = list(preds.columns) == ["date", "ticker", "alpha"] and len(preds) > 0
    print_check("walk_forward_predict returns the expected shape", bool(shape_ok),
                f"{len(preds)} rows across {preds['date'].nunique() if len(preds) else 0} dates")
    passed, failed = passed + shape_ok, failed + (not shape_ok)

    skip_early_ok = preds["date"].min() > reb_dates[0]
    print_check("early dates without enough training history are skipped", bool(skip_early_ok))
    passed, failed = passed + skip_early_ok, failed + (not skip_early_ok)

    ic = rank_ic(preds, df)
    ic_reasonable_ok = ic.notna().any() and ic.mean() > 0
    print_check("rank_ic is positive on average for a genuinely informative feature",
                bool(ic_reasonable_ok), f"mean IC={ic.mean():.3f}" if ic.notna().any() else "all NaN")
    passed, failed = passed + ic_reasonable_ok, failed + (not ic_reasonable_ok)

    # A perfect-agreement synthetic case: predictions == labels exactly -> IC == 1 every date.
    perfect_preds = df[["trade_date", "ticker", "label"]].rename(
        columns={"trade_date": "date", "label": "alpha"}
    )
    perfect_ic = rank_ic(perfect_preds, df)
    perfect_ok = np.allclose(perfect_ic.dropna(), 1.0)
    print_check("rank_ic is exactly 1.0 when predictions perfectly match the label", bool(perfect_ok))
    passed, failed = passed + perfect_ok, failed + (not perfect_ok)

    return passed, failed


def main():
    print_header("test_alpha")
    p1, f1 = test_label_close_dates()
    p2, f2 = test_purge_embargo_mask()
    p7, f7 = test_precomputed_close_dates_matches_recomputed_every_call()
    p3, f3 = test_monotone_constraint()
    p4, f4 = test_predict_empty_date()
    p5, f5 = test_walk_forward_and_rank_ic()
    p6, f6 = test_shrink_alpha()
    passed, failed = p1 + p2 + p3 + p4 + p5 + p6 + p7, f1 + f2 + f3 + f4 + f5 + f6 + f7
    print_section_end(passed, failed)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
