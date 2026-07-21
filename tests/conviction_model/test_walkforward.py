"""
Test: conviction_model/walkforward.py's purge/embargo filter and expanding-
fold wiring (Phase 0 only -- warm-start/cold-restart cadence doesn't exist as
a concept until Phase 5 and isn't tested here; see
docs/conviction_model/CONVICTION_MODEL_PLAN.md's Testing strategy section).

Run from project root:
    python tests/conviction_model/test_walkforward.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.conviction_model.walkforward import (  # noqa: E402
    iter_purged_folds, purge_embargo_mask,
)
from src.h_series.spine import iter_expanding_folds  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402


def test_purge_embargo_drops_row_crossing_train_end(passed, failed):
    calendar = pd.bdate_range("2010-01-01", periods=1200)
    train_end = calendar[800]
    decision_dates = pd.DatetimeIndex([calendar[400]])  # label window: +504 -> calendar[904] > train_end
    mask = purge_embargo_mask(decision_dates, calendar, train_end, max_k=504)
    ok = bool(mask[0] == False)  # noqa: E712 -- explicit bool compare reads clearer than `not mask[0]` here
    print_check("purge_embargo_mask: drops a row whose 504-day label window crosses train_end",
                ok, f"mask={mask}")
    return passed + ok, failed + (not ok)


def test_purge_embargo_keeps_row_not_crossing_train_end(passed, failed):
    calendar = pd.bdate_range("2010-01-01", periods=1200)
    train_end = calendar[800]
    decision_dates = pd.DatetimeIndex([calendar[200]])  # label window: +504 -> calendar[704] <= train_end
    mask = purge_embargo_mask(decision_dates, calendar, train_end, max_k=504)
    ok = bool(mask[0] == True)  # noqa: E712
    print_check("purge_embargo_mask: keeps a row whose 504-day label window resolves at/before train_end",
                ok, f"mask={mask}")
    return passed + ok, failed + (not ok)


def test_purge_embargo_drops_row_past_calendar_end(passed, failed):
    calendar = pd.bdate_range("2010-01-01", periods=600)
    train_end = calendar[-1]  # even the loosest possible cutoff
    decision_dates = pd.DatetimeIndex([calendar[400]])  # +504 -> index 904, past the 600-day calendar -> NaT
    mask = purge_embargo_mask(decision_dates, calendar, train_end, max_k=504)
    ok = bool(mask[0] == False)  # noqa: E712
    print_check("purge_embargo_mask: drops a row whose label isn't computable at all (NaT), "
                "not a fabricated date", ok, f"mask={mask}")
    return passed + ok, failed + (not ok)


def _synthetic_labels_df():
    calendar = pd.bdate_range("2010-01-01", periods=1200)
    # Decision dates spread across the calendar, 2 tickers each. 840 is
    # deliberately included so at least one row lands INSIDE the first
    # fold's OOS window (train_end=800, oos_end~800+3mo~863) -- without it,
    # the OOS half of test_iter_purged_folds_train_oos_split would check its
    # conditions against an empty frame and pass vacuously.
    idxs = [100, 300, 500, 700, 840, 900, 1100]
    rows = []
    for i in idxs:
        for ticker in ("AAA", "BBB"):
            rows.append({"decision_date": calendar[i], "ticker": ticker, "risk_adj_excess_return_k21": 0.0})
    return pd.DataFrame(rows), calendar


def test_iter_purged_folds_boundaries_match_iter_expanding_folds(passed, failed):
    labels_df, calendar = _synthetic_labels_df()
    window_end, initial_train_end, step_months = calendar[-1], calendar[800], 3

    direct_folds = iter_expanding_folds(window_end, initial_train_end, step_months)
    wrapped = list(iter_purged_folds(labels_df, calendar, window_end, initial_train_end, step_months))
    wrapped_folds = [f for f, _, _ in wrapped]

    ok = len(wrapped_folds) == len(direct_folds) and all(
        a.train_end == b.train_end and a.oos_end == b.oos_end for a, b in zip(wrapped_folds, direct_folds)
    )
    print_check("iter_purged_folds: fold boundaries match iter_expanding_folds exactly (pure wiring)",
                ok, f"n_folds={len(wrapped_folds)}")
    return passed + ok, failed + (not ok)


def test_iter_purged_folds_train_oos_split(passed, failed):
    labels_df, calendar = _synthetic_labels_df()
    window_end, initial_train_end = calendar[-1], calendar[800]
    fold, train_df, oos_df = next(iter_purged_folds(labels_df, calendar, window_end, initial_train_end,
                                                      step_months=3, max_k=504))

    # fold.train_end == calendar[800]. Of the 7 decision dates, only idx100's label
    # window resolves in time (100+504=604 <= 800) -- idx300 (804 > 800), idx500
    # (1004 > 800), idx700/900/1100 (all NaT, past the 1200-day calendar) are all
    # purged; idx840 sits AFTER train_end entirely (belongs in oos_df, not train_df).
    train_tickers_dates = set(train_df["decision_date"])
    all_train_before_or_at_cutoff = all(d <= fold.train_end for d in train_tickers_dates)
    oos_all_after_train_end = bool((oos_df["decision_date"] > fold.train_end).all())
    oos_all_le_oos_end = bool((oos_df["decision_date"] <= fold.oos_end).all())

    ok = (len(oos_df) > 0 and all_train_before_or_at_cutoff
          and oos_all_after_train_end and oos_all_le_oos_end)
    print_check("iter_purged_folds: train_df is purge-safe, oos_df is exactly (train_end, oos_end]",
                ok, f"train dates={sorted(d.date() for d in train_tickers_dates)}, "
                    f"oos dates={sorted(d.date() for d in set(oos_df['decision_date']))}, "
                    f"fold=({fold.train_end.date()}, {fold.oos_end.date()}]")
    return passed + ok, failed + (not ok)


def main() -> int:
    print_header("conviction_model/walkforward.py (Phase 0)")
    passed = failed = 0
    for test_fn in [
        test_purge_embargo_drops_row_crossing_train_end,
        test_purge_embargo_keeps_row_not_crossing_train_end,
        test_purge_embargo_drops_row_past_calendar_end,
        test_iter_purged_folds_boundaries_match_iter_expanding_folds,
        test_iter_purged_folds_train_oos_split,
    ]:
        passed, failed = test_fn(passed, failed)
    print_section_end(passed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
