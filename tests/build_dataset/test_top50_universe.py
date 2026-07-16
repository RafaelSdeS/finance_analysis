"""
Test: build_top50_universe.py's point-in-time top-N ranking.

Synthetic 4-ticker panel (monthly rebalance, 5-day trailing window):
  A - always high volume                    -> should qualify every period
  B - high volume in month 1, then delists   -> qualifies only its last period
                                                 before the gap exceeds tolerance
  C - low volume throughout                  -> qualifies only once A/B/E don't
  E - near-zero volume, huge spike in month 3 -> must NOT qualify before the
                                                 spike is inside its trailing
                                                 window (no-lookahead check)

Run from project root:
    python tests/build_dataset/test_top50_universe.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.build_dataset.build_top50_universe import (  # noqa: E402
    build_top50_membership,
    filter_to_top50_universe,
    zero_fill_missing_fundamentals,
)
from test_utils import print_check, print_header, print_section_end  # noqa: E402


def _series(ticker, dates, traded_amount):
    return pd.DataFrame({
        "ticker": ticker,
        "trade_date": pd.to_datetime(dates),
        "traded_amount": traded_amount,
        "note": "x",
    })


def build_synthetic_df():
    jan = pd.bdate_range("2020-01-01", "2020-01-31")
    feb = pd.bdate_range("2020-02-01", "2020-02-29")
    mar = pd.bdate_range("2020-03-01", "2020-03-31")
    full = jan.union(feb).union(mar)

    a = _series("A", full, 1_000_000)
    b = _series("B", jan, 900_000)  # delists after January
    c = _series("C", full, 10)
    e = _series("E", full, [1] * (len(jan) + len(feb)) + [2_000_000] * len(mar))
    return pd.concat([a, b, c, e], ignore_index=True)


def main():
    print_header("test_top50_universe")
    passed = failed = 0

    df = build_synthetic_df()
    membership = build_top50_membership(
        df, top_n=2, trailing_days=5, rebalance_freq="M", tolerance_days=10
    )
    periods = membership[["period_id", "start", "end"]].drop_duplicates().sort_values("start").reset_index(drop=True)

    def members_of(period_id):
        return set(membership.loc[membership["period_id"] == period_id, "ticker"])

    ok = len(periods) == 3
    print_check("3 monthly rebalance periods produced", ok, f"got {len(periods)}")
    passed, failed = passed + ok, failed + (not ok)

    ok = members_of(0) == {"A", "B"}
    print_check("period 0 (Jan-end): A, B qualify", ok, f"got {members_of(0)}")
    passed, failed = passed + ok, failed + (not ok)

    ok = members_of(1) == {"A", "C"}
    print_check("period 1 (Feb-end): B dropped (delisted past tolerance), C fills in", ok, f"got {members_of(1)}")
    passed, failed = passed + ok, failed + (not ok)

    ok = members_of(2) == {"A", "E"}
    print_check("period 2 (Mar-end): E's spike now inside trailing window, C drops", ok, f"got {members_of(2)}")
    passed, failed = passed + ok, failed + (not ok)

    # No-lookahead: E must not appear before its own spike is inside the window.
    ok = "E" not in members_of(0) and "E" not in members_of(1)
    print_check("no-lookahead: E excluded from periods before its March spike", ok)
    passed, failed = passed + ok, failed + (not ok)

    universe_df = filter_to_top50_universe(df, membership)

    ok = "note" in universe_df.columns and set(universe_df.columns) == set(df.columns)
    print_check("all original columns preserved in filtered output", ok)
    passed, failed = passed + ok, failed + (not ok)

    p0, p1, p2 = periods.iloc[0], periods.iloc[1], periods.iloc[2]

    def expected_rows(ticker, start, end):
        rows = df[(df["ticker"] == ticker) & (df["trade_date"] >= start) & (df["trade_date"] < end)]
        return len(rows)

    expected_b = expected_rows("B", p0["start"], p1["start"])
    got_b = len(universe_df[universe_df["ticker"] == "B"])
    ok = got_b == expected_b and expected_b > 0
    print_check("B's rows locked to exactly its qualifying period", ok, f"expected {expected_b}, got {got_b}")
    passed, failed = passed + ok, failed + (not ok)

    got_b_after = len(universe_df[(universe_df["ticker"] == "B") & (universe_df["trade_date"] >= p1["start"])])
    ok = got_b_after == 0
    print_check("B has zero rows after delisting (union recovers it, doesn't leak forward)", ok)
    passed, failed = passed + ok, failed + (not ok)

    expected_e = expected_rows("E", p2["start"], p2["end"])
    got_e = len(universe_df[universe_df["ticker"] == "E"])
    ok = got_e == expected_e and expected_e > 0
    print_check("E's rows locked to exactly its qualifying (post-spike) period", ok, f"expected {expected_e}, got {got_e}")
    passed, failed = passed + ok, failed + (not ok)

    # --- zero_fill_missing_fundamentals: no NaN left on has_fundamentals==0
    # rows, including the sibling columns (sector z-scores, f_score flags,
    # trend cols) that the original list missed
    # (docs/TOP50_UNIVERSE_ML_READINESS_AUDIT.md §1.3) ---
    zf = pd.DataFrame({
        "has_fundamentals": [0, 1],
        "roe": [None, 0.15],
        "market_cap": [None, 1_000_000.0],
        "pl_zscore_sector": [None, 0.5],
        "f_score": [None, 3],
        "roe_trend_4q": [None, 0.02],
    })
    zf = zero_fill_missing_fundamentals(zf)

    checked_cols = ["roe", "market_cap", "pl_zscore_sector", "f_score", "roe_trend_4q"]
    ok = all(zf.loc[0, c] == 0 for c in checked_cols)
    print_check("zero_fill_missing_fundamentals: sibling columns filled on has_fundamentals==0",
                ok, f"row 0: {zf.loc[0, checked_cols].to_dict()}")
    passed, failed = passed + ok, failed + (not ok)

    ok = all(zf.loc[1, c] == v for c, v in
             [("roe", 0.15), ("market_cap", 1_000_000.0), ("pl_zscore_sector", 0.5),
              ("f_score", 3), ("roe_trend_4q", 0.02)])
    print_check("zero_fill_missing_fundamentals: has_fundamentals==1 row left untouched", ok)
    passed, failed = passed + ok, failed + (not ok)

    print_section_end(passed, failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
