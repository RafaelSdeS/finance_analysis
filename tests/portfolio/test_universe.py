"""
test_universe.py -- smoke test for src/portfolio/universe.py on the REAL
dataset (not synthetic): confirms the thin wrapper's real-data behavior
matches what the underlying build_top50_universe.py helper already
guarantees (proposal §2.1 "Done when").

Needs data/processed/ml_dataset.parquet -- data group.
Run: python tests/portfolio/test_universe.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.build_dataset.paths import OUTPUT_PATH  # noqa: E402
from src.portfolio import universe  # noqa: E402
from tests.test_utils import print_check, print_header, print_section_end  # noqa: E402


def main():
    print_header("test_universe (real data)")
    passed = failed = 0

    df = pd.read_parquet(OUTPUT_PATH, columns=["ticker", "trade_date", "traded_amount"])
    membership = universe.liquid_universe(df, top_n=50)

    dates = universe.rebalance_dates(membership)
    ok = len(dates) > 50
    print_check("rebalance calendar has a sane number of periods", ok, f"got {len(dates)}")
    passed, failed = passed + ok, failed + (not ok)

    early, late = dates[3], dates[-4]
    early_members = universe.universe_at(membership, early)
    late_members = universe.universe_at(membership, late)
    churn_ok = early_members != late_members
    print_check(
        "membership churns over time (early period != late period)", churn_ok,
        f"{early.date()}: {len(early_members)} names, {late.date()}: {len(late_members)} names, "
        f"overlap={len(early_members & late_members)}",
    )
    passed, failed = passed + churn_ok, failed + (not churn_ok)

    # Every member's qualifying period start is a trailing-252-day snapshot,
    # so it must be at least 252 calendar-ish trading days after that
    # ticker's very first row in the dataset -- otherwise the ranking saw a
    # window it didn't have 252 real trading days for yet.
    first_trade = df.groupby("ticker")["trade_date"].min()
    m = membership.merge(first_trade.rename("first_trade"), on="ticker", how="left")
    trading_days_available = (
        df.sort_values("trade_date").groupby("ticker").cumcount() + 1
    )
    df_with_rank = df.assign(_rank=trading_days_available)
    rank_at_start = pd.merge_asof(
        m[["ticker", "start"]].sort_values("start"),
        df_with_rank[["ticker", "trade_date", "_rank"]].sort_values("trade_date"),
        left_on="start", right_on="trade_date", by="ticker", direction="backward",
    )
    min_rank = rank_at_start["_rank"].min()
    trailing_ok = bool(min_rank >= 252)
    print_check("every member has >= 252 trailing trading days as of its qualifying date",
                trailing_ok, f"min observed: {min_rank}")
    passed, failed = passed + trailing_ok, failed + (not trailing_ok)

    no_lookahead_ok = bool((m["start"] >= m["first_trade"]).all())
    print_check("no member's qualifying period starts before its first row in the dataset",
                no_lookahead_ok)
    passed, failed = passed + no_lookahead_ok, failed + (not no_lookahead_ok)

    print_section_end(passed, failed)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
