"""
test_sec_fundamentals.py
=========================
Self-check for sec/fundamentals.py's combiner logic (no network; mocks the
three per-tier fetchers).

Real bug, found scaling to ~250 companies (2026-07-28): Item 6's chained
rows are keyed by `fiscal_year` only (an int, e.g. 2006), so
build_company_fundamentals maps it to `end` = that year's Dec-31 -- a
simplification that assumes a calendar fiscal year. Confirmed on ADP (real
fiscal year end is June 30, not December 31): its real Aug-2006-filed 10-K
got labeled with `end`=2006-12-31, a date that HADN'T HAPPENED YET at filing
time -- `end > fundamentals_available_date`, the exact class of lookahead-
shaped artifact this whole pipeline exists to prevent. The fallback derives
an approximate fiscal year-end from the filing date instead of leaving that
impossible ordering in the data.

Usage: python tests/data_collection/test_sec_fundamentals.py
"""

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_collection.sec import fundamentals


def test_non_calendar_fiscal_year_end_does_not_precede_filing():
    # Mirrors ADP's real case: FY2006 10-K filed 2006-08-30 (ADP's actual fiscal
    # year ends June 30, not December 31).
    fake_item6 = pd.DataFrame({
        "fiscal_year": [2006],
        "fundamentals_available_date": pd.to_datetime(["2006-08-30"]),
        "net_income": [500_000_000.0],
        "cik": [8670],
    })
    with mock.patch.object(fundamentals.companyfacts, "fetch_companyfacts", return_value=None), \
         mock.patch.object(fundamentals.fds, "build_cik_history", return_value=pd.DataFrame()), \
         mock.patch.object(fundamentals.item6, "build_cik_history", return_value=fake_item6):
        df = fundamentals.build_company_fundamentals(8670, pd.DataFrame())

    assert len(df) == 1
    row = df.iloc[0]
    assert row["end"] <= row["fundamentals_available_date"], (
        f"end ({row['end']}) must never be after the filing that reported it "
        f"({row['fundamentals_available_date']})")
    assert str(row["end"].date()) == "2006-06-30", (
        "must fall back to the quarter-end ~2 months before filing, not the naive Dec-31")
    print("OK: non-calendar-fiscal-year Item 6 rows fall back instead of producing an impossible end date")


if __name__ == "__main__":
    test_non_calendar_fiscal_year_end_does_not_precede_filing()
