"""
test_sec_fundamentals.py
=========================
Self-check for sec/fundamentals.py's combiner logic (no network; mocks the
three per-tier fetchers).

Two real bugs, found scaling past ~250-465 companies (2026-07-28):

1. Item 6's chained rows are keyed by `fiscal_year` only (an int, e.g. 2006),
   so build_company_fundamentals maps it to `end` = that year's Dec-31 -- a
   simplification that assumes a calendar fiscal year. Confirmed on ADP (real
   fiscal year end is June 30): its Aug-2006-filed 10-K bundles BOTH FY2006
   (current) and FY2005 (comparative) in one Item 6 table. The first fix
   caught FY2006 (Dec-31-2006 postdates the Aug-2006 filing -- an impossible,
   lookahead-shaped ordering) but silently left FY2005 wrong: Dec-31-2005
   comfortably PRECEDES the filing date, so nothing flagged it, even though
   ADP's real FY2005 end is 2005-06-30, six months off. Fixed by deriving the
   company's true fiscal quarter-end ONCE (from whichever row proves Dec-31
   impossible) and applying it to every row for that CIK -- a company's
   fiscal year-end doesn't change year to year.
2. Combining tiers deduped on EXACT `end` equality. Item6's Dec-31-rounded
   guess and xbrl/ex27's real fiscal-calendar dates (e.g. "2007-09-29") can
   describe the SAME real period a few days apart, so exact-equality dedup
   let both survive as separate rows -- confirmed on AAPL, INTC, JNJ, MAR,
   CSX and 35 others (40 pairs across 465 companies already collected).
   Fixed by clustering `end` across tiers with the same tolerance already
   used intra-tier in companyfacts.py, before applying tier priority.

A THIRD bug was found auditing a full top-500 collection run against fix #1
above (2026-07-28): the derivation itself ("filing_date - 2 months, then
round UP to the CONTAINING calendar quarter") rounds forward past the filing
date whenever the filing lands less than ~2 months into its own quarter --
confirmed on CRM (filed 2005-03-25 -> "-2mo" gives 2005-01-25 -> rounds UP to
Q1's end, 2005-03-31 -- 6 days AFTER the filing), and on NTAP/LRCX/ADSK.
Fixed by deriving the latest quarter-end STRICTLY BEFORE the filing date
(safe by construction) plus a year offset (companies like CRM/NTAP, whose
fiscal year-end falls in Jan/Apr, have their nearest safe quarter-end in the
calendar year BEFORE the fiscal_year label -- reusing bare month/day against
each row's own fiscal_year, as fix #1 did, produced a different impossible
date for these).

A FOURTH issue, found auditing the SAME run's XBRL-tier rows (2026-07-28):
WMT has a real CashAndCashEquivalentsAtCarryingValue fact tagged
end=2012-12-31 (not even one of WMT's real Jan/Apr/Jul/Oct fiscal quarter-
ends) filed 2012-03-27 -- nine months before the period it claims to
describe. This is a genuine upstream XBRL tagging error in the source data,
not a derivation bug our code could "fix" correctly (there's no right answer
to derive). build_company_fundamentals now enforces the invariant itself as
a final defensive filter, dropping any row that still violates it regardless
of which tier or root cause produced it -- the one choke point all three
tiers converge through.

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


def test_non_calendar_fiscal_year_end_fixes_every_row_not_just_the_flagged_one():
    # Mirrors ADP's ACTUAL bundled filing: one 2006-08-30 10-K reports BOTH
    # FY2006 (current, trips the impossible check) and FY2005 (comparative,
    # does NOT trip it -- Dec-31-2005 already precedes the filing date).
    fake_item6 = pd.DataFrame({
        "fiscal_year": [2005, 2006],
        "fundamentals_available_date": pd.to_datetime(["2006-08-30", "2006-08-30"]),
        "net_income": [400_000_000.0, 500_000_000.0],
        "cik": [8670, 8670],
    })
    with mock.patch.object(fundamentals.companyfacts, "fetch_companyfacts", return_value=None), \
         mock.patch.object(fundamentals.fds, "build_cik_history", return_value=pd.DataFrame()), \
         mock.patch.object(fundamentals.item6, "build_cik_history", return_value=fake_item6):
        df = fundamentals.build_company_fundamentals(8670, pd.DataFrame())

    ends = sorted(str(e.date()) for e in df["end"])
    assert ends == ["2005-06-30", "2006-06-30"], (
        f"BOTH rows must use the company's real June fiscal year-end, got {ends} "
        f"-- a per-row-only fix leaves the untrigged comparative year wrong")
    print("OK: the derived fiscal year-end is applied to every row for the CIK, not just the flagged one")


def test_short_filing_lag_does_not_round_derived_end_past_filing_date():
    # Mirrors CRM's real case: fiscal year labeled by its ENDING calendar year
    # (real FYE ~Jan 31), filed only ~2 months later. The "-2mo, round UP to
    # containing quarter" derivation (fix #1) computed 2005-01-25 -> Q1 2005's
    # end (2005-03-31) -- 6 days AFTER the real 2005-03-25 filing that
    # reported it. Also exercises the year-offset: the safe quarter-end
    # (2004-12-31) falls in the CALENDAR YEAR BEFORE the fiscal_year label.
    fake_item6 = pd.DataFrame({
        "fiscal_year": [2005, 2006],
        "fundamentals_available_date": pd.to_datetime(["2005-03-25", "2006-03-15"]),
        "net_income": [100_000_000.0, 120_000_000.0],
        "cik": [1108524, 1108524],
    })
    with mock.patch.object(fundamentals.companyfacts, "fetch_companyfacts", return_value=None), \
         mock.patch.object(fundamentals.fds, "build_cik_history", return_value=pd.DataFrame()), \
         mock.patch.object(fundamentals.item6, "build_cik_history", return_value=fake_item6):
        df = fundamentals.build_company_fundamentals(1108524, pd.DataFrame())

    for _, row in df.iterrows():
        assert row["end"] <= row["fundamentals_available_date"], (
            f"end ({row['end']}) must never be after its own filing "
            f"({row['fundamentals_available_date']})")
    ends = sorted(str(e.date()) for e in df["end"])
    assert ends == ["2004-12-31", "2005-12-31"], (
        f"expected the year-before quarter-end for a Jan-FYE company, got {ends}")
    print("OK: a short filing lag doesn't round the derived end past its own filing date")


def test_tier_boundary_near_duplicate_end_dates_are_deduped():
    # Mirrors the real AAPL case: xbrl's FY2007 Q4 ends 2007-09-29 (AAPL's actual
    # fiscal-calendar convention), item6's naive guess for the same real period
    # is 2007-09-30 -- one day apart, describing the SAME period twice.
    fake_facts = {"placeholder": True}
    xbrl_row = pd.DataFrame({
        "end": pd.to_datetime(["2007-09-29"]),
        "fundamentals_available_date": pd.to_datetime(["2007-11-15"]),
        "net_income": [1_000_000_000.0],
    })
    item6_row = pd.DataFrame({
        "fiscal_year": [2007],
        "fundamentals_available_date": pd.to_datetime(["2007-11-16"]),  # ~2mo after AAPL's real Sept FYE
        "net_income": [999_000_000.0],
        "cik": [320193],
    })
    with mock.patch.object(fundamentals.companyfacts, "fetch_companyfacts", return_value=fake_facts), \
         mock.patch.object(fundamentals.companyfacts, "extract_line_items", return_value=xbrl_row), \
         mock.patch.object(fundamentals.companyfacts, "compute_us_ratios", side_effect=lambda df: df), \
         mock.patch.object(fundamentals.fds, "build_cik_history", return_value=pd.DataFrame()), \
         mock.patch.object(fundamentals.item6, "build_cik_history", return_value=item6_row):
        df = fundamentals.build_company_fundamentals(320193, pd.DataFrame())

    assert len(df) == 1, (
        f"one real period reported by two tiers a day apart must collapse to ONE row, got {len(df)}")
    assert df.iloc[0]["fundamentals_tier"] == "xbrl", "higher-priority tier (xbrl) must win"
    assert df.iloc[0]["net_income"] == 1_000_000_000.0
    print("OK: near-duplicate 'end' dates across tiers are clustered and deduped by tier priority")


def test_source_data_anomaly_is_dropped_not_left_in():
    # Mirrors WMT's real XBRL fact: CashAndCashEquivalentsAtCarryingValue tagged
    # end=2012-12-31, filed 2012-03-27 -- nine months before the period it
    # claims to describe. A genuine upstream tagging error (not even one of
    # WMT's real fiscal quarter-ends), not something any derivation logic could
    # correctly "fix". Also includes one good row to confirm the filter only
    # drops the actual violator, not the whole company's data.
    xbrl_rows = pd.DataFrame({
        "end": pd.to_datetime(["2012-12-31", "2012-10-31"]),
        "fundamentals_available_date": pd.to_datetime(["2012-03-27", "2012-12-04"]),
        "cash": [6_600_000_000.0, 8_643_000_000.0],
    })
    with mock.patch.object(fundamentals.companyfacts, "fetch_companyfacts", return_value={"placeholder": True}), \
         mock.patch.object(fundamentals.companyfacts, "extract_line_items", return_value=xbrl_rows), \
         mock.patch.object(fundamentals.companyfacts, "compute_us_ratios", side_effect=lambda df: df), \
         mock.patch.object(fundamentals.fds, "build_cik_history", return_value=pd.DataFrame()), \
         mock.patch.object(fundamentals.item6, "build_cik_history", return_value=pd.DataFrame()):
        df = fundamentals.build_company_fundamentals(104169, pd.DataFrame())

    assert len(df) == 1, f"the anomalous row must be dropped, the good one kept, got {len(df)} rows"
    assert str(df.iloc[0]["end"].date()) == "2012-10-31", "must keep the good row, not the anomalous one"
    print("OK: a source-data tagging anomaly (end > filed) is dropped, not left in the output")


if __name__ == "__main__":
    test_non_calendar_fiscal_year_end_does_not_precede_filing()
    test_non_calendar_fiscal_year_end_fixes_every_row_not_just_the_flagged_one()
    test_short_filing_lag_does_not_round_derived_end_past_filing_date()
    test_tier_boundary_near_duplicate_end_dates_are_deduped()
    test_source_data_anomaly_is_dropped_not_left_in()
