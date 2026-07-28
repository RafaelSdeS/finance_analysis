"""
test_sec_item6.py
==================
Self-check for sec/item6.py's pure parsing logic (no network). Fixtures below
mirror real structural quirks confirmed against Intel's actual 10-K filings
(2026-07-28):

  - find_item6_table: must pick the real financial-summary table out of many
    candidates by scoring (year count + keyword hits), not the first
    ≥3-year table found.
  - extract_years: alias-collision bug -- "Diluted" (real diluted EPS) and
    "Weighted average diluted common shares outstanding" (share count) both
    contain the substring "diluted". Processing order used to let the WRONG
    row silently overwrite the right one. Exact-match-first fixes it.
  - Positional value extraction survives "$" placeholder cells and NaN
    spacer columns interleaved inconsistently between the label and the
    numbers, and parenthesized negatives.

Usage: python tests/data_collection/test_sec_item6.py
"""

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_collection.sec import item6


def test_find_item6_table_picks_best_scoring_candidate():
    # A small false-positive table (years present, no real financial keywords)
    # must lose to the real, fuller, keyword-matching table.
    noise = pd.DataFrame({0: ["Random label", "Another row"],
                          1: ["2005 2006 2007 2008 stray text", "x"]})
    real = pd.DataFrame({
        0: ["(In Millions)", "Net revenue", "Net income", "Total assets"],
        1: [None, "$", "$", "$"],
        2: ["2008", "35127", "4369", "53095"],
        3: [None, None, None, None],
        4: ["2007", "38334", "6976", "55664"],
        5: ["2006", "35382", "5044", "48372"],
        6: ["2005", "38826", "8664", "48309"],
    })
    best = item6.find_item6_table([noise, real])
    assert best is real, "must pick the table with real financial keywords, not just any 4-year table"
    print("OK: find_item6_table scores by keyword hits, not just year count")


def test_extract_years_reconciles_real_intel_figures():
    # Mirrors Intel's real 2010 10-K Item 6 table structure (values match exactly:
    # verified against live EDGAR 2026-07-28).
    table = pd.DataFrame({
        0: [None, "(In Millions, Except Per Share Amounts)", "Net revenue", "Net income",
            "Earnings per common share", "Basic", "Diluted",
            "Weighted average diluted common shares outstanding", "Total assets"],
        1: [None, None, None, None, None, None, None, None, None],
        2: [None, "2009", "$", "$", None, "$", "$", None, "$"],
        3: [None, "2009", "35127", "4369", None, "0.79", "0.77", "5645", "53095"],
        4: [None, "2008", "$", "$", None, "$", "$", None, "$"],
        5: [None, "2008", "37586", "5292", None, "0.93", "0.92", "5748", "50472"],
    })
    years = item6.extract_years(table)
    assert years["2009"]["net_revenue"] == 35127.0
    assert years["2009"]["net_income"] == 4369.0
    assert years["2009"]["eps_basic"] == 0.79
    # The real bug: eps_diluted must be 0.77 (the actual "Diluted" row), NOT
    # 5645 (the "Weighted average diluted common shares outstanding" row that
    # also contains the substring "diluted" and used to silently win).
    assert years["2009"]["eps_diluted"] == 0.77, (
        f"alias collision regressed: got {years['2009']['eps_diluted']}, expected diluted EPS 0.77, "
        f"not the diluted SHARE COUNT 5645")
    assert years["2008"]["eps_diluted"] == 0.92
    assert years["2009"]["total_assets"] == 53095.0
    print("OK: extract_years reconciles real Intel figures and resolves the diluted-EPS alias collision")


def test_parenthesized_negative():
    assert item6._parse_value("(1,234)") == -1234.0
    assert item6._parse_value("$1,234") == 1234.0
    assert item6._parse_value("$") is None
    assert item6._parse_value("NaN") is None
    print("OK: _parse_value handles parenthesized negatives, $ signs, commas, and blanks")


def test_build_cik_history_skips_filing_that_crashes_read_html():
    # Real bug, found retrying fundamentals collection at scale (2026-07-28):
    # pd.read_html raises more than ValueError on malformed real-world HTML.
    # Confirmed on YUM's real filing history: one filing crashed pandas'
    # internal TextParser with an IndexError (not a ValueError), which
    # propagated all the way up and discarded YUM's ENTIRE fundamentals build
    # -- including 611 perfectly good xbrl-tier concepts -- since nothing
    # downstream catches it per-CIK. A single bad filing must not take down
    # the whole company's history; the next (good) filing must still recover.
    filings = pd.DataFrame({
        "cik": [1, 1],
        "form_type": ["10-K", "10-K"],
        "date_filed": pd.to_datetime(["2003-03-01", "2004-03-01"]),
        "filename": ["bad.txt", "good.txt"],
    })
    fake_resp = mock.Mock(text="<html>irrelevant, read_html is mocked directly</html>")
    good_table = pd.DataFrame({0: ["Net revenue"], 1: ["2003"], 2: ["100"]})

    with mock.patch.object(item6.http, "get", return_value=fake_resp), \
         mock.patch.object(item6.pd, "read_html", side_effect=[IndexError("list index out of range"), [good_table]]), \
         mock.patch.object(item6, "find_item6_table", return_value=good_table), \
         mock.patch.object(item6, "extract_years", return_value={"2003": {"net_revenue": 100.0}}):
        df = item6.build_cik_history(1, filings)

    assert len(df) == 1, "must skip the crashing filing and still recover the next one, not lose the whole CIK"
    assert df.iloc[0]["fiscal_year"] == 2003
    print("OK: build_cik_history skips a filing whose HTML crashes pd.read_html, doesn't lose the whole CIK")


if __name__ == "__main__":
    test_find_item6_table_picks_best_scoring_candidate()
    test_extract_years_reconciles_real_intel_figures()
    test_parenthesized_negative()
    test_build_cik_history_skips_filing_that_crashes_read_html()
