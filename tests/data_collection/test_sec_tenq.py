"""
test_sec_tenq.py
=================
Self-check for sec/tenq.py's pure logic (no network; synthetic
pandas.read_html-shaped tables reproducing REAL AAPL 2004-02-10 10-Q
structure, verified live against EDGAR 2026-08-01):

  - parse_period_header: the two-row "Three/Nine Months Ended" + dates header,
    including the colspan-duplication artifact pandas.read_html produces
    (a header cell's text repeated across every physical column its HTML
    colspan covers).
  - extract_statement: reconciles AAPL's real Q1 FY2004 figures (Net sales
    $2,006M/$1,472M current/prior, Cost of sales $1,470M/$1,066M, Net income
    $63M/$(8)M) -- including two real bugs found building this against live
    data (both fixed at their root in selected_financial_data.py, reused
    here): a whitespace-collapsed alias match ("Cost of  sales", embedded
    double space) and a 3-cell colspan-duplicated paren split
    (['(8', '(8', ')'] for a negative value, not the simpler 2-cell
    ['(25', ')'] shape selected_financial_data's own fix already covered).
  - find_statement_table: block-count-first, keyword-score-second, row-count-
    last scoring (same shape as selected_financial_data's Item 6 locator) --
    must prefer the real, fuller income statement over a smaller MD&A summary
    table that happens to state the same keywords more "cleanly".
  - build_cik_history: as-first-reported dedup, skips a filing whose HTML
    crashes pd.read_html or is the pre-HTML plain-.txt era (Feb 2001 AAPL
    shape) without losing the CIK's other good filings.

Usage: python tests/data_collection/test_sec_tenq.py
"""

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_collection.sec import tenq


def _real_income_table() -> pd.DataFrame:
    """Reproduces AAPL's real 2004-02-10 10-Q (CIK 320193, accession
    0001104659-04-003080) income statement table, verified live 2026-08-01 --
    including its two real HTML-rendering quirks: "Cost of  sales" (embedded
    double space) and "Net income  (loss)" whose FY2002 column renders as
    three cells ['63', '63', ..., '(8', '(8', ')'] (colspan-duplicated value,
    closing paren alone)."""
    return pd.DataFrame([
        [None, None, "Three Months Ended", "Three Months Ended", "Three Months Ended",
         "Three Months Ended", "Three Months Ended", None],
        [None, None, "December 27, 2003", "December 27, 2003", None,
         "December 28, 2002", "December 28, 2002", None],
        ["(In Millions)", None, None, None, None, None, None, None],
        ["Net sales", None, "$", "2006", None, "$", "1472", None],
        ["Cost of  sales", None, "1470", "1470", None, "1066", "1066", None],
        ["Net income  (loss)", None, "63", "63", None, "(8", "(8", ")"],
    ])


def test_parse_period_header_detects_blocks_with_colspan_duplication():
    blocks = tenq.parse_period_header(_real_income_table())
    assert blocks == [(3, pd.Timestamp("2003-12-27")), (3, pd.Timestamp("2002-12-28"))], (
        f"got {blocks}")
    print("OK: parse_period_header detects both period-blocks despite colspan-duplicated header cells")


def test_extract_statement_reconciles_real_aapl_figures():
    stmt = tenq.extract_statement(_real_income_table(), unit_multiplier=1_000_000.0)
    cur = stmt[(pd.Timestamp("2003-12-27"), 3)]
    prior = stmt[(pd.Timestamp("2002-12-28"), 3)]
    assert cur == {"net_revenue": 2_006_000_000.0, "cost_of_revenue": 1_470_000_000.0,
                   "net_income": 63_000_000.0}, cur
    assert prior["net_revenue"] == 1_472_000_000.0
    assert prior["cost_of_revenue"] == 1_066_000_000.0
    assert prior["net_income"] == -8_000_000.0, (
        f"the colspan-duplicated 3-cell paren split ['(8','(8',')'] must parse as a real "
        f"NET LOSS, got {prior['net_income']}")
    print("OK: extract_statement reconciles AAPL's real Q1 FY2004 figures, incl. the prior-year net loss")


def test_find_statement_table_prefers_the_real_statement_over_a_decoy_summary():
    # Real bug, confirmed live (2026-08-01): a smaller 7-row MD&A "Results of
    # Operations" summary table elsewhere in the SAME filing states the same
    # keywords with clean single-space text, while the real 43-row income
    # statement's "Cost of  sales" double-space label used to fail the
    # keyword match entirely -- letting the incomplete decoy (no net_income
    # at all) win the tie on keyword score. Fixed by whitespace-normalizing
    # before keyword matching; this pins that fix at the table-selection level.
    decoy = pd.DataFrame([
        [None, None, "Three Months Ended", "Three Months Ended", "Three Months Ended",
         "Three Months Ended", "Three Months Ended", None],
        [None, None, "December 27, 2003", "December 27, 2003", None,
         "December 28, 2002", "December 28, 2002", None],
        ["Net sales", None, "$", "2006", None, "$", "1472", None],
        ["Cost of sales", None, "1470", "1470", None, "1066", "1066", None],
    ])
    real = _real_income_table()
    picked = tenq.find_statement_table([decoy, real], tenq._INCOME_KEYWORDS)
    assert picked is real, "must prefer the real, fuller statement over the smaller decoy summary"
    print("OK: find_statement_table prefers the real statement over a decoy that only 'looks' cleaner")


def test_current_quarter_items_picks_max_end_date():
    three_mo = {
        (pd.Timestamp("2002-12-28"), 3): {"net_revenue": 1472.0},
        (pd.Timestamp("2003-12-27"), 3): {"net_revenue": 2006.0},
    }
    end, items = tenq._current_quarter_items(three_mo)
    assert end == pd.Timestamp("2003-12-27") and items["net_revenue"] == 2006.0
    print("OK: _current_quarter_items picks the most recent (non-comparative) period by date, not position")


def test_build_cik_history_end_to_end():
    filings = pd.DataFrame({
        "cik": [320193],
        "form_type": ["10-Q"],
        "date_filed": pd.to_datetime(["2004-02-10"]),
        "filename": ["a04-1622_110q.htm"],
    })
    fake_resp = mock.Mock(text="(In Millions)")
    with mock.patch.object(tenq.http, "get", return_value=fake_resp), \
         mock.patch.object(tenq.pd, "read_html", return_value=[_real_income_table()]):
        df = tenq.build_cik_history(320193, filings)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["end"] == pd.Timestamp("2003-12-27")
    assert row["period_months"] == 3
    assert row["net_revenue"] == 2_006_000_000.0
    assert row["net_income"] == 63_000_000.0
    assert row["flows_derived"] == 0, "a directly-printed 3-month figure is never 'derived'"
    assert row["flows_defined"] == 1
    assert "net_margin" in row and row["net_margin"] is not None, "ratios must be computed"
    print("OK: build_cik_history end-to-end reproduces AAPL's real current-quarter figures")


def test_build_cik_history_skips_a_filing_whose_html_crashes_pd_read_html():
    # Same "skip a filing that doesn't parse cleanly, try the next one" rule
    # as selected_financial_data.build_cik_history (real YUM bug: pd.read_html
    # raises more than ValueError on malformed real-world HTML) -- also
    # covers the pre-HTML plain-.txt era (AAPL's real 2001-02-12 10-Q has no
    # <table> at all, confirmed live 2026-08-01).
    filings = pd.DataFrame({
        "cik": [320193, 320193],
        "form_type": ["10-Q", "10-Q"],
        "date_filed": pd.to_datetime(["2001-02-12", "2004-02-10"]),
        "filename": ["a2038036z10-q.txt", "a04-1622_110q.htm"],
    })
    fake_resp = mock.Mock(text="(In Millions)")
    with mock.patch.object(tenq.http, "get", return_value=fake_resp), \
         mock.patch.object(tenq.pd, "read_html",
                            side_effect=[ValueError("No tables found"), [_real_income_table()]]):
        df = tenq.build_cik_history(320193, filings)
    assert len(df) == 1, "the crashing/tableless filing must be skipped, not lose the whole CIK"
    assert df.iloc[0]["end"] == pd.Timestamp("2003-12-27")
    print("OK: build_cik_history skips a filing with no parseable tables, keeps the CIK's other good data")


def test_build_cik_history_as_first_reported_dedup():
    # The SAME quarter reported twice (its own original filing, then again as
    # a prior-year comparative inside a later filing) must keep the EARLIEST
    # filing's figures, same as-first-reported rule as every other tier.
    later_table = _real_income_table()
    later_table.iloc[3, 3] = "9999"  # a restated/different net_revenue if this filing won
    filings = pd.DataFrame({
        "cik": [320193, 320193],
        "form_type": ["10-Q", "10-Q"],
        "date_filed": pd.to_datetime(["2004-02-10", "2005-02-01"]),
        "filename": ["original.htm", "later.htm"],
    })
    fake_resp = mock.Mock(text="(In Millions)")
    with mock.patch.object(tenq.http, "get", return_value=fake_resp), \
         mock.patch.object(tenq.pd, "read_html",
                            side_effect=[[_real_income_table()], [later_table]]):
        df = tenq.build_cik_history(320193, filings)
    row = df[df["end"] == pd.Timestamp("2003-12-27")].iloc[0]
    assert row["net_revenue"] == 2_006_000_000.0, (
        "the EARLIEST filing's figures must win, not a later filing's restated comparative")
    print("OK: build_cik_history keeps the earliest filing per quarter (as-first-reported)")


if __name__ == "__main__":
    test_parse_period_header_detects_blocks_with_colspan_duplication()
    test_extract_statement_reconciles_real_aapl_figures()
    test_find_statement_table_prefers_the_real_statement_over_a_decoy_summary()
    test_current_quarter_items_picks_max_end_date()
    test_build_cik_history_end_to_end()
    test_build_cik_history_skips_a_filing_whose_html_crashes_pd_read_html()
    test_build_cik_history_as_first_reported_dedup()
