"""
test_sec_selected_financial_data.py
====================================
Self-check for sec/selected_financial_data.py's pure parsing logic (no
network) -- the "Item 6" gap tier, see that module's docstring for what
SEC Item 6 "Selected Financial Data" actually is. Fixtures below
mirror real structural quirks confirmed against Intel's actual 10-K filings
(2026-07-28):

  - find_selected_financial_data_table: must pick the real financial-summary table out of many
    candidates by scoring (year count + keyword hits), not the first
    ≥3-year table found.
  - extract_years: alias-collision bug -- "Diluted" (real diluted EPS) and
    "Weighted average diluted common shares outstanding" (share count) both
    contain the substring "diluted". Processing order used to let the WRONG
    row silently overwrite the right one. Exact-match-first fixes it.
  - Positional value extraction survives "$" placeholder cells and NaN
    spacer columns interleaved inconsistently between the label and the
    numbers, and parenthesized negatives.
  - detect_unit_multiplier/extract_years(unit_multiplier=...): Item 6's dollar
    figures are printed under a "(in millions)"-style caption that lives
    OUTSIDE the parsed table -- confirmed against adjacent tiers for the SAME
    company (2026-07-28): unscaled INTC item6 net_revenue read ~1e6 too small
    vs INTC's own xbrl tier. Per-share rows (EPS, dividends/share) must NOT be
    rescaled even when every other row in the same table is.
  - find_selected_financial_data_table must rank by YEAR COUNT first, keyword score second (2026-07-30,
    ZION): a business-segment income-statement fragment can spell out "Total
    assets" verbatim and outscore the real, fuller company-wide table, which
    sometimes just says "Assets".
  - _row_values must collapse colspan-duplicated adjacent cells BEFORE the
    footnote-marker check (2026-07-30, BOOM): stacking both anomalies in one
    row defeats the footnote-only guard, which requires an EXACT token-count
    match to fire.
  - build_cik_history's unit_multiplier must come from the WINNING TABLE's own
    caption, not a whole-document scan (2026-07-30, ZION): a huge combined
    submission's dominant caption can belong to an unrelated, larger table.

Usage: python tests/data_collection/test_sec_selected_financial_data.py
"""

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_collection.sec import selected_financial_data as sfd


def test_find_selected_financial_data_table_picks_best_scoring_candidate():
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
    best = sfd.find_selected_financial_data_table([noise, real])
    assert best is real, "must pick the table with real financial keywords, not just any 4-year table"
    print("OK: find_selected_financial_data_table scores by keyword hits, not just year count")


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
    years = sfd.extract_years(table)
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
    assert sfd._parse_value("(1,234)") == -1234.0
    assert sfd._parse_value("$1,234") == 1234.0
    assert sfd._parse_value("$") is None
    assert sfd._parse_value("NaN") is None
    print("OK: _parse_value handles parenthesized negatives, $ signs, commas, and blanks")


def test_row_values_strips_footnote_marker_not_real_negative():
    # Real bug, confirmed on ORCL's actual 2006 10-K (2026-07-30): "Total
    # assets" carries a "(3)" footnote-reference cell for 2006 and 2005 only
    # (not 2004-2002), shaped identically to a real parenthesized negative
    # under _parse_value. Left in, it doesn't just corrupt one cell -- it
    # shifts every LATER year's real value one position early. Shape below
    # mirrors the real row (5 years, markers on the first two).
    row = pd.Series(["Total assets", None, None, 29029, "(3)", None, None,
                      20687, "(3)", None, None, 12763, None, None, 10967,
                      None, None, 10800])
    vals = sfd._row_values(row.iloc[1:], 5)
    assert vals == [29029.0, 20687.0, 12763.0, 10967.0, 10800.0], (
        f"footnote markers must be stripped and real values kept aligned to their year, got {vals}")
    print("OK: _row_values strips excess footnote-marker tokens without disturbing alignment")


def test_row_values_keeps_genuine_small_negative_when_already_aligned():
    # The marker-stripping fix above must never fire on a row that's already
    # correctly aligned (token count == n_years) -- a real small negative
    # dollar figure (e.g. a loss year) must survive untouched.
    row = pd.Series(["Net income", None, 100, None, -3, None, 50])
    vals = sfd._row_values(row.iloc[1:], 3)
    assert vals == [100.0, -3.0, 50.0], (
        f"a genuine negative value in an already-aligned row must not be stripped, got {vals}")
    print("OK: _row_values leaves a genuine small negative alone when the row is already aligned")


def test_row_values_merges_paren_split_across_cells():
    # Real bug, confirmed on AAPL's actual 2005 10-K (2026-07-30): the "Net
    # income (loss)" row's FY2001 column renders as two separate table cells,
    # "(25" and ")", not one "(25)" cell -- pandas.read_html splits them
    # because of how that row's HTML is padded. _parse_value only recognizes
    # a negative when both parens are in the SAME cell, so "(25" alone parsed
    # as a positive 25 -- Apple's real $25M FY2001 net LOSS was silently
    # stored as a $25M profit. Row shape below is the real one (verified
    # against the live filing text).
    row = pd.Series(["Net income (loss)", "$", 1335, "$", 276, "$", 69, "$", 65, "$", "(25", ")"])
    vals = sfd._row_values(row.iloc[1:], 5)
    assert vals == [1335.0, 276.0, 69.0, 65.0, -25.0], (
        f"a paren split across two cells must still parse as one negative value, got {vals}")
    print("OK: _row_values merges a parenthesized negative split across two adjacent cells")


def test_detect_unit_multiplier_prefer_first_breaks_ties_deterministically():
    # Real bug, confirmed on AAPL's actual 2005 10-K (2026-07-30): the Item 6
    # table states its governing caption once, "(In millions, except share
    # and per share amounts)", then separately captions its shares-outstanding
    # sub-row "(in thousands)" -- a 1-vs-1 tie under the old mode-based
    # `max(set(hits), key=hits.count)` selection, whose outcome depends on
    # set() iteration order (hash-seed dependent, not even deterministic
    # across runs). Confirmed live: AAPL's FY2001-2005 net_revenue stored
    # 1000x too small. prefer_first=True must pick whichever unit word
    # appears FIRST in the text, regardless of tie or set ordering -- proven
    # here both ways, not just "always millions wins".
    millions_first = "(In millions, except share amounts) ... shares (in thousands): ..."
    thousands_first = "(In thousands) ... shares (in millions): ..."
    assert sfd.detect_unit_multiplier(millions_first, prefer_first=True) == 1_000_000.0
    assert sfd.detect_unit_multiplier(thousands_first, prefer_first=True) == 1_000.0
    print("OK: detect_unit_multiplier(prefer_first=True) breaks a unit-caption tie by first occurrence, not set order")


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

    with mock.patch.object(sfd.http, "get", return_value=fake_resp), \
         mock.patch.object(sfd.pd, "read_html", side_effect=[IndexError("list index out of range"), [good_table]]), \
         mock.patch.object(sfd, "find_selected_financial_data_table", return_value=good_table), \
         mock.patch.object(sfd, "extract_years", return_value={"2003": {"net_revenue": 100.0}}):
        df = sfd.build_cik_history(1, filings)

    assert len(df) == 1, "must skip the crashing filing and still recover the next one, not lose the whole CIK"
    assert df.iloc[0]["fiscal_year"] == 2003
    print("OK: build_cik_history skips a filing whose HTML crashes pd.read_html, doesn't lose the whole CIK")


def test_extract_years_scales_dollar_rows_but_not_per_share():
    # Same real Intel table shape as above, exercising unit_multiplier: dollar
    # rows (net_revenue/net_income/total_assets) must scale by the caption's
    # implied multiplier; EPS rows must NOT (per-share figures are never
    # expressed in the caption's units -- confirmed on the same real table,
    # "0.79"/"0.77" sit next to "35127"/"4369" under one "(In Millions)" caption).
    table = pd.DataFrame({
        0: [None, "(In Millions, Except Per Share Amounts)", "Net revenue", "Net income",
            "Earnings per common share", "Basic", "Diluted", "Total assets"],
        1: [None, None, None, None, None, None, None, None],
        2: [None, "2009", "$", "$", None, "$", "$", "$"],
        3: [None, "2009", "35127", "4369", None, "0.79", "0.77", "53095"],
    })
    years = sfd.extract_years(table, unit_multiplier=1_000_000.0)
    assert years["2009"]["net_revenue"] == 35_127_000_000.0
    assert years["2009"]["net_income"] == 4_369_000_000.0
    assert years["2009"]["total_assets"] == 53_095_000_000.0
    assert years["2009"]["eps_basic"] == 0.79, "EPS must stay a per-share dollar figure, not get scaled by 1e6"
    assert years["2009"]["eps_diluted"] == 0.77
    print("OK: extract_years scales dollar rows by unit_multiplier but leaves per-share rows alone")


def test_find_selected_financial_data_table_rejects_quarterly_and_embedded_digit_false_positives():
    # Real bug, found auditing the collected dataset (2026-07-29): AAPL got a
    # fiscal_year=1909 row because find_selected_financial_data_table picked its Selected
    # Quarterly Financial Data table -- quarterly net sales figures ($2,014M /
    # $1,909M / $2,006M) are themselves 4-digit, year-shaped numbers, and the
    # old code flattened df.head(3) together instead of checking one row at a
    # time. AMG hit a second variant: a stock-comp footnote table where large
    # bare numbers ("119069", "22054.0") contain embedded year-shaped
    # substrings ("1906", "2054") an unanchored regex still matched. Shapes
    # below mirror the real filings (values reconciled against live EDGAR).
    quarterly = pd.DataFrame({
        0: ["2004", "Net sales", "Net income"],
        1: [None, "$", "$"],
        2: [None, 2350.0, 106.0],
        3: [None, "$", "$"],
        4: [None, 2014.0, 61.0],
        5: [None, "$", "$"],
        6: [None, 1909.0, 46.0],
        7: [None, "$", "$"],
        8: [None, "2006", "63"],
    })
    footnote = pd.DataFrame({
        0: ["Net Income—as reported",
            "Less: comp expense, net of tax",
            "Less: comp expense related to 2003 Amendment, net of tax"],
        1: [None, None, None],
        2: ["$", None, None],
        3: [60528.0, 10614.0, 22054.0],
        4: ["$", None, None],
        5: ["77147", "14326", "—"],
        6: ["$", None, None],
        7: ["119069", "709", "—"],
    })
    real = pd.DataFrame({
        0: ["(In Millions)", "Net revenue", "Net income", "Total assets"],
        1: [None, "$", "$", "$"],
        2: ["2008", "35127", "4369", "53095"],
        3: [None, None, None, None],
        4: ["2007", "38334", "6976", "55664"],
        5: ["2006", "35382", "5044", "48372"],
        6: ["2005", "38826", "8664", "48309"],
    })
    assert sfd._year_header_row(quarterly) == ["2004"], (
        "quarterly-earnings table's data rows (all carrying '$') must not "
        "contribute their dollar figures as extra 'years'")
    assert sfd._year_header_row(footnote) == ["2003"], (
        "footnote table must not yield '1906'/'2054' embedded-digit false years")
    best = sfd.find_selected_financial_data_table([quarterly, footnote, real])
    assert best is real, "must pick the real Item 6 table over both false-positive shapes"
    print("OK: find_selected_financial_data_table rejects quarterly-data and embedded-digit false-positive tables")


def test_find_selected_financial_data_table_prefers_more_years_over_keyword_count():
    # Real bug, confirmed on ZION's actual 2005 10-K (2026-07-30): Item 6
    # there is incorporated by reference to an exhibit, so no real table
    # exists in the parsed document -- but a business SEGMENT's condensed
    # income statement buried in the MD&A spells out "Total assets"/"Net
    # income (loss)"/"Total revenue" verbatim (score 3, only 3 years: its own
    # segment history), while the REAL 5-year company-wide table just says
    # "Assets" under an "AT YEAR-END" header (score 2, since "Assets" alone
    # doesn't match "TOTAL ASSETS"). The old (score, years, rows) ordering let
    # the keyword-richer fragment beat the real, fuller table.
    segment_fragment = pd.DataFrame({
        0: ["CONDENSED INCOME STATEMENT", "Total revenue", "Net income (loss)", "Total assets"],
        1: ["2004", "3.5", "-21.8", "-572.0"],
        2: ["2003", "82.1", "10.4", "1120.0"],
        3: ["2002", "0.4", "-15.2", "900.0"],
    })
    real_table = pd.DataFrame({
        0: ["FOR THE YEAR", "Total revenue", "Net income", "Assets"],
        1: ["2004", "3500", "191.8", "39958"],
        2: ["2003", "3900", "-21.2", "42115"],
        3: ["2002", "3200", "-21.8", "40200"],
        4: ["2001", "1585.6", "337.8", "38500"],
        5: ["2000", "1411.9", "256.3", "37000"],
    })
    best = sfd.find_selected_financial_data_table([segment_fragment, real_table])
    assert best is real_table, "the real 5-year table must win over a keyword-richer 3-year segment fragment"
    print("OK: find_selected_financial_data_table ranks by year count first, keyword score second")


def test_row_values_collapses_colspan_duplicated_cells_before_footnote_check():
    # Real bug, confirmed on BOOM's (Dynamic Materials) actual 2005 10-K
    # (2026-07-30): the "Total assets" row's HTML renders each year's value as
    # TWO identical adjacent cells (a colspan-to-columns artifact unique to
    # this row), stacked with one genuine footnote-marker cell after the 2003
    # value. Stripping only the marker left 10 tokens for n_years=5 -- not an
    # exact match -- so the existing guard silently gave up and returned the
    # first 5 RAW (still-duplicated) tokens, corrupting every year one
    # position off. Real values reconciled against the live EDGAR filing:
    # 2004=43752521, 2003=35261408, 2002=33697992, 2001=36913345, 2000=35406455.
    row = pd.Series([
        43752521, 43752521, 35261408, 35261408, "(1)",
        33697992, 33697992, 36913345, 36913345, 35406455, 35406455,
    ])
    vals = sfd._row_values(row, 5)
    assert vals == [43752521.0, 35261408.0, 33697992.0, 36913345.0, 35406455.0], (
        f"colspan-duplicated cells must collapse before the footnote-marker check, got {vals}")
    print("OK: _row_values collapses colspan-duplicated cell pairs before the footnote-marker check")


def test_build_cik_history_uses_winning_tables_own_unit_caption():
    # Real bug, confirmed on ZION's actual 2005 10-K (2026-07-30): the winning
    # table's own caption said "(Amounts in millions)", but detect_unit_multiplier
    # was run over the WHOLE filing document, whose dominant caption (from the
    # much larger main financial statements elsewhere in the same combined
    # submission) was "thousands" -- silently rescaling the winning table's
    # figures 1000x too small. The multiplier must come from the winning
    # table's own text first, falling back to the whole document only if the
    # table doesn't state its own units.
    filings = pd.DataFrame({
        "cik": [7],
        "form_type": ["10-K"],
        "date_filed": pd.to_datetime(["2005-03-01"]),
        "filename": ["zion.txt"],
    })
    doc_text = "figures in thousands " * 5 + "(Amounts in millions)"
    fake_resp = mock.Mock(text=doc_text)
    table = pd.DataFrame({
        0: ["(Amounts in millions)", "Net revenue", "Net income", "Total assets"],
        1: [None, None, None, None],
        2: ["2004", "3.5", "0.3", "43.8"],
    })
    with mock.patch.object(sfd.http, "get", return_value=fake_resp), \
         mock.patch.object(sfd.pd, "read_html", return_value=[table]), \
         mock.patch.object(sfd, "find_selected_financial_data_table", return_value=table):
        df = sfd.build_cik_history(7, filings)

    assert len(df) == 1
    assert df.iloc[0]["net_revenue"] == 3_500_000.0, (
        "must scale by the WINNING TABLE's own 'millions' caption, not the document-wide 'thousands' mention")
    print("OK: build_cik_history scales by the winning table's own unit caption, not the whole document's")


def test_build_cik_history_ignores_shares_row_caption_falls_back_to_document():
    # Real bug, confirmed on TXN's actual 2006 10-K (2026-07-30): the winning
    # table states NO table-wide dollar caption in its own cells at all (its
    # real "(in millions)" caption lives in a preceding paragraph outside the
    # parsed table) -- its ONLY units mention is a share-count row's own local
    # caption, "...shares outstanding ... in thousands". The bug-14 fix above
    # (prefer the winning table's own caption) then wrongly read THAT as the
    # table's governing caption, applying x1000 to net_revenue/net_income too
    # and understating both 1000x (TXN's real 2005 net_revenue $13.392B was
    # stored as $13.392M). A row whose label mentions "shares" must be
    # excluded from caption detection so this table correctly falls through
    # to the whole-document scan (dominated by "millions" elsewhere in TI's
    # combined annual-report exhibit, same as any other filing's fallback).
    filings = pd.DataFrame({
        "cik": [10],
        "form_type": ["10-K"],
        "date_filed": pd.to_datetime(["2006-02-28"]),
        "filename": ["txn.txt"],
    })
    doc_text = "reported in millions " * 5 + "elsewhere in thousands"
    fake_resp = mock.Mock(text=doc_text)
    table = pd.DataFrame({
        0: [None, "Net revenue", "Net income",
            "Average common and dilutive potential common shares outstanding, in thousands"],
        1: [None, None, None, None],
        2: ["2005", "13392", "1198", "1670916"],
    })
    with mock.patch.object(sfd.http, "get", return_value=fake_resp), \
         mock.patch.object(sfd.pd, "read_html", return_value=[table]), \
         mock.patch.object(sfd, "find_selected_financial_data_table", return_value=table):
        df = sfd.build_cik_history(10, filings)

    assert len(df) == 1
    assert df.iloc[0]["net_revenue"] == 13_392_000_000.0, (
        "must fall back to the whole-document 'millions' caption, not the shares row's local 'thousands'")
    print("OK: build_cik_history ignores a shares-count row's local caption, falls back to the whole document")


def test_detect_unit_multiplier():
    assert sfd.detect_unit_multiplier("Some prose (In Millions, Except Per Share)") == 1_000_000.0
    assert sfd.detect_unit_multiplier("figures in thousands of dollars") == 1_000.0
    assert sfd.detect_unit_multiplier("no caption anywhere") == 1.0
    print("OK: detect_unit_multiplier reads the filing's units caption, defaults to 1.0 if absent")


def test_build_cik_history_scales_and_computes_ratios():
    # End-to-end: a filing whose Item 6 table is captioned "(In Millions)" must
    # produce full-dollar figures AND derived ratios (roa/net_margin/etc.), not
    # the bare table values with zero ratio columns -- the two-part gap-tier
    # bug found auditing a 120-ticker sample of the real collected dataset
    # (2026-07-28): item6 net_revenue read ~1e6 too small and roe/roa/net_margin
    # were 100% NaN across every item6 row sampled.
    filings = pd.DataFrame({
        "cik": [8],
        "form_type": ["10-K"],
        "date_filed": pd.to_datetime(["2003-03-01"]),
        "filename": ["intel.txt"],
    })
    fake_resp = mock.Mock(text="irrelevant, read_html is mocked (In Millions, Except Per Share Amounts)")
    table = pd.DataFrame({
        0: [None, "(In Millions)", "Net revenue", "Net income", "Total assets"],
        1: [None, None, None, None, None],
        2: [None, "2002", "$", "$", "$"],
        3: [None, "2002", "25000", "3000", "50000"],
    })
    with mock.patch.object(sfd.http, "get", return_value=fake_resp), \
         mock.patch.object(sfd.pd, "read_html", return_value=[table]), \
         mock.patch.object(sfd, "find_selected_financial_data_table", return_value=table):
        df = sfd.build_cik_history(8, filings)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["net_revenue"] == 25_000_000_000.0, "must be scaled to full dollars, not the bare table value"
    assert pd.notna(row["net_margin"]), "ratios must be computed for the gap tier, not left all-NaN"
    print("OK: build_cik_history scales item6 figures and populates ratio columns")


def test_build_cik_history_drops_implausible_fiscal_year():
    # Real bug, found auditing the collected dataset (2026-07-29): AMG's
    # chained history had fiscal_year=1906/1961/2054 rows (from a mis-selected
    # footnote table, since fixed in find_selected_financial_data_table). Those bogus years
    # didn't just add junk rows -- fundamentals.py's non-calendar-FYE
    # correction picked the 2054 row as "impossible" (end > filing date) and
    # derived a company-wide -49-year offset from it, corrupting AMG's
    # otherwise-correct 2002-2006 rows too. This last-line-of-defense bound
    # drops any single implausible year before it can reach that cascade,
    # independent of whatever mis-parse produced it.
    filings = pd.DataFrame({
        "cik": [9],
        "form_type": ["10-K"],
        "date_filed": pd.to_datetime(["2006-03-16"]),
        "filename": ["amg.txt"],
    })
    fake_resp = mock.Mock(text="irrelevant, read_html is mocked")
    table = pd.DataFrame({0: ["Net revenue"], 1: ["2005"], 2: ["100"]})

    with mock.patch.object(sfd.http, "get", return_value=fake_resp), \
         mock.patch.object(sfd.pd, "read_html", return_value=[table]), \
         mock.patch.object(sfd, "find_selected_financial_data_table", return_value=table), \
         mock.patch.object(sfd, "extract_years",
                            return_value={"2054": {"net_revenue": 100.0}, "2005": {"net_revenue": 200.0}}):
        df = sfd.build_cik_history(9, filings)

    assert len(df) == 1, "the implausible fiscal_year=2054 row must be dropped, not just the good one kept"
    assert df.iloc[0]["fiscal_year"] == 2005
    print("OK: build_cik_history drops implausible fiscal_year rows before they can reach fundamentals.py's cascade")


if __name__ == "__main__":
    test_find_selected_financial_data_table_picks_best_scoring_candidate()
    test_extract_years_reconciles_real_intel_figures()
    test_parenthesized_negative()
    test_row_values_strips_footnote_marker_not_real_negative()
    test_row_values_keeps_genuine_small_negative_when_already_aligned()
    test_row_values_merges_paren_split_across_cells()
    test_detect_unit_multiplier_prefer_first_breaks_ties_deterministically()
    test_build_cik_history_skips_filing_that_crashes_read_html()
    test_extract_years_scales_dollar_rows_but_not_per_share()
    test_find_selected_financial_data_table_rejects_quarterly_and_embedded_digit_false_positives()
    test_find_selected_financial_data_table_prefers_more_years_over_keyword_count()
    test_row_values_collapses_colspan_duplicated_cells_before_footnote_check()
    test_build_cik_history_uses_winning_tables_own_unit_caption()
    test_build_cik_history_ignores_shares_row_caption_falls_back_to_document()
    test_detect_unit_multiplier()
    test_build_cik_history_scales_and_computes_ratios()
    test_build_cik_history_drops_implausible_fiscal_year()
