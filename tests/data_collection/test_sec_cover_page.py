"""
test_sec_cover_page.py
=======================
Self-check for sec/cover_page.py's pure parsing logic (no network). The 3
fixture strings below are the EXACT text (whitespace and all) pulled live
from real EDGAR filings 2026-08-12 -- see cover_page.py's own docstring for
the accession numbers this reconciles against.

Usage: python tests/data_collection/test_sec_cover_page.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_collection.sec import cover_page

# AAPL 10-K, filed 1994-12-13 (accession 0000320193-94-000016).
_AAPL_1994_COVER = """
The aggregate market value of voting stock held by nonaffiliates of the
Registrant was approximately $ 4,193,708,403 as of December 2, 1994, based
upon the closing price on the Nasdaq National Market reported for such
date.  Shares of Common Stock held by each executive officer and director
and by each person who beneficially owns more than 5% of the outstanding
Common Stock have been excluded in that such persons may under certain
circumstances be deemed to be affiliates.  This determination of
executive officer or affiliate status is not necessarily a conclusive
determination for other purposes.

    119,891,418 shares of Common Stock Issued and Outstanding as of
          December 2, 1994.
"""

# XOM 10-K, filed 2002-03-27 (accession 0000930661-02-000889).
_XOM_10K_COVER = """
Common Stock, without par value (6,792,598,170 shares
   outstanding at February 28, 2002)                       New York Stock Exchange
"""

# XOM 10-Q, filed 2003-05-14 (accession 0000034088-03-000063).
_XOM_10Q_COVER = """
Indicate the number of shares outstanding of each of the issuer's classes
of common stock, as of the latest practicable date.

             Class                   Outstanding as of March 31, 2003
_______________________________      ________________________________
Common stock, without par value                6,679,390,610
"""


def test_extracts_aapl_sentence_style():
    shares, asof = cover_page.extract_shares_outstanding(_AAPL_1994_COVER, "1994-12-13")
    assert shares == 119_891_418.0
    assert asof == pd.Timestamp("1994-12-02")
    print("OK: extract_shares_outstanding parses AAPL's real 1994 sentence-style cover page")


def test_extracts_xom_10k_parenthetical_style():
    shares, asof = cover_page.extract_shares_outstanding(_XOM_10K_COVER, "2002-03-27")
    assert shares == 6_792_598_170.0
    assert asof == pd.Timestamp("2002-02-28")
    print("OK: extract_shares_outstanding parses XOM's real 2002 parenthetical cover page "
          "(a real newline splits 'shares' from 'outstanding')")


def test_extracts_xom_10q_tabular_style():
    shares, asof = cover_page.extract_shares_outstanding(_XOM_10Q_COVER, "2003-05-14")
    assert shares == 6_679_390_610.0
    assert asof == pd.Timestamp("2003-03-31")
    print("OK: extract_shares_outstanding parses XOM's real 2003 tabular 10-Q cover page")


def test_no_match_returns_nan():
    shares, asof = cover_page.extract_shares_outstanding("Item 1. Business. We make widgets.", "2003-01-01")
    assert pd.isna(shares) and pd.isna(asof)
    print("OK: extract_shares_outstanding returns NaN/NaT on ordinary filing text with no cover-page match")


def test_none_text_returns_nan():
    shares, asof = cover_page.extract_shares_outstanding(None, "2003-01-01")
    assert pd.isna(shares) and pd.isna(asof)
    print("OK: extract_shares_outstanding handles a missing filing (None text) without raising")


def test_rejects_implausibly_large_share_count():
    # Same pattern as the AAPL fixture, but with a share count far beyond any
    # real public filer (a garbled/OCR-shaped false positive) -- must not be
    # accepted just because the surrounding template matched.
    text = "999,999,999,999,999 shares of Common Stock Issued and Outstanding as of January 1, 2003."
    shares, asof = cover_page.extract_shares_outstanding(text, "2003-01-02")
    assert pd.isna(shares) and pd.isna(asof)
    print("OK: extract_shares_outstanding rejects a share count outside any real filer's plausible range")


def test_rejects_date_far_from_filing_date():
    # A real template match, but the "as of" date is nowhere near this
    # filing's own date_filed -- a cover-page snapshot is always close to the
    # filing that states it; a stray same-shaped match elsewhere in the
    # document (e.g. an unrelated historical reference) must not be accepted.
    text = "119,891,418 shares of Common Stock Issued and Outstanding as of December 2, 1994."
    shares, asof = cover_page.extract_shares_outstanding(text, "2003-01-01")
    assert pd.isna(shares) and pd.isna(asof)
    print("OK: extract_shares_outstanding rejects a match whose as-of date is far from the filing date")


def test_date_first_there_were_style():
    # Common boilerplate, date BEFORE the share count, no repeated trailing
    # date -- distinct from all 3 live-verified templates (all trailing-date).
    text = "As of June 30, 2004, there were 45,678,901 shares of common stock outstanding."
    shares, asof = cover_page.extract_shares_outstanding(text, "2004-08-05")
    assert shares == 45_678_901.0
    assert asof == pd.Timestamp("2004-06-30")
    print("OK: extract_shares_outstanding parses the date-first 'As of DATE, there were N shares "
          "... outstanding' template")


if __name__ == "__main__":
    test_extracts_aapl_sentence_style()
    test_extracts_xom_10k_parenthetical_style()
    test_extracts_xom_10q_tabular_style()
    test_no_match_returns_nan()
    test_none_text_returns_nan()
    test_rejects_implausibly_large_share_count()
    test_rejects_date_far_from_filing_date()
    test_date_first_there_were_style()
    print("\nAll cover_page tests passed.")
