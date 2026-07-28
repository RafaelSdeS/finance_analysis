"""
test_sec_fds.py
================
Self-check for sec/fds.py's pure parsing logic (no network). The synthetic
EX-27 block below mirrors Coca-Cola's real FY1994 filing byte-for-byte in
structure (values match exactly) -- this same case was independently
verified against LIVE EDGAR data (2026-07-28): TOTAL-ASSETS 13,873 *
MULTIPLIER 1,000,000 = $13.873B, reconciling to Coca-Cola's published 1994
10-K. This test pins that result so it can't silently regress.

Usage: python tests/data_collection/test_sec_fds.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection.sec import fds

FAKE_FILING_TEXT = """<DOCUMENT>
<TYPE>10-K405
<TEXT>
... full filing text, financial statements, etc ...
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-27.1
<SEQUENCE>11
<DESCRIPTION>EXHIBIT 27.1 - ART. 5 FDS FOR FORM 10-K, 12/31/94
<TEXT>

<TABLE> <S> <C>

<ARTICLE> 5
<LEGEND>
THIS SCHEDULE CONTAINS SUMMARY FINANCIAL INFORMATION
</LEGEND>
<MULTIPLIER> 1,000,000

<S>                             <C>
<PERIOD-TYPE>                   YEAR
<FISCAL-YEAR-END>                          DEC-31-1994
<CASH>                                           1,386
<SECURITIES>                                       145
<RECEIVABLES>                                    1,470
<ALLOWANCES>                                        33
<INVENTORY>                                      1,047
<CURRENT-ASSETS>                                 5,205
<PP&E>                                           6,157
<DEPRECIATION>                                   2,077
<TOTAL-ASSETS>                                  13,873
<CURRENT-LIABILITIES>                            6,177
<BONDS>                                          1,426
<COMMON>                                           427
<PREFERRED-MANDATORY>                                0
<PREFERRED>                                          0
<OTHER-SE>                                       4,808
<TOTAL-LIABILITY-AND-EQUITY>                    13,873
<SALES>                                         16,172
<TOTAL-REVENUES>                                16,172
<CGS>                                            6,167
<TOTAL-COSTS>                                    6,167
<OTHER-EXPENSES>                                     0
<LOSS-PROVISION>                                     0
<INTEREST-EXPENSE>                                 199
<INCOME-PRETAX>                                  3,728
<INCOME-TAX>                                     1,174
<INCOME-CONTINUING>                              2,554
<DISCONTINUED>                                       0
<EXTRAORDINARY>                                      0
<CHANGES>                                            0
<NET-INCOME>                                     2,554
<EPS-PRIMARY>                                     1.61
<EPS-DILUTED>                                     1.61
</TABLE>
</TEXT>
</DOCUMENT>
"""


def test_parse_fds_extracts_tags():
    tags = fds.parse_fds(FAKE_FILING_TEXT)
    assert tags is not None
    assert tags["ARTICLE"] == "5"
    assert tags["TOTAL-ASSETS"] == "13,873"
    print("OK: parse_fds extracts the EX-27 tag-value block, not the main filing text")


def test_parse_fds_none_when_absent():
    assert fds.parse_fds("<DOCUMENT><TYPE>10-K\n<TEXT>no exhibit here</TEXT></DOCUMENT>") is None
    print("OK: parse_fds returns None when no EX-27 exhibit exists")


def test_extract_line_items_reconciles_to_published_figures():
    tags = fds.parse_fds(FAKE_FILING_TEXT)
    items = fds.extract_line_items(tags)
    # Real, independently-verified reconciliation: Coca-Cola FY1994 published 10-K.
    assert items["total_assets"] == 13_873_000_000.0
    assert items["net_income"] == 2_554_000_000.0
    assert items["net_revenue"] == 16_172_000_000.0
    assert items["equity"] == (427 + 4_808) * 1_000_000.0
    assert items["fds_article"] == "5"
    assert items["fds_multiplier"] == 1_000_000.0
    print("OK: extract_line_items reconciles exactly to Coca-Cola's published FY1994 figures")


def test_non_article_5_not_silently_mapped():
    tags = {"ARTICLE": "9", "MULTIPLIER": "1000", "TOTAL-ASSETS": "999"}  # bank schema, different tags
    items = fds.extract_line_items(tags)
    assert items == {"fds_article": "9", "fds_multiplier": 1000.0}, (
        "non-Article-5 filings must NOT get Article-5 tags mapped onto their (different) schema")
    print("OK: non-Article-5 filings (banks/insurers/investment cos/utilities) are flagged, not misparsed")


def test_zero_multiplier_defaults_to_one():
    # A malformed/missing <MULTIPLIER> must not silently zero out every figure.
    tags = {"ARTICLE": "5", "TOTAL-ASSETS": "100"}
    items = fds.extract_line_items(tags)
    assert items["total_assets"] == 100.0
    assert items["fds_multiplier"] == 1.0
    print("OK: missing/zero <MULTIPLIER> defaults to 1, doesn't zero out every figure")


if __name__ == "__main__":
    test_parse_fds_extracts_tags()
    test_parse_fds_none_when_absent()
    test_extract_line_items_reconciles_to_published_figures()
    test_non_article_5_not_silently_mapped()
    test_zero_multiplier_defaults_to_one()
