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
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

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
    exhibits = fds.parse_fds(FAKE_FILING_TEXT)
    assert len(exhibits) == 1
    tags = exhibits[0]
    assert tags["ARTICLE"] == "5"
    assert tags["TOTAL-ASSETS"] == "13,873"
    print("OK: parse_fds extracts the EX-27 tag-value block, not the main filing text")


def test_parse_fds_empty_when_absent():
    assert fds.parse_fds("<DOCUMENT><TYPE>10-K\n<TEXT>no exhibit here</TEXT></DOCUMENT>") == []
    print("OK: parse_fds returns [] when no EX-27 exhibit exists")


def test_parse_fds_finds_every_bundled_exhibit():
    # Real bug, found immediately on scaling past a single company (2026-07-28):
    # Coca-Cola's real 1998-03-09 10-K bundles THREE EX-27 exhibits at once --
    # EX-27.1 (FY1995, restated comparative), EX-27.2 (FY1996, restated comparative),
    # EX-27.3 (FY1997, the actual current year this filing exists to report). The
    # original parse_fds used re.search() (first match only), which kept ONLY
    # EX-27.1's restated FY1995 figures and silently discarded FY1996 and FY1997 --
    # losing the current year's data on every filing that bundles comparatives.
    multi_exhibit_text = """<DOCUMENT>
<TYPE>EX-27.1
<TEXT>
<ARTICLE> 5
<MULTIPLIER> 1,000,000
<FISCAL-YEAR-END>                          DEC-31-1995
<TOTAL-ASSETS>                                  15,041
<NET-INCOME>                                     2,986
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-27.2
<TEXT>
<ARTICLE> 5
<MULTIPLIER> 1,000,000
<FISCAL-YEAR-END>                          DEC-31-1996
<TOTAL-ASSETS>                                  16,161
<NET-INCOME>                                     3,492
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-27.3
<TEXT>
<ARTICLE> 5
<MULTIPLIER> 1,000,000
<FISCAL-YEAR-END>                          DEC-31-1997
<TOTAL-ASSETS>                                  16,940
<NET-INCOME>                                     4,129
</TEXT>
</DOCUMENT>
"""
    exhibits = fds.parse_fds(multi_exhibit_text)
    assert len(exhibits) == 3, f"must find all 3 bundled exhibits, found {len(exhibits)}"
    fyes = [e["FISCAL-YEAR-END"] for e in exhibits]
    assert fyes == ["DEC-31-1995", "DEC-31-1996", "DEC-31-1997"], (
        "must preserve every exhibit, in document order, not just the first")
    print("OK: parse_fds finds every EX-27 exhibit a filing bundles, not just the first")


def test_extract_line_items_reconciles_to_published_figures():
    tags = fds.parse_fds(FAKE_FILING_TEXT)[0]
    items = fds.extract_line_items(tags)
    # Real, independently-verified reconciliation: Coca-Cola FY1994 published 10-K.
    assert items["total_assets"] == 13_873_000_000.0
    assert items["net_income"] == 2_554_000_000.0
    assert items["net_revenue"] == 16_172_000_000.0
    assert items["equity"] == (427 + 4_808) * 1_000_000.0
    assert items["fds_article"] == "5"
    assert items["fds_multiplier"] == 1_000_000.0
    print("OK: extract_line_items reconciles exactly to Coca-Cola's published FY1994 figures")


def test_extract_and_compute_returns_one_result_per_exhibit():
    results = fds.extract_and_compute(FAKE_FILING_TEXT)
    assert len(results) == 1  # FAKE_FILING_TEXT has a single EX-27 exhibit
    r = results[0]
    assert r["total_assets"] == 13_873_000_000.0
    assert str(r["fds_period_end"].date()) == "1994-12-31"
    print("OK: extract_and_compute returns one dict per exhibit, each with its own fds_period_end")


def test_non_article_5_not_silently_mapped():
    tags = {"ARTICLE": "9", "MULTIPLIER": "1000", "TOTAL-ASSETS": "999"}  # bank schema, different tags
    items = fds.extract_line_items(tags)
    assert items == {"fds_article": "9", "fds_multiplier": 1000.0, "fds_multiplier_explicit": True}, (
        "non-Article-5 filings must NOT get Article-5 tags mapped onto their (different) schema")
    print("OK: non-Article-5 filings (banks/insurers/investment cos/utilities) are flagged, not misparsed")


def test_zero_multiplier_defaults_to_one():
    # A malformed/missing <MULTIPLIER> must not silently zero out every figure.
    tags = {"ARTICLE": "5", "PERIOD-TYPE": "YEAR", "TOTAL-ASSETS": "100"}
    items = fds.extract_line_items(tags)
    assert items["total_assets"] == 100.0
    assert items["fds_multiplier"] == 1.0
    assert items["fds_multiplier_explicit"] is False, "an absent tag must be distinguishable from a genuine '1'"
    print("OK: missing/zero <MULTIPLIER> defaults to 1, doesn't zero out every figure")


def test_quarterly_period_type_is_mapped_with_period_months():
    # Phase 2 (docs/US_QUARTERLY_BACKFILL_PLAN.md): 3/6/9-MOS exhibits are now
    # genuinely mapped (they carry real quarterly data), unlike the pre-Phase-2
    # behavior of skipping every non-YEAR exhibit outright.
    tags = {"ARTICLE": "5", "PERIOD-TYPE": "6-MOS", "TOTAL-ASSETS": "999", "MULTIPLIER": "1000000"}
    items = fds.extract_line_items(tags)
    assert items.get("total_assets") == 999_000_000.0
    assert items.get("period_months") == 6
    print("OK: a 6-MOS exhibit is mapped with period_months=6, not skipped")


def test_fiscal_year_end_never_used_as_a_quarterly_exhibits_own_period_end():
    # Real bug, found scaling to ~250 companies (2026-07-28): <FISCAL-YEAR-END> is
    # only reliable as an exhibit's OWN period end when PERIOD-TYPE is YEAR.
    # Confirmed on ADP's real 1998-09-23 10-K: it bundles an Article-5 exhibit with
    # PERIOD-TYPE=6-MOS but FISCAL-YEAR-END=DEC-31-1998 (the eventual full-year
    # cutoff, not the ~1998-06-30 the 6-month figures actually describe) --
    # produced a fundamentals_available_date earlier than its own fds_period_end,
    # a lookahead-shaped artifact. _fds_period_end (not extract_line_items) is
    # where this is now enforced: a quarterly exhibit with no <PERIOD-END> tag
    # must yield NaT, never fall back to the misleading <FISCAL-YEAR-END>.
    tags = {"ARTICLE": "5", "PERIOD-TYPE": "6-MOS", "FISCAL-YEAR-END": "DEC-31-1998",
            "TOTAL-ASSETS": "999", "MULTIPLIER": "1000000"}
    assert pd.isna(fds._fds_period_end(tags)), \
        "a quarterly exhibit with no <PERIOD-END> must not fall back to <FISCAL-YEAR-END>"
    tags["PERIOD-END"] = "JUN-30-1998"
    assert fds._fds_period_end(tags) == pd.Timestamp("1998-06-30"), \
        "<PERIOD-END>, when present, is this exhibit's real own period end"
    # A YEAR exhibit still correctly uses <FISCAL-YEAR-END> (unchanged, already-
    # verified annual behavior -- the two tags coincide there).
    year_tags = {"ARTICLE": "5", "PERIOD-TYPE": "YEAR", "FISCAL-YEAR-END": "DEC-31-1998"}
    assert fds._fds_period_end(year_tags) == pd.Timestamp("1998-12-31")
    print("OK: <FISCAL-YEAR-END> is never a quarterly exhibit's period end, only <PERIOD-END> is")


def test_missing_multiplier_borrows_from_sibling_exhibit():
    # Real bug, confirmed on WMT's actual filings (2026-07-30): <MULTIPLIER> is
    # genuinely OPTIONAL per SEC's EX-27 schema -- WMT's 1995/1996 10-Ks tag it
    # explicitly (1,000,000), but 1997-2000 omit the tag entirely even though the
    # raw figures are STILL reported at the same implicit millions scale (1997's
    # real TOTAL-ASSETS=39,604 is Walmart's actual ~$39.6B, not $39,604 -- silently
    # defaulting the absent tag to 1.0 understated every dollar field by 10^6 for
    # exactly the filings that omit it).
    tags = {"ARTICLE": "5", "PERIOD-TYPE": "YEAR", "MULTIPLIER": "1,000,000",
            "TOTAL-ASSETS": "32819", "NET-INCOME": "1608", "TOTAL-REVENUES": "83412",
            "CURRENT-ASSETS": "0", "CURRENT-LIABILITIES": "0", "CASH": "0", "BONDS": "0", "CGS": "0"}
    explicit_row = {**fds.extract_line_items(tags), "fds_period_end": pd.Timestamp("1995-01-31")}
    tags_missing = {**tags, "TOTAL-ASSETS": "39604", "NET-INCOME": "3056", "TOTAL-REVENUES": "106146"}
    del tags_missing["MULTIPLIER"]
    missing_row = {**fds.extract_line_items(tags_missing), "fds_period_end": pd.Timestamp("1997-01-31")}
    assert missing_row["fds_multiplier_explicit"] is False
    assert missing_row["total_assets"] == 39604.0, "before the fix: undeclared multiplier defaults to 1"

    df = pd.DataFrame([explicit_row, missing_row])
    fixed = fds._fill_missing_multipliers(df)
    borrowed = fixed[fixed["fds_period_end"] == pd.Timestamp("1997-01-31")].iloc[0]
    assert borrowed["total_assets"] == 39_604_000_000.0, (
        f"must borrow the sibling exhibit's 1,000,000 multiplier, got {borrowed['total_assets']}")
    assert borrowed["net_revenue"] == 106_146_000_000.0
    assert borrowed["fds_multiplier"] == 1_000_000.0
    print("OK: a missing <MULTIPLIER> borrows the value from a sibling exhibit of the same CIK")


def test_missing_multiplier_left_flagged_when_no_sibling_available():
    # Confirmed on TXT's actual filing history (2026-07-30): every single
    # collected EX-27 exhibit omits <MULTIPLIER> -- there is no sibling to
    # borrow a real scale from. Must stay visibly flagged (fds_multiplier_explicit
    # stays False) rather than silently presented as equally trustworthy as a
    # confirmed exhibit -- guessing a scale with zero evidence would be worse
    # than leaving it honestly unresolved.
    tags = {"ARTICLE": "5", "PERIOD-TYPE": "YEAR", "TOTAL-ASSETS": "20925",
            "NET-INCOME": "100", "TOTAL-REVENUES": "9683"}
    row = {**fds.extract_line_items(tags), "fds_period_end": pd.Timestamp("1994-12-31")}
    df = pd.DataFrame([row])
    result = fds._fill_missing_multipliers(df)
    assert result.iloc[0]["total_assets"] == 20925.0, "no sibling to borrow from -- must stay unchanged"
    assert result.iloc[0]["fds_multiplier_explicit"] == False  # noqa: E712 (real bool, not np.bool_)
    print("OK: a missing multiplier with no sibling anywhere in the CIK's history stays honestly flagged, not guessed")


def test_build_cik_history_skips_post_ex27_era_filings():
    # Real efficiency bug, found scaling past a handful of companies (2026-07-28):
    # the original code fetched EVERY 10-K a CIK ever filed just to check for an
    # EX-27, including decades of post-2001 filings that structurally cannot have
    # one (this tier's own prevalence measurement found 2001 ~0%, nothing later).
    # For a company with 30 years of post-2001 history, that's ~30x wasted fetches.
    filings = pd.DataFrame({
        "cik": [1, 1, 1],
        "form_type": ["10-K", "10-K", "10-K"],
        "date_filed": pd.to_datetime(["1996-03-01", "2010-03-01", "2023-03-01"]),
        "filename": ["old.txt", "mid.txt", "recent.txt"],
    })
    requested = []
    def fake_fetch(filename):
        requested.append(filename)
        return None  # content doesn't matter for this test
    with mock.patch.object(fds, "fetch_filing_text", fake_fetch):
        fds.build_cik_history(1, filings)
    assert requested == ["old.txt"], (
        f"must only fetch filings up to EX27_ERA_END, fetched {requested}")
    print("OK: build_cik_history skips filings past the EX-27 era, not every 10-K ever filed")


def test_build_cik_history_drops_unparseable_period_end_instead_of_merging():
    # Real bug: fds_period_end is NaT whenever <FISCAL-YEAR-END> is missing/
    # malformed (a documented unreliability of that tag -- see
    # extract_line_items's ADP docstring). drop_duplicates(subset="fds_period_end")
    # treats NaT == NaT, so two DIFFERENT real fiscal years that both fail to
    # parse a period end used to collapse into one bogus survivor (the
    # earlier-filed one, itself still useless with a NaT end) instead of both
    # being dropped -- silently discarding the later one's real financial data.
    good_text = ("<TYPE>EX-27\n<ARTICLE>5\n<PERIOD-TYPE>YEAR\n"
                 "<FISCAL-YEAR-END>DEC-31-1994\n<TOTAL-ASSETS>100\n<NET-INCOME>10\n")
    # <FISCAL-YEAR-END> omitted entirely -> fds_period_end = NaT
    bad_text_1 = "<TYPE>EX-27\n<ARTICLE>5\n<PERIOD-TYPE>YEAR\n<TOTAL-ASSETS>200\n<NET-INCOME>20\n"
    bad_text_2 = "<TYPE>EX-27\n<ARTICLE>5\n<PERIOD-TYPE>YEAR\n<TOTAL-ASSETS>300\n<NET-INCOME>30\n"

    filings = pd.DataFrame({
        "cik": [1, 1, 1],
        "form_type": ["10-K", "10-K", "10-K"],
        "date_filed": pd.to_datetime(["1994-03-01", "1995-03-01", "1996-03-01"]),
        "filename": ["good.txt", "bad1.txt", "bad2.txt"],
    })
    with mock.patch.object(fds, "fetch_filing_text",
                           side_effect=[good_text, bad_text_1, bad_text_2]):
        result = fds.build_cik_history(1, filings)

    assert len(result) == 1, (
        f"both NaT-period-end exhibits must be dropped, not collapsed into one "
        f"bogus survivor via drop_duplicates treating NaT == NaT; got {len(result)} row(s)")
    assert result.iloc[0]["total_assets"] == 100.0
    assert str(result.iloc[0]["fds_period_end"].date()) == "1994-12-31"
    print("OK: build_cik_history drops (not merges) exhibits with an unparseable period end")


def test_measure_prevalence_handles_list_return_from_parse_fds():
    # Real bug: parse_fds returns a LIST (a filing can bundle multiple EX-27
    # exhibits), but measure_prevalence used to treat it like a dict/None --
    # `tags is not None` was True even for an empty list (has_ex27 always
    # True regardless of content), and `(tags or {}).get("ARTICLE")` raised
    # AttributeError on the first filing that genuinely had an exhibit (a
    # non-empty list has no .get method). Covers both: a filing with an
    # exhibit and one without.
    filings = pd.DataFrame({
        "cik": [1, 1], "form_type": ["10-K", "10-K"],
        "date_filed": pd.to_datetime(["1996-03-01", "1997-03-01"]),
        "filename": ["has_ex27.txt", "no_ex27.txt"],
    })
    with mock.patch.object(fds, "fetch_filing_text",
                           side_effect=["<TYPE>EX-27\n<ARTICLE>5", "no exhibit here"]):
        result = fds.measure_prevalence(filings, years=[1996, 1997], sample_per_year=1)
    by_year = result.set_index("year")
    assert by_year.loc[1996, "has_ex27"] and by_year.loc[1996, "article"] == "5"
    assert not by_year.loc[1997, "has_ex27"] and by_year.loc[1997, "article"] is None
    print("OK: measure_prevalence handles parse_fds's list return without crashing or misreporting")


def test_build_cik_history_produces_discrete_quarters_end_to_end():
    # End-to-end Phase 2 (docs/US_QUARTERLY_BACKFILL_PLAN.md): real EX-27 tag
    # shape, CIK 1000366 FY1999 (fetched live from EDGAR 2026-08-01), through
    # the FULL pipeline -- fetch -> parse -> multiplier-fill -> as-first-
    # reported dedup -> ytd_to_discrete -> ratio recompute. MULTIPLIER=1
    # throughout so raw tag values are directly comparable to
    # test_ytd_to_discrete_reconciles_to_real_filing's companyfacts-level fixture.
    def exhibit(period_type, period_end, revenue, ni, assets):
        return (f"<TYPE>EX-27\n<ARTICLE>5\n<PERIOD-TYPE>{period_type}\n"
                f"<FISCAL-YEAR-END>DEC-31-1999\n<PERIOD-END>{period_end}\n"
                f"<MULTIPLIER>1\n<TOTAL-REVENUES>{revenue}\n<NET-INCOME>{ni}\n"
                f"<TOTAL-ASSETS>{assets}\n")

    texts = {
        "q1.txt": exhibit("3-MOS", "MAR-31-1999", 121701, 9212, 1061164),
        "q2.txt": exhibit("6-MOS", "JUN-30-1999", 275712, 18369, 1131387),
        "q3.txt": exhibit("9-MOS", "SEP-30-1999", 428558, 36739, 850394),
        "fy.txt": exhibit("YEAR", "DEC-31-1999", 576997, 32563, 971809),
    }
    filings = pd.DataFrame({
        "cik": [1] * 4,
        "form_type": ["10-Q", "10-Q", "10-Q", "10-K"],
        "date_filed": pd.to_datetime(["1999-05-14", "1999-08-13", "1999-11-12", "2000-03-30"]),
        "filename": list(texts.keys()),
    })
    with mock.patch.object(fds, "fetch_filing_text", lambda fn: texts[fn]):
        df = fds.build_cik_history(1, filings)

    assert len(df) == 4, f"expected 4 discrete quarters, got {len(df)}"
    df = df.sort_values("fds_period_end").reset_index(drop=True)
    assert df["net_revenue"].tolist() == [121701.0, 154011.0, 152846.0, 148439.0]
    assert df["net_income"].tolist() == [9212.0, 9157.0, 18370.0, -4176.0], \
        "Q4 must be a real loss quarter, invisible in the old annual-only row"
    assert df["total_assets"].tolist() == [1061164.0, 1131387.0, 850394.0, 971809.0], \
        "instant column must never be differenced"
    q4 = df.iloc[3]
    assert q4["net_margin"] == q4["net_income"] / q4["net_revenue"] * 100, \
        "ratios must be recomputed on the DISCRETE Q4 figures, not the annual YTD ones"
    assert (df["period_months"] == 3).all()
    assert df["flows_derived"].tolist() == [0, 1, 1, 1]
    print("OK: build_cik_history end-to-end produces discrete quarters with ratios recomputed on them")


if __name__ == "__main__":
    test_parse_fds_extracts_tags()
    test_parse_fds_empty_when_absent()
    test_parse_fds_finds_every_bundled_exhibit()
    test_extract_line_items_reconciles_to_published_figures()
    test_extract_and_compute_returns_one_result_per_exhibit()
    test_non_article_5_not_silently_mapped()
    test_zero_multiplier_defaults_to_one()
    test_quarterly_period_type_is_mapped_with_period_months()
    test_fiscal_year_end_never_used_as_a_quarterly_exhibits_own_period_end()
    test_missing_multiplier_borrows_from_sibling_exhibit()
    test_missing_multiplier_left_flagged_when_no_sibling_available()
    test_build_cik_history_skips_post_ex27_era_filings()
    test_build_cik_history_drops_unparseable_period_end_instead_of_merging()
    test_measure_prevalence_handles_list_return_from_parse_fds()
    test_build_cik_history_produces_discrete_quarters_end_to_end()
