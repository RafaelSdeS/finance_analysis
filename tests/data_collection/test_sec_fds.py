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


def test_to_number_parses_parenthesized_negatives():
    # Real bug, confirmed on TCX's actual 1998-09-30 10-Q (2026-08-01):
    # <OTHER-SE>(2,424,212) -- standard accounting notation for a real
    # negative (TCX's genuine financial distress that quarter) -- silently
    # returned NaN, unlike selected_financial_data.py's _parse_value, which
    # already handles this. Compounds in extract_line_items's equity =
    # nan_to_num(COMMON) + nan_to_num(OTHER-SE): a missing COMPONENT
    # masquerades as a real value of 0 instead of propagating NaN, turning a
    # genuine -$2.4M equity into a false exact $0.00 -- caught by
    # fundamentals.py's _FLOORS, but for the wrong reason (looks like a
    # parsing gap, is actually a silently-dropped real negative).
    assert fds._to_number("(2,424,212)") == -2424212.0
    assert fds._to_number("500") == 500.0, "a plain positive must be unaffected"
    assert fds._to_number("(500") == 500.0, (
        "an unmatched leading paren (malformed, no real EX-27 case seen) must not crash -- "
        "stripped and treated as positive, same as before this fix")
    print("OK: _to_number parses a parenthesized negative, not just a leading-minus one")


def test_extract_line_items_equity_survives_a_parenthesized_negative_component():
    # End-to-end: the real TCX tags (verified live 2026-08-01) must derive
    # equity as the real negative sum, not a false $0.
    tags = {"ARTICLE": "5", "PERIOD-TYPE": "9-MOS", "PERIOD-END": "SEP-30-1998",
            "COMMON": "0", "OTHER-SE": "(2,424,212)",
            "TOTAL-ASSETS": "13,701,265", "TOTAL-REVENUES": "10,312,996"}
    items = fds.extract_line_items(tags)
    assert items["equity"] == -2424212.0, (
        f"a missing/zero COMMON plus a real negative OTHER-SE must derive a real negative "
        f"equity, not a false 0.0 -- got {items['equity']}")
    print("OK: extract_line_items derives real negative equity, not a false $0 from a masked component")


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


def test_missing_multiplier_rejects_implausible_borrow():
    # Real bug, found via cross-vendor validation (tests/data_collection/
    # validate_us_vs_vendor.py, check_tier_seams) on GIS (CIK 40704, 2026-08-06):
    # only 1 of ~19 exhibits declares <MULTIPLIER> explicitly (1,000,000, giving
    # a plausible ~$3.3B total_assets); blindly borrowing that same factor onto
    # EVERY other exhibit turned an already-correct ~$5.19B raw exhibit (its own
    # tag values were evidently already near full-dollar scale, needing no
    # further scaling at all) into a nonsense $5.19e15. The borrowed multiplier
    # must only be accepted when it lands CLOSER (log-scale) to this CIK's own
    # explicit-tier reference than leaving the exhibit unscaled would.
    explicit_tags = {"ARTICLE": "5", "PERIOD-TYPE": "YEAR", "MULTIPLIER": "1,000,000",
                      "TOTAL-ASSETS": "3300", "NET-INCOME": "100", "TOTAL-REVENUES": "5000",
                      "CURRENT-ASSETS": "0", "CURRENT-LIABILITIES": "0", "CASH": "0", "BONDS": "0", "CGS": "0"}
    explicit_row = {**fds.extract_line_items(explicit_tags), "fds_period_end": pd.Timestamp("1999-12-31")}

    # This exhibit's raw TOTAL-ASSETS is already ~full-dollar scale (5.19B) --
    # borrowing the 1,000,000 factor would blow it up to 5.19e15, wildly further
    # from the ~3.3e9 reference than leaving it alone.
    implausible_tags = {**explicit_tags, "TOTAL-ASSETS": "5190000000", "NET-INCOME": "150000000",
                         "TOTAL-REVENUES": "6000000000"}
    del implausible_tags["MULTIPLIER"]
    implausible_row = {**fds.extract_line_items(implausible_tags), "fds_period_end": pd.Timestamp("2000-06-30")}

    df = pd.DataFrame([explicit_row, implausible_row])
    fixed = fds._fill_missing_multipliers(df)
    kept = fixed[fixed["fds_period_end"] == pd.Timestamp("2000-06-30")].iloc[0]
    assert kept["total_assets"] == 5_190_000_000.0, (
        f"an already-plausible exhibit must NOT be force-rescaled by a borrowed multiplier, got {kept['total_assets']}")
    assert kept["fds_multiplier_explicit"] == False  # noqa: E712 (real bool, not np.bool_)
    print("OK: a borrowed multiplier is rejected when it would make the exhibit LESS plausible, not more")


def test_fill_missing_multipliers_canonical_reference_ignores_malformed_total_assets():
    # Real bug, confirmed 2026-08-12: the "explicit" reference set used to
    # require BOTH fds_multiplier_explicit AND total_assets.notna() -- if most
    # genuinely-explicit rows have a malformed TOTAL-ASSETS tag, canonical
    # (the scale itself) got derived from whichever unrepresentative minority
    # still had a clean total_assets, not from the true explicit population.
    # Here, 2 of 3 explicit rows have a NaN total_assets but the correct
    # 1,000,000 multiplier -- canonical must still resolve to 1,000,000, not
    # get skewed or starved by the malformed majority.
    explicit_clean = {"fds_multiplier_explicit": True, "fds_multiplier": 1_000_000.0,
                       "total_assets": 5_000_000_000.0, "fds_period_end": pd.Timestamp("1996-12-31")}
    explicit_malformed_1 = {"fds_multiplier_explicit": True, "fds_multiplier": 1_000_000.0,
                             "total_assets": float("nan"), "fds_period_end": pd.Timestamp("1997-12-31")}
    explicit_malformed_2 = {"fds_multiplier_explicit": True, "fds_multiplier": 1_000_000.0,
                             "total_assets": float("nan"), "fds_period_end": pd.Timestamp("1998-12-31")}
    missing_row = {"fds_multiplier_explicit": False, "fds_multiplier": 1.0,
                   "total_assets": 5200.0, "fds_period_end": pd.Timestamp("1999-12-31")}
    df = pd.DataFrame([explicit_clean, explicit_malformed_1, explicit_malformed_2, missing_row])
    fixed = fds._fill_missing_multipliers(df)
    borrowed = fixed[fixed["fds_period_end"] == pd.Timestamp("1999-12-31")].iloc[0]
    assert borrowed["fds_multiplier"] == 1_000_000.0, (
        f"canonical must resolve from all 3 explicit rows, not just the 1 with clean total_assets, "
        f"got {borrowed['fds_multiplier']}")
    assert borrowed["total_assets"] == 5_200_000_000.0
    print("OK: _fill_missing_multipliers derives canonical from every explicit row, not just those with usable total_assets")


def test_infer_multiplier_from_trusted_tiers_never_touches_a_row_already_resolved_to_canonical_one():
    # Real bug, confirmed 2026-08-12: infer_multiplier_from_trusted_tiers used
    # to treat `fds_multiplier == 1.0` as its own "still unresolved" proxy.
    # A non-explicit exhibit whose raw value already sits at this CIK's own
    # CONFIRMED canonical scale (here canonical == 1.0 -- a company whose
    # explicit filings genuinely declare no scaling) never goes through
    # _fill_missing_multipliers' accept-and-rescale path at all (it was never
    # "missing" in the first place), yet it's just as resolved as one that
    # did -- indistinguishable, under the old bare-value check, from AEO's
    # shape (no explicit tag ANYWHERE, genuinely still unresolved). It must
    # not be silently re-anchored to an unrelated cross-tier reference.
    explicit_row = {"fds_multiplier_explicit": True, "fds_multiplier": 1.0,
                     "total_assets": 500.0, "fds_period_end": pd.Timestamp("1996-12-31")}
    already_matching_row = {"fds_multiplier_explicit": False, "fds_multiplier": 1.0,
                             "total_assets": 480.0, "fds_period_end": pd.Timestamp("1997-12-31")}
    df = pd.DataFrame([explicit_row, already_matching_row])
    fixed = fds._fill_missing_multipliers(df)
    resolved = fixed[fixed["fds_period_end"] == pd.Timestamp("1997-12-31")].iloc[0]
    assert resolved["fds_multiplier"] == 1.0 and resolved["fds_multiplier_resolved"], (
        "setup check: this row must already be marked resolved (matches this CIK's own canonical=1.0) "
        "before the second pass runs")

    # A wildly different trusted reference that, under the old bug, would
    # have looked "unresolved" (fds_multiplier == 1.0) and gotten rescaled.
    trusted = pd.DataFrame([{"end": pd.Timestamp("1997-12-25"), "total_assets": 480_000_000.0}])
    twice_fixed = fds.infer_multiplier_from_trusted_tiers(fixed, trusted)
    untouched = twice_fixed[twice_fixed["fds_period_end"] == pd.Timestamp("1997-12-31")].iloc[0]
    assert untouched["fds_multiplier"] == 1.0, (
        f"a row already resolved by _fill_missing_multipliers must never be re-anchored here, "
        f"got {untouched['fds_multiplier']}")
    assert untouched["total_assets"] == 480.0
    print("OK: infer_multiplier_from_trusted_tiers never re-anchors a row already resolved to canonical multiplier=1.0")


def test_infer_multiplier_from_trusted_tiers_resolves_real_aeo_shape():
    # Real bug, confirmed end-to-end on AEO's (American Eagle Outfitters, CIK
    # 919012) actual filing history (2026-08-06): NOT ONE of its ~16 ex27
    # exhibits ever declares <MULTIPLIER> explicitly, so _fill_missing_multipliers
    # (borrowing from a same-tier sibling) has zero signal to work with --
    # every dollar field stays at the untouched default (multiplier=1.0)
    # forever, understated 1000x. Confirmed by cross-referencing AEO's OWN
    # item6 tier (already correct after today's other fixes): FY1998 net
    # sales $405,713,000 annual: a raw ex27 TOTAL-REVENUES of 104902 for the
    # Q3 1997 quarter is genuinely thousands-scale ($104,902,000, ~26% of the
    # annual figure -- plausible for a retailer's pre-holiday quarter), not
    # $104,902 as multiplier=1.0 leaves it.
    ex27_tags = {"ARTICLE": "5", "PERIOD-TYPE": "3-MOS", "TOTAL-ASSETS": "134570",
                 "NET-INCOME": "6276", "TOTAL-REVENUES": "104902",
                 "CURRENT-ASSETS": "0", "CURRENT-LIABILITIES": "0", "CASH": "0", "BONDS": "0", "CGS": "0"}
    ex27_row = {**fds.extract_line_items(ex27_tags), "end": pd.Timestamp("1997-11-01")}
    assert ex27_row["fds_multiplier_explicit"] is False
    assert ex27_row["total_assets"] == 134570.0, "before the fix: no sibling to borrow from, stays at raw scale"

    df = pd.DataFrame([ex27_row])
    # trusted = AEO's own real item6 FY1998 row (already correctly scaled)
    trusted = pd.DataFrame([{"end": pd.Timestamp("1998-03-31"), "total_assets": 144_795_000.0}])
    fixed = fds.infer_multiplier_from_trusted_tiers(df, trusted)
    row = fixed.iloc[0]
    assert row["total_assets"] == 134_570_000.0, (
        f"must infer the x1,000 scale from the trusted item6 reference, got {row['total_assets']}")
    assert row["net_revenue"] == 104_902_000.0
    assert row["fds_multiplier"] == 1_000.0
    print("OK: infer_multiplier_from_trusted_tiers resolves AEO's real never-explicit-anywhere shape")


def test_infer_multiplier_from_trusted_tiers_rejects_no_close_candidate():
    # None of the 3 valid EX-27 multipliers (1, 1,000, 1,000,000) land within
    # 3x of a reference that doesn't correspond to any real scale of this
    # row's raw digits -- must leave the row untouched (multiplier=1.0)
    # rather than pick the "least bad" candidate anyway.
    ex27_tags = {"ARTICLE": "5", "PERIOD-TYPE": "YEAR", "TOTAL-ASSETS": "500",
                 "NET-INCOME": "10", "TOTAL-REVENUES": "100",
                 "CURRENT-ASSETS": "0", "CURRENT-LIABILITIES": "0", "CASH": "0", "BONDS": "0", "CGS": "0"}
    ex27_row = {**fds.extract_line_items(ex27_tags), "end": pd.Timestamp("1997-01-01")}
    df = pd.DataFrame([ex27_row])
    # 500 * {1, 1000, 1e6} = {500, 500000, 5e8} -- none within 3x of 42 (a
    # reference matching none of them).
    trusted = pd.DataFrame([{"end": pd.Timestamp("1997-01-15"), "total_assets": 42.0}])
    fixed = fds.infer_multiplier_from_trusted_tiers(df, trusted)
    assert fixed.iloc[0]["fds_multiplier"] == 1.0, "must not force a candidate that isn't actually plausible"
    print("OK: infer_multiplier_from_trusted_tiers leaves a row untouched when no candidate is within 3x")


def test_infer_multiplier_from_trusted_tiers_respects_max_gap_days():
    # A trusted reference decades away from the ex27 row's own period must
    # not be used as a scale anchor at all, plausible-looking match or not.
    ex27_tags = {"ARTICLE": "5", "PERIOD-TYPE": "YEAR", "TOTAL-ASSETS": "134570",
                 "NET-INCOME": "6276", "TOTAL-REVENUES": "104902",
                 "CURRENT-ASSETS": "0", "CURRENT-LIABILITIES": "0", "CASH": "0", "BONDS": "0", "CGS": "0"}
    ex27_row = {**fds.extract_line_items(ex27_tags), "end": pd.Timestamp("1997-11-01")}
    df = pd.DataFrame([ex27_row])
    trusted = pd.DataFrame([{"end": pd.Timestamp("2020-01-01"), "total_assets": 144_795_000.0}])
    fixed = fds.infer_multiplier_from_trusted_tiers(df, trusted)
    assert fixed.iloc[0]["fds_multiplier"] == 1.0, "a reference decades away must not be used, however plausible"
    print("OK: infer_multiplier_from_trusted_tiers ignores a trusted reference outside max_gap_days")


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


def test_build_cik_history_attaches_shares_outstanding_from_cover_page():
    # The full submission text fetch_filing_text returns already carries the
    # filing's cover page ahead of the EX-27 exhibit (real EDGAR layout, see
    # cover_page.py) -- build_cik_history must parse it out and attach it to
    # this filing's own exhibit row(s), no separate fetch needed.
    text = ("119,891,418 shares of Common Stock Issued and Outstanding as of\n"
            "          March 1, 1997.\n"
            "<TYPE>EX-27\n<ARTICLE>5\n<PERIOD-TYPE>YEAR\n"
            "<FISCAL-YEAR-END>DEC-31-1996\n<TOTAL-ASSETS>100\n<NET-INCOME>10\n")
    filings = pd.DataFrame({
        "cik": [1], "form_type": ["10-K"],
        "date_filed": pd.to_datetime(["1997-03-05"]),
        "filename": ["cover.txt"],
    })
    with mock.patch.object(fds, "fetch_filing_text", lambda fn: text):
        result = fds.build_cik_history(1, filings)
    assert len(result) == 1
    assert result.iloc[0]["shares_outstanding"] == 119_891_418.0
    assert result.iloc[0]["shares_outstanding_asof"] == pd.Timestamp("1997-03-01")
    print("OK: build_cik_history attaches a cover-page shares_outstanding onto its own exhibit row")


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
    test_to_number_parses_parenthesized_negatives()
    test_extract_line_items_equity_survives_a_parenthesized_negative_component()
    test_missing_multiplier_borrows_from_sibling_exhibit()
    test_missing_multiplier_left_flagged_when_no_sibling_available()
    test_missing_multiplier_rejects_implausible_borrow()
    test_fill_missing_multipliers_canonical_reference_ignores_malformed_total_assets()
    test_infer_multiplier_from_trusted_tiers_never_touches_a_row_already_resolved_to_canonical_one()
    test_infer_multiplier_from_trusted_tiers_resolves_real_aeo_shape()
    test_infer_multiplier_from_trusted_tiers_rejects_no_close_candidate()
    test_infer_multiplier_from_trusted_tiers_respects_max_gap_days()
    test_build_cik_history_skips_post_ex27_era_filings()
    test_build_cik_history_drops_unparseable_period_end_instead_of_merging()
    test_measure_prevalence_handles_list_return_from_parse_fds()
    test_build_cik_history_produces_discrete_quarters_end_to_end()
    test_build_cik_history_attaches_shares_outstanding_from_cover_page()
