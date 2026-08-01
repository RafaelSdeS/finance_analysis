"""
test_sec_companyfacts.py
=========================
Self-check for sec/companyfacts.py's pure logic (no network; synthetic XBRL
companyfacts-shaped dicts):

  - as_first_reported: EARLIEST filing wins per (start, end) -- point-in-time
    correctness, the whole reason this tier exists. Modeled on the real AAPL
    FY2008 NetIncomeLoss case: first filed 2009-10-27 at $4.834B, restated to
    $6.119B a filing later -- as_first_reported must return the FIRST value.
  - _quarterly_only: XBRL tags the same fiscal `end` with quarterly, half-year,
    and annual durations at once (confirmed on AAPL: 96 NetIncomeLoss periods
    share an `end` with a different-duration sibling) -- must keep only the
    ~90-day period, or line items collide when merged into one row per `end`.
  - _resolve_item: a REAL bug caught empirically while building this (not a
    hypothetical) -- the original code picked the first non-empty CONCEPT in
    a fallback list and used it for the WHOLE company's history. AAPL's
    revenue tag alone moved SalesRevenueNet (2008-2018) -> Revenues (2016-2018
    transition label, only 8 periods) -> RevenueFromContractWithCustomer...
    (2017-2026); the old code saw "Revenues" was non-empty and silently
    truncated coverage to those 8 periods, discarding 2008-2016 and
    2019-2026 entirely. _resolve_item must union per PERIOD, not per company.
  - extract_line_items: fundamentals_available_date = MAX of populated items'
    filed dates (conservative bundling -- never exposes a row before every
    item in it was genuinely public).
  - as_first_reported must drop facts with an implausibly ancient `end` date
    (2026-07-30, NG/CLSK/TENX): garbage placeholder XBRL contexts (always
    val=0) that predate any plausible fiscal period, unguarded for instant
    concepts since they have no `start`/duration to filter on at all.
  - _derive_q4: most 10-K filers never tag a standalone ~90-day Q4 duration --
    only the full fiscal year -- so _quarterly_only alone leaves every flow
    item (revenue, net income, ...) NaN at fiscal year-end. Confirmed on a
    120-ticker sample of the real collected dataset (2026-07-28): net_revenue
    NaN 22.9% overall, 58.7% of those NaN rows in December vs 26.4% of all
    rows. Fixed by deriving Q4 = FY total - (Q1+Q2+Q3), only when exactly 3
    quarters nest inside the FY window.

Usage: python tests/data_collection/test_sec_companyfacts.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_collection.sec import companyfacts


def _fact(start, end, val, filed, form="10-K", accn="0001-1"):
    d = {"end": end, "val": val, "filed": filed, "form": form, "accn": accn}
    if start is not None:
        d["start"] = start
    return d


def _facts(concept_facts: dict, taxonomy="us-gaap", unit="USD"):
    return {"facts": {taxonomy: {c: {"units": {unit: facts}} for c, facts in concept_facts.items()}}}


def _merge_facts(*facts_dicts):
    """Merge multiple _facts()-shaped dicts (e.g. different taxonomies/units in
    one fixture -- needed once shares_outstanding concepts genuinely require
    unit="shares" while other concepts in the same test still want unit="USD")."""
    merged = {"facts": {}}
    for fd in facts_dicts:
        for taxonomy, concepts in fd["facts"].items():
            merged["facts"].setdefault(taxonomy, {}).update(concepts)
    return merged


def test_as_first_reported_takes_earliest_filing():
    # Mirrors the real AAPL FY2008 case (values match exactly), restated a year later.
    # Quarterly (~90-day) duration -- as_first_reported now also applies _quarterly_only,
    # so an annual-duration fact here would be filtered out before reaching the dedup.
    facts = _facts({"NetIncomeLoss": [
        _fact("2008-06-29", "2008-09-27", 4_834_000_000, "2009-10-27"),
        _fact("2008-06-29", "2008-09-27", 6_119_000_000, "2010-10-27"),
    ]})
    df = companyfacts.as_first_reported(facts, "NetIncomeLoss")
    assert len(df) == 1
    assert df.iloc[0]["val"] == 4_834_000_000, "must keep the AS-FIRST-REPORTED value, not the restatement"
    assert str(df.iloc[0]["filed"].date()) == "2009-10-27"
    print("OK: as_first_reported keeps the earliest filing, not the latest restatement")


def test_quarterly_only_drops_annual_duplicate():
    # Same `end`, two durations: a genuine ~90-day quarter and a ~363-day annual figure.
    facts = _facts({"NetIncomeLoss": [
        _fact("2008-06-29", "2008-09-27", 1_000_000_000, "2009-10-27", form="10-Q"),  # ~90 days
        _fact("2007-09-30", "2008-09-27", 4_834_000_000, "2009-10-27", form="10-K"),  # ~363 days
    ]})
    df = companyfacts.as_first_reported(facts, "NetIncomeLoss")
    assert len(df) == 1
    assert df.iloc[0]["val"] == 1_000_000_000, "must keep the ~90-day quarterly figure, not the annual one"
    print("OK: _quarterly_only resolves same-end duration collisions to the quarterly figure")


def test_malformed_start_date_dropped_not_crashing():
    # Real bug, confirmed on MIND (CIK 926423, 2026-07-30): a raw XBRL fact's
    # `start` was literally "0202-02-01" (a year-digit typo in the SOURCE
    # filing) -- pandas can't represent that at nanosecond resolution and
    # used to raise OutOfBoundsDatetime uncaught, discarding this company's
    # ENTIRE fundamentals build (every tier), not just the one bad fact.
    facts = _facts({"NetIncomeLoss": [
        _fact("0202-02-01", "2008-09-27", 999_000_000, "2009-10-27"),  # malformed, must be dropped
        _fact("2008-06-29", "2008-09-27", 4_834_000_000, "2009-10-27"),  # genuine quarterly fact
    ]})
    df = companyfacts.as_first_reported(facts, "NetIncomeLoss")  # must not raise
    assert len(df) == 1, f"the malformed-date fact must be dropped, only the good one kept, got {len(df)} rows"
    assert df.iloc[0]["val"] == 4_834_000_000
    print("OK: a malformed start/end date is dropped (coerced to NaT), not a crash that loses the whole company")


def test_as_first_reported_drops_implausible_ancient_end_date():
    # Real bug, confirmed on NG/CLSK/TENX (2026-07-30): a small number of
    # small-cap filers' XBRL carries a genuine fact with an "end" date decades
    # before the company plausibly existed (e.g. NG: end=1984-12-04,
    # CLSK: end=1991-10-01, TENX: end=1967-05-25/08-25) -- checked directly
    # against SEC's own companyfacts API, every one carries val=0, a garbage
    # placeholder context from the filer's own XBRL-tooling, not real data.
    # Instant concepts (no "start") have no duration filter to catch this at
    # all; item6.py already has an equivalent year bound for its own version
    # of this failure shape, companyfacts.py never had one.
    facts = _facts({"Assets": [
        _fact(None, "1984-12-04", 0, "2014-02-12"),             # garbage placeholder context
        _fact(None, "2013-12-31", 500_000_000, "2014-02-12"),  # genuine
    ]})
    df = companyfacts.as_first_reported(facts, "Assets")
    assert len(df) == 1, f"the implausible ancient end date must be dropped, got {len(df)} rows"
    assert df.iloc[0]["val"] == 500_000_000
    print("OK: as_first_reported drops an implausibly ancient end date instead of treating it as real data")


def test_resolve_item_unions_across_concepts_per_period():
    # The real bug: old tag covers early periods, new tag covers later ones -- must NOT
    # pick one concept for the whole company just because it happens to be checked first.
    facts = _facts({
        "Revenues": [_fact("2016-01-01", "2016-03-31", 500.0, "2016-05-01")],           # transition-label, 1 period only
        "SalesRevenueNet": [_fact("2008-01-01", "2008-03-31", 100.0, "2008-05-01")],    # old tag, early history
        "RevenueFromContractWithCustomerExcludingAssessedTax":
            [_fact("2020-01-01", "2020-03-31", 900.0, "2020-05-01")],                  # new tag, later history
    })
    resolved = companyfacts._resolve_item(facts, companyfacts.CONCEPT_MAP["net_revenue"])
    ends = {d.strftime("%Y-%m-%d") for d in resolved["end"]}
    assert ends == {"2008-03-31", "2016-03-31", "2020-03-31"}, (
        "must cover ALL three eras, not truncate to whichever concept is non-empty first")
    print("OK: _resolve_item unions coverage across a filer's whole tag-history, not just one concept")


def test_resolve_item_priority_on_overlap():
    # Both concepts report the SAME end (transition-year overlap) -- earlier-priority wins.
    facts = _facts({
        "Revenues": [_fact("2017-01-01", "2017-03-31", 111.0, "2017-05-01")],
        "SalesRevenueNet": [_fact("2017-01-01", "2017-03-31", 222.0, "2017-05-01")],
    })
    resolved = companyfacts._resolve_item(facts, ["Revenues", "SalesRevenueNet"])
    assert len(resolved) == 1 and resolved.iloc[0]["val"] == 111.0, "first-priority concept wins on overlap"
    print("OK: _resolve_item breaks same-end overlap ties by fallback-list priority")


def test_extract_line_items_conservative_available_date():
    # equity known earlier (comparative in an earlier 10-Q); net_income only in the 10-K.
    # The row's bundled availability date must be the LATER of the two -- never early.
    facts = _facts({
        "NetIncomeLoss": [_fact("2008-06-29", "2008-09-27", 1_000_000_000, "2009-10-27")],
        "StockholdersEquity": [_fact(None, "2008-09-27", 21_030_000_000, "2009-07-22")],
    })
    li = companyfacts.extract_line_items(facts)
    assert len(li) == 1
    row = li.iloc[0]
    assert str(row["fundamentals_available_date"].date()) == "2009-10-27", (
        "must take the LATEST populated item's filed date, never the earliest")
    print("OK: extract_line_items bundles to the conservative (max) availability date")


def test_extract_line_items_clusters_nearby_period_ends():
    # Real bug, found scaling to a full company history (2026-07-28): different XBRL
    # concepts for the SAME fiscal quarter carry slightly different `end` dates.
    # Coca-Cola tags NetIncomeLoss's Q2 2008 as ending 2008-06-27 (a business-day
    # quarter end) while StockholdersEquity for "the same" quarter is 2008-06-28 --
    # one day apart. An exact-date merge fragments one real quarter into two
    # near-empty rows; this must instead collapse to ONE row.
    facts = _facts({
        "NetIncomeLoss": [_fact("2008-03-29", "2008-06-27", 1_422_000_000, "2008-07-25")],
        "StockholdersEquity": [_fact(None, "2008-06-28", 20_900_000_000, "2008-07-25")],
    })
    li = companyfacts.extract_line_items(facts)
    assert len(li) == 1, f"one real quarter must become ONE row, got {len(li)}"
    row = li.iloc[0]
    assert row["net_income"] == 1_422_000_000.0
    assert row["equity"] == 20_900_000_000.0
    print("OK: extract_line_items clusters near-identical period ends into one row")


def test_shares_outstanding_does_not_fragment_periods():
    # Real bug: shares_outstanding's dei concept (EntityCommonStockSharesOutstanding)
    # is "as of the filing's cover page date", NOT a fiscal period end -- confirmed
    # on Coca-Cola, where it floats ~3 weeks from the real quarter end. Treating it
    # as a period-anchor (the original bug) created a spurious extra row every
    # quarter; it must instead attach to the nearest real quarter via nearest-match.
    # unit="shares" (not the _facts() default "USD") -- confirmed against the real
    # live SEC API (AAPL, 2026-07-31) that shares_outstanding concepts are always
    # tagged under "shares"; _facts_to_frame now only accepts that unit for these
    # two concepts (see test_facts_to_frame_rejects_wrong_unit_key_for_shares_concepts).
    facts = _merge_facts(
        _facts({"NetIncomeLoss": [_fact("2009-04-04", "2009-07-03", 2_037_000_000, "2009-07-30")]}),
        _facts({"EntityCommonStockSharesOutstanding":
                [_fact(None, "2009-07-24", 2_313_000_000, "2009-07-30")]}, unit="shares"),
    )
    li = companyfacts.extract_line_items(facts)
    assert len(li) == 1, (
        f"shares_outstanding's cover-page date (21 days from the real quarter end) "
        f"must NOT create a second row, got {len(li)}")
    row = li.iloc[0]
    assert row["net_income"] == 2_037_000_000.0
    assert row["shares_outstanding"] == 2_313_000_000.0
    print("OK: shares_outstanding attaches via nearest-match instead of fragmenting periods")


def test_facts_to_frame_rejects_wrong_unit_key_for_shares_concepts():
    # Real bug: _facts_to_frame used to iterate ANY unit key with no check at
    # all -- a shares_outstanding fact mistakenly tagged under a non-"shares"
    # unit (here "USD", the _facts() default) would be silently admitted as if
    # it were a real share count.
    facts = _facts({"CommonStockSharesOutstanding": [
        _fact(None, "2020-12-31", 999_000_000_000, "2021-02-01"),
    ]}, unit="USD")
    df = companyfacts._facts_to_frame(facts, "CommonStockSharesOutstanding")
    assert df.empty, (
        f"a shares_outstanding fact tagged under a non-'shares' unit key must be dropped, got {len(df)} rows")
    print("OK: _facts_to_frame rejects a shares-concept fact tagged under the wrong unit key")


def test_reject_sequential_outliers_flags_isolated_bad_quarter():
    # Real bug, confirmed 2026-07-31: BTI's real shares_outstanding is
    # ~2.456 BILLION every fiscal year except FY2019, which reads
    # 2,456,520,738,000,000 -- ~1,000,000x too big, surrounded by otherwise-
    # normal values on both sides.
    df = pd.DataFrame({
        "end": pd.to_datetime(["2018-12-31", "2019-12-31", "2020-12-31", "2021-12-31"]),
        "shares_outstanding": [2_456_415_884, 2_456_520_738_000_000, 2_456_591_597, 2_456_617_788],
    })
    out = companyfacts._reject_sequential_outliers(df, "shares_outstanding")
    assert out["shares_outstanding_rejected_outlier"].tolist() == [False, True, False, False], (
        "only the isolated 1,000,000x-inflated quarter must be rejected")
    assert pd.isna(out.loc[1, "shares_outstanding"]), "a rejected value must become NaN, not a guessed number"
    assert out.loc[0, "shares_outstanding"] == 2_456_415_884, "surrounding normal values must be left untouched"
    print("OK: _reject_sequential_outliers flags an isolated bad quarter, leaves neighbors alone")


def test_reject_sequential_outliers_does_not_anchor_on_a_bad_first_value():
    # Real regression, found the hard way on a live recollection (2026-08-01):
    # CCI's and TFC's own FIRST-EVER XBRL-era shares_outstanding values
    # (2008-2009, when this tier begins) are themselves the corrupted ones --
    # confirmed on CCI: 288,464,431,000 (2008-12-31) and 290,792,627,000
    # (2009-06-30), then ~67 genuinely correct ~2.9e8-scale quarters follow.
    # A naive forward-only walk anchors on the bad first value and rejects
    # every good quarter that follows it (confirmed: turned CCI's real 2-row
    # bug into a 67-row wipeout on the actual recollected data). The seed must
    # come from the MAJORITY magnitude cluster (here, the 5 good quarters),
    # not index 0 -- walking backward from that seed then correctly flags the
    # 2 early bad values instead.
    df = pd.DataFrame({
        "end": pd.to_datetime(["2008-12-31", "2009-06-30", "2009-09-30", "2009-12-31",
                                "2010-03-31", "2010-06-30", "2010-09-30"]),
        "shares_outstanding": [288_464_431_000, 290_792_627_000,
                                289_000_000, 289_500_000, 290_100_000, 290_800_000, 291_200_000],
    })
    out = companyfacts._reject_sequential_outliers(df, "shares_outstanding")
    assert out["shares_outstanding_rejected_outlier"].tolist() == [
        True, True, False, False, False, False, False
    ], "the 2 bad FIRST-EVER values must be rejected, not the 5 good quarters that follow them"
    assert out.loc[:1, "shares_outstanding"].isna().all(), "both early bad values must become NaN"
    assert out.loc[2:, "shares_outstanding"].notna().all(), "all 5 genuinely good quarters must survive untouched"
    print("OK: _reject_sequential_outliers seeds on the majority cluster, "
          "not blindly the chronologically-first value")


def test_reject_sequential_outliers_does_not_reanchor_on_a_persistent_bad_run():
    # Real bug this guards against: LTM's shares_outstanding has been wrong for
    # 4 CONSECUTIVE fiscal years (2022-2025, ~1,000,000x too big), not just one
    # isolated quarter. A guard that compares only to the immediately PRECEDING
    # raw value would catch the first bad transition (2021->2022) but then treat
    # 2023/2024/2025 as "consistent" with 2022's already-bad value and silently
    # accept them. This must instead keep comparing against the last GOOD
    # (2021) value throughout, rejecting every one of the 4 bad years.
    df = pd.DataFrame({
        "end": pd.to_datetime(["2018-12-31", "2019-12-31", "2020-12-31", "2021-12-31",
                                "2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]),
        "shares_outstanding": [606_407_693, 606_407_693, 606_407_693, 606_407_693,
                                605_231_854_725, 604_437_877_587, 604_437_877_587, 574_215_983_709],
    })
    out = companyfacts._reject_sequential_outliers(df, "shares_outstanding")
    assert out["shares_outstanding_rejected_outlier"].tolist() == [
        False, False, False, False, True, True, True, True
    ], "all 4 persistently-bad years must be rejected, not just the first bad transition"
    assert out.loc[4:, "shares_outstanding"].isna().all(), "every rejected year must become NaN"
    assert out.loc[:3, "shares_outstanding"].notna().all(), "the 4 genuinely good years must survive untouched"
    print("OK: _reject_sequential_outliers never re-anchors on a rejected value -- "
          "a multi-year persistent bad run stays rejected throughout, not just its first transition")


def test_reject_sequential_outliers_accepts_a_plausible_large_jump():
    # A real stock split can legitimately multiply shares_outstanding by a large
    # factor in one period (e.g. a 10-for-1 split) -- must NOT be rejected as an
    # outlier. _MAX_PLAUSIBLE_RATIO (20x) is set well above real-world split
    # ratios and well below every one of the 27 real corruption cases (800x+).
    df = pd.DataFrame({
        "end": pd.to_datetime(["2019-12-31", "2020-12-31", "2021-12-31"]),
        "shares_outstanding": [100_000_000, 1_000_000_000, 1_010_000_000],  # a genuine 10-for-1 split
    })
    out = companyfacts._reject_sequential_outliers(df, "shares_outstanding")
    assert not out["shares_outstanding_rejected_outlier"].any(), (
        "a plausible 10x jump (well within a real stock split's range) must not be rejected")
    print("OK: _reject_sequential_outliers does not false-positive on a plausible large jump (e.g. a real split)")


def test_extract_line_items_rejects_outlier_shares_outstanding_end_to_end():
    # Wiring check: the outlier guard must actually run inside extract_line_items
    # (via the attached-items loop), not just exist as a standalone helper.
    facts = _merge_facts(
        _facts({"Assets": [
            _fact(None, "2018-12-31", 100.0, "2019-02-01"),
            _fact(None, "2019-12-31", 105.0, "2020-02-01"),
        ]}),
        _facts({"CommonStockSharesOutstanding": [
            _fact(None, "2018-12-31", 2_456_415_884, "2019-02-01"),
            _fact(None, "2019-12-31", 2_456_520_738_000_000, "2020-02-01"),  # ~1,000,000x too big
        ]}, unit="shares"),
    )
    li = companyfacts.extract_line_items(facts)
    assert len(li) == 2
    row_2019 = li[li["end"] == pd.Timestamp("2019-12-31")].iloc[0]
    assert pd.isna(row_2019["shares_outstanding"]), "the inflated 2019 value must be rejected end-to-end"
    assert bool(row_2019["shares_outstanding_rejected_outlier"]) is True
    row_2018 = li[li["end"] == pd.Timestamp("2018-12-31")].iloc[0]
    assert row_2018["shares_outstanding"] == 2_456_415_884, "the genuine 2018 value must survive untouched"
    print("OK: extract_line_items rejects an implausible shares_outstanding value end-to-end")


def test_extract_line_items_picks_up_ifrs_full_taxonomy():
    # Real gap, found auditing the top-500 collection run (2026-07-28): foreign
    # private issuers filing 20-F report under IFRS, tagged under a separate
    # "ifrs-full" taxonomy key that _facts_to_frame never checked at all --
    # confirmed on HSBC/RIO/TECK/SAN, each with 350-450 populated ifrs-full
    # concepts silently ignored. "Revenue"/"ProfitLoss"/"Assets" mirror the real
    # concept names these companies actually use (RIO/TECK/SAN all have "Revenue";
    # "ProfitLoss" and "Assets" are common to every one of the four, including
    # HSBC which lacks a plain "Revenue" tag, matching the same kind of
    # financial-sector gap already known for us-gaap banks/insurers).
    facts = _facts({
        "Revenue": [_fact("2023-01-01", "2023-12-31", 50_000_000_000.0, "2024-02-15")],
        "ProfitLoss": [_fact("2023-01-01", "2023-12-31", 5_000_000_000.0, "2024-02-15")],
        "Assets": [_fact(None, "2023-12-31", 300_000_000_000.0, "2024-02-15")],
    }, taxonomy="ifrs-full")
    li = companyfacts.extract_line_items(facts)
    assert len(li) == 1, f"expected 1 row from ifrs-full facts, got {len(li)}"
    row = li.iloc[0]
    assert row["net_revenue"] == 50_000_000_000.0
    assert row["net_income"] == 5_000_000_000.0
    assert row["total_assets"] == 300_000_000_000.0
    print("OK: extract_line_items picks up ifrs-full concepts for 20-F/foreign filers")


def test_derive_q4_fills_missing_fiscal_year_end():
    # Q1-Q3 tagged normally (~90-day durations); the FY is tagged only as one
    # full-year duration, never as a standalone Q4 -- the overwhelmingly common
    # real-world shape (10-Qs cover Q1-Q3, the 10-K's own duration IS the FY).
    facts = _facts({"NetIncomeLoss": [
        _fact("2020-01-01", "2020-03-31", 100.0, "2020-05-01"),
        _fact("2020-04-01", "2020-06-30", 110.0, "2020-08-01"),
        _fact("2020-07-01", "2020-09-30", 120.0, "2020-11-01"),
        _fact("2020-01-01", "2020-12-31", 460.0, "2021-02-01"),
    ]})
    quarterly = companyfacts._resolve_item(facts, ["NetIncomeLoss"])
    annual = companyfacts._resolve_item(facts, ["NetIncomeLoss"], annual=True)
    derived = companyfacts._derive_q4(quarterly, annual)
    assert len(derived) == 4, f"expected Q1-Q4, got {len(derived)} rows"
    q4 = derived[derived["end"] == pd.Timestamp("2020-12-31")].iloc[0]
    assert q4["val"] == 130.0, f"Q4 must be FY(460) - (100+110+120) = 130, got {q4['val']}"
    assert str(q4["filed"].date()) == "2021-02-01", "derived Q4 must inherit the FY total's own filed date"
    print("OK: _derive_q4 fills a missing standalone Q4 duration as FY total minus Q1+Q2+Q3")


def test_derive_q4_skips_when_quarters_incomplete():
    # Only 2 of the 3 needed quarters are present (a gap in the filer's own
    # history) -- must NOT guess at a Q4 value from an incomplete sum.
    facts = _facts({"NetIncomeLoss": [
        _fact("2020-01-01", "2020-03-31", 100.0, "2020-05-01"),
        _fact("2020-04-01", "2020-06-30", 110.0, "2020-08-01"),
        _fact("2020-01-01", "2020-12-31", 460.0, "2021-02-01"),
    ]})
    quarterly = companyfacts._resolve_item(facts, ["NetIncomeLoss"])
    annual = companyfacts._resolve_item(facts, ["NetIncomeLoss"], annual=True)
    derived = companyfacts._derive_q4(quarterly, annual)
    assert len(derived) == 2, "must leave Q4 undetermined (not guess) when fewer than 3 quarters are known"
    print("OK: _derive_q4 refuses to derive Q4 when the quarterly history has its own gaps")


def test_derive_q4_handles_a_quarterly_frame_with_no_start_column():
    # Real bug, confirmed on EPWKF (CIK 1900720, 2026-07-30): a concept can
    # have facts with no `start` at all (an instant-shaped tag used for a
    # nominally flow item). _quarterly_only/_annual_only already return such a
    # frame UNCHANGED rather than crash, but _derive_q4 didn't mirror that --
    # quarterly["start"] raised KeyError, discarding EPWKF's whole
    # fundamentals build (every tier) over this one concept.
    facts = _facts({"NetIncomeLoss": [_fact(None, "2020-03-31", 100.0, "2020-05-01")]})
    quarterly = companyfacts._resolve_item(facts, ["NetIncomeLoss"])
    annual = companyfacts._resolve_item(facts, ["NetIncomeLoss"], annual=True)
    assert "start" not in quarterly.columns, "test setup: fact has no start, so _facts_to_frame must not invent one"
    derived = companyfacts._derive_q4(quarterly, annual)  # must not raise
    assert len(derived) == len(quarterly), "a start-less frame must pass through unchanged, not crash"
    print("OK: _derive_q4 passes a start-less quarterly frame through unchanged instead of crashing")


def test_extract_line_items_derives_q4_and_clusters_with_instant_concepts():
    # End-to-end: a flow item (net_income, FY-only tagged) must get its Q4
    # derived AND cluster correctly against an instant concept (total_assets)
    # that DOES have a real Q4 data point -- the exact real-world shape found
    # auditing the collected dataset.
    facts = _facts({
        "NetIncomeLoss": [
            _fact("2020-01-01", "2020-03-31", 100.0, "2020-05-01"),
            _fact("2020-04-01", "2020-06-30", 110.0, "2020-08-01"),
            _fact("2020-07-01", "2020-09-30", 120.0, "2020-11-01"),
            _fact("2020-01-01", "2020-12-31", 460.0, "2021-02-01"),
        ],
        "Assets": [
            _fact(None, "2020-03-31", 1000.0, "2020-05-01"),
            _fact(None, "2020-06-30", 1050.0, "2020-08-01"),
            _fact(None, "2020-09-30", 1100.0, "2020-11-01"),
            _fact(None, "2020-12-31", 1200.0, "2021-02-01"),
        ],
    })
    li = companyfacts.extract_line_items(facts)
    assert len(li) == 4, f"expected 4 quarters, got {len(li)}"
    q4 = li[li["end"] == pd.Timestamp("2020-12-31")].iloc[0]
    assert q4["net_income"] == 130.0, "Q4 net_income must be derived, not left NaN"
    assert q4["total_assets"] == 1200.0, "derived Q4 row must still carry the real instant-concept value"
    print("OK: extract_line_items derives Q4 for flow items and clusters it with instant concepts")


if __name__ == "__main__":
    test_as_first_reported_takes_earliest_filing()
    test_quarterly_only_drops_annual_duplicate()
    test_malformed_start_date_dropped_not_crashing()
    test_derive_q4_handles_a_quarterly_frame_with_no_start_column()
    test_as_first_reported_drops_implausible_ancient_end_date()
    test_resolve_item_unions_across_concepts_per_period()
    test_resolve_item_priority_on_overlap()
    test_extract_line_items_conservative_available_date()
    test_extract_line_items_clusters_nearby_period_ends()
    test_shares_outstanding_does_not_fragment_periods()
    test_facts_to_frame_rejects_wrong_unit_key_for_shares_concepts()
    test_reject_sequential_outliers_flags_isolated_bad_quarter()
    test_reject_sequential_outliers_does_not_anchor_on_a_bad_first_value()
    test_reject_sequential_outliers_does_not_reanchor_on_a_persistent_bad_run()
    test_reject_sequential_outliers_accepts_a_plausible_large_jump()
    test_extract_line_items_rejects_outlier_shares_outstanding_end_to_end()
    test_extract_line_items_picks_up_ifrs_full_taxonomy()
    test_derive_q4_fills_missing_fiscal_year_end()
    test_derive_q4_skips_when_quarters_incomplete()
    test_extract_line_items_derives_q4_and_clusters_with_instant_concepts()
