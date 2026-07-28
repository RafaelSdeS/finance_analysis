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

Usage: python tests/data_collection/test_sec_companyfacts.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection.sec import companyfacts


def _fact(start, end, val, filed, form="10-K", accn="0001-1"):
    d = {"end": end, "val": val, "filed": filed, "form": form, "accn": accn}
    if start is not None:
        d["start"] = start
    return d


def _facts(concept_facts: dict, taxonomy="us-gaap", unit="USD"):
    return {"facts": {taxonomy: {c: {"units": {unit: facts}} for c, facts in concept_facts.items()}}}


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
    facts = _facts({
        "NetIncomeLoss": [_fact("2009-04-04", "2009-07-03", 2_037_000_000, "2009-07-30")],
        "EntityCommonStockSharesOutstanding": [_fact(None, "2009-07-24", 2_313_000_000, "2009-07-30")],
    })
    li = companyfacts.extract_line_items(facts)
    assert len(li) == 1, (
        f"shares_outstanding's cover-page date (21 days from the real quarter end) "
        f"must NOT create a second row, got {len(li)}")
    row = li.iloc[0]
    assert row["net_income"] == 2_037_000_000.0
    assert row["shares_outstanding"] == 2_313_000_000.0
    print("OK: shares_outstanding attaches via nearest-match instead of fragmenting periods")


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


if __name__ == "__main__":
    test_as_first_reported_takes_earliest_filing()
    test_quarterly_only_drops_annual_duplicate()
    test_resolve_item_unions_across_concepts_per_period()
    test_resolve_item_priority_on_overlap()
    test_extract_line_items_conservative_available_date()
    test_extract_line_items_clusters_nearby_period_ends()
    test_shares_outstanding_does_not_fragment_periods()
    test_extract_line_items_picks_up_ifrs_full_taxonomy()
