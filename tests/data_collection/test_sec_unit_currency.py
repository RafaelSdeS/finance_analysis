"""
test_sec_unit_currency.py
=========================
Spec: SEC XBRL monetary facts must be USD-denominated before they are read as
dollars.

`companyfacts._facts_to_frame` restricts the *shares*-denominated concepts to
the "shares" unit key, but deliberately leaves every dollar-denominated
concept on "any unit key" behaviour. Its own docstring records why:

    Deliberately NOT extended to "USD only" for the other (dollar-
    denominated) concepts -- whether every ifrs-full foreign filer's dollar
    facts are uniformly tagged "USD" is unverified, and the current "any unit
    key" behavior for those is unchanged/working

That assumption is false. Measured live against SEC companyfacts 2026-08-23:

    TM    us-gaap:StockholdersEquity  units -> ['JPY', 'USD']
    HDB   us-gaap:StockholdersEquity  units -> ['INR', 'USD']
    MUFG  us-gaap:StockholdersEquity  units -> ['JPY']          (no USD at all)
    SMFG  ifrs-full:Equity            units -> ['JPY']          (no USD at all)

Foreign private issuers tag under their reporting currency. `_facts_to_frame`
concatenates every unit key, so a JPY equity fact enters the pipeline as if it
were dollars and is then divided by a share count and compared against a USD
price. The damage is visible in the built panel (us_ml_dataset.parquet,
latest row per ticker):

    TM    close $191.11   vpa 2826.20  -> pvp 0.068   (Toyota book/share in JPY)
    SMFG  close  $26.32   vpa 4244.77  -> pvp 0.0062
    MUFG  close  $23.06   vpa 1649.28  -> pvp 0.014
    HDB   close  $23.37   vpa 1003.22  -> pvp 0.023

None of this is visible to test_unit_scale_invariants.py: every identity it
asserts (vpa*shares==equity, pvp*vpa==close, book_to_market*pvp==1) holds
exactly, because the same mis-denominated `equity` sits on both sides. All 21
US identity checks are green on these rows.

Synthetic fixtures only -- no network, FAST group.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection.sec import companyfacts  # noqa: E402


def _facts(concept: str, units: dict, taxonomy: str = "us-gaap") -> dict:
    """A minimal companyfacts payload: {unit_key: [fact, ...]}."""
    return {"facts": {taxonomy: {concept: {"units": {
        u: [{"start": "2023-01-01", "end": "2023-03-31", "val": v,
             "filed": "2023-05-01", "form": "10-Q", "accn": f"acc-{u}"}]
        for u, v in units.items()}}}}}


def test_usd_facts_are_kept():
    """The ordinary case must be untouched: a USD fact is read."""
    df = companyfacts._facts_to_frame(
        _facts("StockholdersEquity", {"USD": 1_000.0}), "StockholdersEquity")
    assert len(df) == 1, f"USD fact dropped: {df}"
    assert df["val"].iloc[0] == 1_000.0


def test_shares_concept_still_restricted_to_shares_unit():
    """Regression guard on the restriction that already exists."""
    df = companyfacts._facts_to_frame(
        _facts("EntityCommonStockSharesOutstanding",
               {"shares": 2.46e9, "USD": 2.46e15}, taxonomy="dei"),
        "EntityCommonStockSharesOutstanding")
    assert list(df["val"]) == [2.46e9], f"non-'shares' unit admitted: {list(df['val'])}"


def test_foreign_currency_fact_is_rejected():
    """A JPY-only equity fact must not be read as dollars.

    Real shape: MUFG / SMFG, whose StockholdersEquity / Equity carry a JPY
    unit key and no USD key at all. Correct behaviour is an empty frame (the
    ticker gets NaN fundamentals), NOT a yen figure silently priced in USD.
    """
    df = companyfacts._facts_to_frame(
        _facts("StockholdersEquity", {"JPY": 1.2e13}), "StockholdersEquity")
    assert df.empty, (
        f"JPY fact admitted as USD: val={list(df['val'])}. This is the MUFG/SMFG "
        "shape -- 12 trillion yen enters the panel as 12 trillion dollars, is "
        "divided by the share count into `vpa`, and is then compared against a "
        "USD close price.")


def test_usd_is_preferred_when_both_currencies_present():
    """TM / HDB shape: both a home-currency and a USD tagging of the same fact.

    Only the USD one may survive. Today both are concatenated and which one
    wins downstream is decided by dedup order, not by currency.
    """
    df = companyfacts._facts_to_frame(
        _facts("StockholdersEquity", {"JPY": 3.6e13, "USD": 2.4e11}),
        "StockholdersEquity")
    assert list(df["val"]) == [2.4e11], (
        f"expected the USD fact only, got {list(df['val'])} -- this is the TM "
        "shape (us-gaap:StockholdersEquity units ['JPY', 'USD']).")


def test_ifrs_foreign_currency_fact_is_rejected():
    """Same rule under the ifrs-full taxonomy (SMFG's Equity is JPY-only there)."""
    df = companyfacts._facts_to_frame(
        _facts("Equity", {"JPY": 1.1e13}, taxonomy="ifrs-full"), "Equity")
    assert df.empty, f"ifrs-full JPY fact admitted as USD: {list(df['val'])}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
