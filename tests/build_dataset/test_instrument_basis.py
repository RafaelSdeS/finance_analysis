"""
test_instrument_basis.py
========================
Spec: the price and the share count in a row must describe the SAME instrument,
and the resulting market cap must be physically possible.

Why this file exists as its own thing. test_unit_scale_invariants.py checks
seven algebraic identities per market:

    vpa*shares == equity · lpa*shares == net_income · market_cap == close*shares
    book_to_market*pvp == 1 · pl*lpa == close · pvp*vpa == close
    earnings_yield*pl == 1

Every one of them is a *closed* identity in `shares_outstanding` and `equity`:
whatever error those two carry appears on both sides and divides back out. All
45 checks (24 BR + 21 US) are green while the panel carries a company with 210
trillion shares and a R$645 trillion market cap. That is not a gap in the
tolerance band -- it is structural. This file checks the inputs those
identities cancel, using facts that come from *outside* the row.

Three independent defect families, all measured 2026-08-23 on the built
panels (BR dataset_v7, us_ml_dataset.parquet):

  1. SHARE-COUNT SCALE (BR).  cvm/shares.py reads FRE Quantidade_Total_Acoes
     verbatim; some filers report a scale-broken number. VSPT3 carries
     2.10e14 shares, CBEE3 3.92e12, TOYB3/4 1.32e12. Caught at the source by
     test_br_data_quality.py::test_reference_tables_clean; caught here where
     it actually lands. CEGR3 is why BOTH are needed -- its 2.60e11 shares sit
     INSIDE any plausible absolute band, and only the resulting R$1,194
     trillion market cap gives it away.

  2. UNIT / DEPOSITARY-RECEIPT BASIS (BR + US).  `close` is the price of the
     traded instrument; `shares_outstanding` counts the underlying ordinary
     shares. For a BR unit (SANB11 = 1 ON + 1 PN) or a US ADR (TSM = 1 ADR :
     5 ordinary) these are different things, and market_cap = close * shares
     over-counts by exactly the bundle size. Measured on BR, as the ratio of
     a unit ticker's pvp to its own ON/PN sibling's pvp over shared dates:

         SAPR11/SAPR4 5.01   TIET11/TIET4 5.01   TAEE11/TAEE3 3.00
         VVAR11/VVAR4 3.08   BIDI11/BIDI3 3.03   SANB11/SANB4 2.06

     Those are not noise -- they are the unit compositions (SAPR11 = 1 ON +
     4 PN, TAEE11 = 1 ON + 2 PN, SANB11 = 1 ON + 1 PN). Same company, same
     book, same quarter: the only difference is the share basis.

  3. REPORTING CURRENCY (US).  sec/companyfacts.py accepts a monetary fact
     under any XBRL unit key, so a foreign private issuer's JPY/KRW/INR
     equity is read as dollars -- see tests/data_collection/
     test_sec_unit_currency.py for the root-cause spec. It surfaces here as a
     book value per share tens to thousands of times the USD price.

DATA group: needs a built panel.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_dataset.paths import OUTPUT_PATH, US_OUTPUT_PATH  # noqa: E402

BR_PANEL = OUTPUT_PATH
US_PANEL = US_OUTPUT_PATH

# -- ceilings ---------------------------------------------------------------
# Largest market cap that has ever been real in each market, with headroom.
# BR: Petrobras' all-time peak is ~R$700bn; Vale ~R$450bn. R$2tn is ~3x the
# record and ~18% of Brazilian GDP -- nothing legitimate reaches it.
# Measured over the panel: 5 tickers above it (CEGR3 R$1,194tn, VSPT3 R$645tn,
# CBEE3 R$131tn, SANB11 R$8.2tn, TOYB3 R$3.6tn).
_BR_MAX_MARKET_CAP = 2e12
# US: NVDA legitimately reaches $5.73tn inside this panel (2026), so the
# ceiling has to clear it. $8tn is ~1.4x the real record. Measured above it:
# TSM $12.4tn, BCH $9.8tn -- both ADRs priced against ordinary-share counts.
# Deliberately loose: the ADR defect is caught properly by
# test_book_value_currency_basis + the companyfacts unit spec. This is the
# backstop for the physically-impossible tail only.
_US_MAX_MARKET_CAP = 8e12

# A unit's pvp must match its ON/PN sibling's. ON and PN prices legitimately
# diverge (a 10-30% class spread is ordinary), so the band is generous; the
# smallest real bundle is 2x, and the smallest measured offender is 2.06x, so
# 1.5 separates cleanly with margin on both sides.
_MAX_SIBLING_PVP_RATIO = 1.5
_MIN_SHARED_DATES = 200

# Book value per share vs. the traded price. A sustained median of 20 means a
# price-to-book of 0.05 held for half a company's life, which does not happen;
# it means the two numbers are in different currencies (or the share count is
# broken). Measured: 35 US tickers and 15 BR tickers above 20.
_MAX_BOOK_TO_PRICE = 20.0

# Absolute plausibility band for a share count as it lands in the panel. Same
# band test_br_data_quality.py applies to cvm/shares.parquet at the source, so
# a BR failure here that is green there means the corruption was introduced
# BETWEEN the source table and the panel (cvm/ratios.py's split/event
# adjustment), not read in from CVM. The US half has no source-side analogue
# at all -- sec/companyfacts.py restricts shares concepts to the "shares" unit
# key but never range-checks the value.
# Measured 2026-08-23: BR 8,543 rows / 13 tickers; US 479 rows / 8 tickers
# (BRUN, CHWY, FNKO, FOX, FOXA, GLXY, PLNT, SRG -- the multi-class Class A/B
# shape, and the same names FOX/FOXA/FLWS/AOS that top the book/price screen).
_SHARES_MIN, _SHARES_MAX = 1_000, 1e12

_COLS = ["ticker", "trade_date", "close", "shares_outstanding", "market_cap", "vpa", "pvp"]


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path, columns=_COLS)


def _report(market: str, offenders: pd.Series, unit: str) -> None:
    print(f"  {market}: {len(offenders)} offending tickers")
    for tk, v in offenders.sort_values(ascending=False).head(10).items():
        print(f"      {tk:8s} {v:,.4g} {unit}")


@pytest.mark.parametrize("market,path,ceiling", [
    ("BR", BR_PANEL, _BR_MAX_MARKET_CAP),
    ("US", US_PANEL, _US_MAX_MARKET_CAP),
])
def test_market_cap_physically_possible(market, path, ceiling):
    """No company may be worth more than the ceiling above.

    This is the one check that catches CEGR3, whose share count is inside
    every plausible absolute band and whose seven unit-scale identities are
    all green -- the impossibility only exists once the price is applied.
    """
    df = _load(path)
    if df is None:
        pytest.skip(f"{market} panel not built: {path}")
    peak = df.groupby("ticker")["market_cap"].max().dropna()
    bad = peak[peak > ceiling]
    _report(market, bad, "peak market cap")
    assert bad.empty, (
        f"{market}: {len(bad)} tickers exceed the {ceiling:.0e} market-cap ceiling: "
        f"{sorted(bad.index)}")


def test_unit_ticker_share_basis():
    """A BR unit ticker must be valued on units, not on underlying shares.

    Compares each `*11` ticker's pvp against a same-company ON/PN sibling on
    shared dates. Identical company, identical book value, identical quarter --
    a persistent ratio is the unit bundle size leaking into the valuation.
    """
    df = _load(BR_PANEL)
    if df is None:
        pytest.skip(f"BR panel not built: {BR_PANEL}")

    units = sorted(t for t in df["ticker"].unique() if t.endswith("11"))
    pvp = {t: g.set_index("trade_date")["pvp"] for t, g in df.groupby("ticker")}
    offenders, checked, skipped = {}, 0, []
    for u in units:
        sibs = [t for t in units_siblings(df, u) if t in pvp]
        if not sibs:
            skipped.append(u)
            continue
        for s in sibs:
            joined = pd.concat({"u": pvp[u], "s": pvp[s]}, axis=1).dropna()
            joined = joined[joined["s"] != 0]
            if len(joined) < _MIN_SHARED_DATES:
                continue
            checked += 1
            ratio = float((joined["u"] / joined["s"]).median())
            if not (1 / _MAX_SIBLING_PVP_RATIO) <= ratio <= _MAX_SIBLING_PVP_RATIO:
                offenders[f"{u} vs {s}"] = ratio

    print(f"  BR: {len(units)} unit-style tickers, {checked} sibling pairs comparable, "
          f"{len(skipped)} unit tickers with no comparable ON/PN sibling "
          f"(unchecked: {skipped})")
    for pair, r in sorted(offenders.items(), key=lambda kv: -kv[1]):
        print(f"      {pair:20s} pvp ratio {r:.2f}x")
    assert not offenders, (
        f"BR: {len(offenders)} unit/sibling pairs disagree on valuation by more than "
        f"{_MAX_SIBLING_PVP_RATIO}x -- the unit price is being divided by the "
        f"underlying share count: {offenders}")


def units_siblings(df: pd.DataFrame, unit_ticker: str) -> list[str]:
    """Same 4-letter root, ON (`3`) or PN (`4`) class, present in the panel."""
    root = unit_ticker[:4]
    return sorted(t for t in df["ticker"].unique()
                  if t != unit_ticker and t.startswith(root) and t[-1] in "34")


@pytest.mark.parametrize("market,path", [("BR", BR_PANEL), ("US", US_PANEL)])
def test_book_value_currency_basis(market, path):
    """Book value per share and the close price must be the same currency.

    US offenders read as a currency roster, not a value screen: PKX/KEP/KT/
    LPL/SKM/SHG/KB (KRW), SMFG/NMR/MUFG/MFG/SONY/HMC/TM (JPY), CCU/EC/BBAR/BMA
    (CLP/COP/ARS), BBD (BRL), KSPI (KZT). FOX/FOXA (2.4e8) and FLWS/AOS (~8e3)
    are US-domestic and therefore a *different* defect on the same axis -- a
    broken share count, not a currency (both appear in the panel's
    shares_outstanding < 1000 population).
    """
    df = _load(path)
    if df is None:
        pytest.skip(f"{market} panel not built: {path}")
    d = df.dropna(subset=["vpa", "close"])
    d = d[(d["vpa"] > 0) & (d["close"] > 0)]
    ratio = (d.assign(r=d["vpa"] / d["close"])
              .groupby("ticker")["r"].median())
    bad = ratio[ratio > _MAX_BOOK_TO_PRICE]
    _report(market, bad, "x book/share per unit of price")
    assert bad.empty, (
        f"{market}: {len(bad)} tickers carry a book value per share more than "
        f"{_MAX_BOOK_TO_PRICE}x the traded price: {sorted(bad.index)}")


@pytest.mark.parametrize("market,path", [("BR", BR_PANEL), ("US", US_PANEL)])
def test_share_count_plausible(market, path):
    """A share count in the panel must be physically possible.

    Distinct from the market-cap ceiling: a count can be absurd without the
    resulting cap clearing the ceiling (a broken count on a penny stock), and
    a cap can be absurd on a count inside the band (CEGR3). Distinct from the
    source-table check in test_br_data_quality.py: this is where the value
    actually gets used, after every transformation between the two.
    """
    df = _load(path)
    if df is None:
        pytest.skip(f"{market} panel not built: {path}")
    s = df["shares_outstanding"].dropna()
    bad_rows = int((~s.between(_SHARES_MIN, _SHARES_MAX)).sum())
    per_ticker = df.dropna(subset=["shares_outstanding"]).groupby("ticker")["shares_outstanding"]
    bad = per_ticker.apply(lambda g: not g.between(_SHARES_MIN, _SHARES_MAX).all())
    bad = bad[bad]
    print(f"  {market}: {bad_rows:,} rows outside [{_SHARES_MIN:,}, {_SHARES_MAX:.0e}] "
          f"across {len(bad)} tickers -> {sorted(bad.index)}")
    assert bad_rows == 0, (
        f"{market}: {bad_rows:,} rows carry an impossible share count across "
        f"{len(bad)} tickers: {sorted(bad.index)}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
