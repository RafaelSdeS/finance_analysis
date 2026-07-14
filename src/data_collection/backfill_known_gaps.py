"""
backfill_known_gaps.py — one-off historical backfill for confirmed BolsAI
vendor data gaps in data/raw/prices/.

Background: the anomaly-report investigation (2026-07-14, see
ANOMALY_INVESTIGATION.md) found 44 ATIVO-status tickers with a >400-day void
in their raw BolsAI price history. A first pass spot-checked each against
yfinance by counting rows returned for the same window and treated a dense
row count as proof of real trading — WRONG: a full first run of this script
found yfinance itself pads holes in its own coverage with a carried-forward
stale price for many of these tickers, producing a dense, correctly-dated
row count that's actually 90%+ a single repeated close. 24 tickers were
silently corrupted this way and had to be reverted from data/raw/prices/ by
hand; 2 more (EUCA4, NUTR3) were ~50% contaminated and reverted out of an
abundance of caution. backfill_price_gap() now has a flat-run guard
(_flat_run_fraction in yf_collectors.py) that rejects any fetch matching
this signature before it can reach disk, so GAPS below only lists tickers
CONFIRMED clean by that guard (i.e. already successfully filled once it
existed). The corrupted/ambiguous ones are listed separately in
FLAT_RUN_PADDING for the record — do not retry them from yfinance.

Each entry below is (ticker, gap_start, gap_end) as directly observed in the
raw file (the trade_date immediately before, and immediately after, the
gap). Dates are NOT tight bounds — backfill_price_gap() only ever writes
dates that are genuinely absent from the existing file (see its docstring
in yf_collectors.py), so padding here is harmless.

Run from project root: python -m src.data_collection.backfill_known_gaps
"""

import logging

from .yf_collectors import backfill_price_gap

log = logging.getLogger(__name__)

# ticker, gap_start (day after last known row), gap_end (day before next known row)
# Confirmed clean by the flat-run guard (already filled; re-running is a
# harmless no-op since backfill_price_gap only ever writes missing dates).
GAPS = [
    ("ENEV3", "2014-12-10", "2016-07-01"),
    ("LUPA3", "2014-02-13", "2023-03-15"),
    ("INEP3", "2014-08-29", "2022-11-18"),
    ("MWET4", "2016-02-01", "2022-10-20"),
    ("FIGE3", "2016-03-01", "2022-08-22"),
    ("ETER3", "2018-03-20", "2024-08-12"),
    ("RNEW4", "2019-10-16", "2025-02-14"),
    ("VIVR3", "2016-09-16", "2021-08-04"),
    ("PDGR3", "2017-02-22", "2021-10-15"),
    ("REDE3", "2012-11-22", "2016-09-08"),
    ("MGEL4", "2013-11-01", "2017-03-17"),
    ("FHER3", "2019-02-05", "2022-03-25"),
    ("TPIS3", "2017-07-24", "2020-01-24"),
    # SNSY5/RCSL4: real, varying yfinance data (not flat-run padding), but
    # rejected by validate_prices for a handful of rows with open/close a
    # hair outside [low, high] -- genuine yfinance OHLC noise on these two
    # heavily-split, thinly-traded names. Left in so a future OHLC-bracket
    # repair (mirroring _repair_nonpositive_ohlc) can retry them; harmless
    # to attempt as-is, they'll just fail validation again.
    ("SNSY5", "2005-12-19", "2020-12-11"),
    ("RCSL4", "2006-01-26", "2008-12-29"),
]

# Spot-checked but yfinance has zero rows for the gap window either — not a
# BolsAI-only problem, so this script can't fix them. Candidates for a
# targeted BolsAI re-fetch of just this window, not attempted here since it
# costs API credits and hasn't been diagnosed.
NO_YFINANCE_COVERAGE = ["FRAS3", "BAUH4", "PEAB4", "BMIN4"]

# yfinance ALSO doesn't have real data for these -- it silently pads its own
# coverage hole with a carried-forward stale price instead. Confirmed
# directly against yfinance's raw feed with zero transformation applied.
# Do not retry: the flat-run guard will correctly reject all of these again,
# there is no fix available from this vendor for this window.
FLAT_RUN_PADDING = [
    "LREN3", "UGPA3", "TEND3", "CALI3", "BSLI3", "AHEB3", "VULC3", "LEVE3",
    "NORD3", "AZEV4", "MSPA3", "PCAR3", "MNPR3", "RSUL4", "BIOM3", "JOPA3",
    "CEGR3", "PATI3", "LUXM4", "EALT4", "BALM4", "HETA4", "BPAR3",
    "EUCA4", "NUTR3",
]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    filled, rejected = [], []
    for ticker, gap_start, gap_end in GAPS:
        try:
            saved = backfill_price_gap(ticker, gap_start, gap_end)
            (filled if saved is not None else rejected).append(ticker)
        except Exception as e:
            log.error("backfill %s: unexpected error: %s", ticker, e)
            rejected.append(ticker)

    print(f"\nDone. filled={len(filled)} rejected_or_already_done={len(rejected)}")
    print("(rejected_or_already_done = no missing dates left, flat-run guard "
          "fired, or validation failed — see the log lines above for which, "
          "per ticker)")
    if rejected:
        print("Rejected/already-done:", rejected)
    print(f"Not attempted, no yfinance coverage: {NO_YFINANCE_COVERAGE}")
    print(f"Not attempted, yfinance confirmed flat-run padding (see docstring): {FLAT_RUN_PADDING}")


if __name__ == "__main__":
    main()
