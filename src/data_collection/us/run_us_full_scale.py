"""run_us_full_scale.py — driver for the US-equities full-scale collection
(Phase 6 of docs/US_EQUITIES_EXPANSION_PLAN.md: full ~10,432-ticker universe).

No CLI wiring exists for the US path in pipeline.py yet (BR-only there) --
this is the minimal entry point so the exact invocation (mode name, ticker
list, path/suffix overrides) survives a restart instead of having to be
reconstructed from checkpoint archaeology, as happened 2026-07-28 when this
run was stopped mid-flight to fix 5 real bugs (item6 unit scaling, missing
Q4 income-statement figures, universe.py's partial-quarter cache freeze,
fds.measure_prevalence, crosswalk.py's None-response crash -- see
docs/US_COLLECTOR_BUG_AUDIT.md).

Mode "us_full_scale_v2" resumes prices/dividends exactly where the prior run
left off (real incremental checkpoints, untouched by any of the above fixes).
Fundamentals has no such resume by default -- collect_fundamentals_us() rebuilds
every currently-priced ticker from scratch, which is what's needed after a
derivation fix (item6/Q4, a CONCEPT_MAP addition) that must reach every
already-collected company, not just new ones.

Set RESUME=1 to instead pass skip_existing=True to fundamentals AND prices --
skips a ticker outright if its output file already exists. For fundamentals,
two real uses: resuming after a crash/kill with no derivation change since,
and running fundamentals CONCURRENTLY with a separate `prices` run (they hit
different services -- yfinance vs SEC EDGAR -- so there's no rate-limit
conflict) without redoing tickers a prior pass already covered. For prices,
this is narrower still -- a normal run always re-fetches every ticker's
ENTIRE span on purpose (a dividend paid after collection needs its whole
history's adj_close revisited, see collect_prices_yf's docstring), so
RESUME=1 is only safe for resuming an interrupted FIRST-TIME backfill within
the same short window (hours). Never use it to resume a --mode update run
spanning weeks/months -- that's exactly the staleness bug the full re-fetch
design fixed.

Usage: python -m src.data_collection.us.run_us_full_scale [prices|dividends|fundamentals|universe|macro|company_info]
       (no argument runs all six, in order -- "macro" (FRED) was missing from
       STEPS entirely until 2026-08-12, so no run through this driver ever
       collected US macro data)
       RESUME=1 python -m src.data_collection.us.run_us_full_scale prices fundamentals
"""

import logging
import os
import sys

import pandas as pd

from .. import config
from ..sec import company_info, crosswalk, fundamentals, universe
from ..yf_collectors import collect_dividends_yf, collect_prices_yf
from .fred_collectors import collect_macro_us

log = logging.getLogger(__name__)
MODE = "us_full_scale_v2"


def _all_tickers() -> list[str]:
    cw = pd.read_parquet(crosswalk.CROSSWALK_PATH) if crosswalk.CROSSWALK_PATH.exists() \
        else crosswalk.build_crosswalk_tier1()
    return cw["ticker"].tolist()


def run_universe():
    filings = universe.build_filings()
    universe.build_roster(filings)


def run_macro():
    collect_macro_us(MODE)


def run_prices():
    # workers=4: same value refresh.py/collect_dividends_yf already default to for this
    # exact vendor/endpoint (empirically safe; 8 triggered a real Yahoo 429 storm) --
    # this call used to omit it entirely, silently falling back to workers=1.
    collect_prices_yf(_all_tickers(), mode=MODE, price_dir=config.US_PRICES_DIR,
                       suffix="", floor="1900-01-01", skip_existing=os.environ.get("RESUME") == "1",
                       workers=4)


def run_dividends():
    collect_dividends_yf(_all_tickers(), mode=MODE, dividend_dir=config.US_DIVIDENDS_DIR,
                          suffix="", floor="1900-01-01")


def run_fundamentals():
    all_priced = sorted(p.stem for p in config.US_PRICES_DIR.glob("*.parquet"))
    fundamentals.collect_fundamentals_us(all_priced, skip_existing=os.environ.get("RESUME") == "1")


def run_company_info():
    company_info.collect_company_info(_all_tickers())


STEPS = {"universe": run_universe, "macro": run_macro, "prices": run_prices,
         "dividends": run_dividends, "fundamentals": run_fundamentals,
         "company_info": run_company_info}

if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(message)s")
    steps = sys.argv[1:] or list(STEPS)
    for step in steps:
        log.info("=== run_us_full_scale: %s ===", step)
        STEPS[step]()
