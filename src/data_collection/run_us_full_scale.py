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
Fundamentals has no such resume -- collect_fundamentals_us() always rebuilds
every currently-priced ticker from scratch, which is what's needed here since
the item6/Q4 fixes must reach every already-collected company, not just new
ones.

Usage: python -m src.data_collection.run_us_full_scale [prices|dividends|fundamentals|universe]
       (no argument runs all four, in order)
"""

import logging
import sys

import pandas as pd

from . import config
from .sec import crosswalk, fundamentals, universe
from .yf_collectors import collect_dividends_yf, collect_prices_yf

log = logging.getLogger(__name__)
MODE = "us_full_scale_v2"


def _all_tickers() -> list[str]:
    cw = pd.read_parquet(crosswalk.CROSSWALK_PATH) if crosswalk.CROSSWALK_PATH.exists() \
        else crosswalk.build_crosswalk_tier1()
    return cw["ticker"].tolist()


def run_universe():
    filings = universe.build_filings()
    universe.build_roster(filings)


def run_prices():
    collect_prices_yf(_all_tickers(), mode=MODE, price_dir=config.US_PRICES_DIR,
                       suffix="", floor="1900-01-01")


def run_dividends():
    collect_dividends_yf(_all_tickers(), mode=MODE, dividend_dir=config.US_DIVIDENDS_DIR,
                          suffix="", floor="1900-01-01")


def run_fundamentals():
    all_priced = sorted(p.stem for p in config.US_PRICES_DIR.glob("*.parquet"))
    fundamentals.collect_fundamentals_us(all_priced)


STEPS = {"universe": run_universe, "prices": run_prices,
         "dividends": run_dividends, "fundamentals": run_fundamentals}

if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(message)s")
    steps = sys.argv[1:] or list(STEPS)
    for step in steps:
        log.info("=== run_us_full_scale: %s ===", step)
        STEPS[step]()
