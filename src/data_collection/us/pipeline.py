"""pipeline.py — orchestration + CLI for the US-equities collection pipeline
(Phase 6 of docs/US_EQUITIES_EXPANSION_PLAN.md: full ~10,432-ticker universe).

Same shape as br/pipeline.py (docs/DATA_LAYER_ORGANIZATION_PLAN.md §O2):
--mode/--tickers/--dry-run over the six run_* stage functions below. Renamed
from run_us_full_scale.py -- the exact invocation (mode name, ticker list,
path/suffix overrides) needs to survive a restart instead of being
reconstructed from checkpoint archaeology, as happened 2026-07-28 when this
run was stopped mid-flight to fix 5 real bugs (item6 unit scaling, missing
Q4 income-statement figures, universe.py's partial-quarter cache freeze,
fds.measure_prevalence, crosswalk.py's None-response crash -- see
docs/US_COLLECTOR_BUG_AUDIT.md).

Mode "us_full_scale_v2" (the default) resumes prices/dividends exactly where
the prior run left off (real incremental checkpoints, untouched by any of the
above fixes) -- pass --mode to run under a different, throwaway checkpoint
namespace (e.g. for a manual smoke test) without touching that state.
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
the same short window (hours). Never use it to resume a long-running backfill
spanning weeks/months -- that's exactly the staleness bug the full re-fetch
design fixed.

Usage (from project root):
    python -m src.data_collection.us.pipeline
    python -m src.data_collection.us.pipeline --steps prices dividends
    python -m src.data_collection.us.pipeline --tickers AAPL MSFT --dry-run
    RESUME=1 python -m src.data_collection.us.pipeline --steps prices fundamentals

("macro" (FRED) was missing from STEPS entirely until 2026-08-12, so no run
through this driver before that date ever collected US macro data.)
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd

from .. import config
from ..sec import company_info, crosswalk, fundamentals, universe
from ..yf.dividends import collect_dividends_yf
from ..yf.prices import collect_prices_yf
from .fred_collectors import collect_macro_us

log = logging.getLogger("us.pipeline")
MODE = "us_full_scale_v2"


def _all_tickers() -> list[str]:
    cw = pd.read_parquet(crosswalk.CROSSWALK_PATH) if crosswalk.CROSSWALK_PATH.exists() \
        else crosswalk.build_crosswalk_tier1()
    return cw["ticker"].tolist()


def run_universe():
    filings = universe.build_filings()
    universe.build_roster(filings)


def run_macro(mode: str = MODE):
    collect_macro_us(mode)


def run_prices(tickers: list[str] | None = None, mode: str = MODE):
    # workers=4: same value refresh.py/collect_dividends_yf already default to for this
    # exact vendor/endpoint (empirically safe; 8 triggered a real Yahoo 429 storm) --
    # this call used to omit it entirely, silently falling back to workers=1.
    collect_prices_yf(tickers or _all_tickers(), mode=mode, price_dir=config.US_PRICES_DIR,
                       suffix="", floor="1900-01-01", skip_existing=os.environ.get("RESUME") == "1",
                       workers=4)


def run_dividends(tickers: list[str] | None = None, mode: str = MODE):
    collect_dividends_yf(tickers or _all_tickers(), mode=mode, dividend_dir=config.US_DIVIDENDS_DIR,
                          suffix="", floor="1900-01-01")


def run_fundamentals(tickers: list[str] | None = None):
    t = tickers or sorted(p.stem for p in config.US_PRICES_DIR.glob("*.parquet"))
    fundamentals.collect_fundamentals_us(t, skip_existing=os.environ.get("RESUME") == "1")


def run_company_info(tickers: list[str] | None = None):
    company_info.collect_company_info(tickers or _all_tickers())


# Dependency order: universe/macro are ticker-independent; fundamentals reads
# back whatever prices already wrote to disk, so it must run after prices.
STEPS = {"universe": run_universe, "macro": run_macro, "prices": run_prices,
         "dividends": run_dividends, "fundamentals": run_fundamentals,
         "company_info": run_company_info}

# Steps that take (tickers, mode) vs (tickers,) vs () -- explicit here because
# the three shapes are real (macro and universe are ticker-independent by
# construction), not an accident worth papering over with a uniform signature.
_TICKERS_AND_MODE = {"prices", "dividends"}
_TICKERS_ONLY = {"fundamentals", "company_info"}
_MODE_ONLY = {"macro"}


def setup_logging():
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = config.LOG_DIR / f"us-collection-{datetime.now():%Y%m%d-%H%M%S}.log"
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logfile)],
    )
    return log


def run(steps: list[str], tickers: list[str] | None, mode: str, dry_run: bool = False) -> bool:
    if dry_run:
        log.info("DRY RUN | mode=%s | steps=%s", mode, steps)
        if tickers:
            log.info("tickers (override): %d -> %s", len(tickers),
                      tickers[:20] + (["..."] if len(tickers) > 20 else []))
        return True

    for step in steps:
        log.info("=== us.pipeline: %s ===", step)
        fn = STEPS[step]
        try:
            if step in _TICKERS_AND_MODE:
                fn(tickers, mode)
            elif step in _TICKERS_ONLY:
                fn(tickers)
            elif step in _MODE_ONLY:
                fn(mode)
            else:
                fn()
        except Exception as e:
            log.error("step %s failed: %s", step, e, exc_info=True)
            return False
    return True


def main():
    p = argparse.ArgumentParser(description="US-equities staged data collection pipeline")
    p.add_argument("--mode", default=MODE, help="checkpoint namespace (default: %(default)s)")
    p.add_argument("--tickers", nargs="+", help="override ticker universe for prices/dividends/"
                   "fundamentals/company_info (default: SEC crosswalk, or on-disk priced set "
                   "for fundamentals)")
    p.add_argument("--steps", nargs="+", choices=list(STEPS),
                    help="which steps to run, in order (default: all six)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    setup_logging()
    steps = args.steps or list(STEPS)
    tickers = [t.upper() for t in args.tickers] if args.tickers else None

    ok = run(steps, tickers, args.mode, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
