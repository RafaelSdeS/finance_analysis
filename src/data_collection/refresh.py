"""refresh.py — one-command fast top-up for BR + US: macro, dividends, prices,
fundamentals, tail-only by default.

Sequencing is deliberate: dividends run BEFORE prices in each market, because
the dividends pass reports which tickers had a new dividend or split
(collect_dividends_yf's return value) -- those are the only tickers whose
stored adj_close can actually be stale, so everyone else gets a cheap
tail-only price fetch (collect_prices_yf's `full_refetch` param) instead of
re-walking their entire history. `--full` forces the old full-span behavior
for every ticker.

Reuses the existing per-source collectors and their checkpoints as-is (BR
mode "update", US mode "us_full_scale_v2") -- this module adds no new
collection logic, only orchestration + the US fundamentals delta (below).

Usage (from project root):
    python -m src.data_collection.refresh                    # both markets, all stages
    python -m src.data_collection.refresh --market us
    python -m src.data_collection.refresh --only macro prices
    python -m src.data_collection.refresh --full              # force full-span prices
    python -m src.data_collection.refresh --workers 12
"""

import argparse
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import checkpoint, config
from .br import collectors as br_collectors
from .br.pipeline import _active_tickers, setup_logging
from .sec import crosswalk, fundamentals as sec_fundamentals, universe as sec_universe
from .us.fred_collectors import collect_macro_us
from .yf_collectors import collect_dividends_yf, collect_fundamentals_yf, collect_prices_yf

log = logging.getLogger("refresh")

DEFAULT_WORKERS = 8
US_MODE = "us_full_scale_v2"  # reuse run_us_full_scale.py's checkpoints, not a fresh mode
ALL_STAGES = ["macro", "dividends", "prices", "fundamentals"]

# First-ever run has no "refresh" checkpoint to diff against -- look back this
# far rather than rebuilding every US company's fundamentals from scratch.
FUNDAMENTALS_LOOKBACK_DAYS = 90


def _us_tickers() -> list[str]:
    return sorted(p.stem for p in config.US_PRICES_DIR.glob("*.parquet"))


def _refresh_us_fundamentals(all_priced: list[str], workers: int) -> None:
    """Only companies with a new 10-K/10-Q filing since the last refresh --
    sec/universe.build_filings() is already incrementally cached per quarter
    (a rerun only re-fetches the current in-progress quarter), so this is
    cheap on a warm cache. Deliberately narrower than
    sec.fundamentals.collect_fundamentals_us's default "rebuild every
    currently-priced ticker", which exists for the different case of a
    derivation fix that must reach already-collected companies too.
    """
    cp = checkpoint.load("us_fundamentals_delta", "refresh")
    since = cp.get("last_run")
    since = pd.Timestamp(since) if since else pd.Timestamp.now() - timedelta(days=FUNDAMENTALS_LOOKBACK_DAYS)

    filings = sec_universe.build_filings(progress=False)
    cw = pd.read_parquet(crosswalk.CROSSWALK_PATH) if crosswalk.CROSSWALK_PATH.exists() \
        else crosswalk.build_crosswalk_tier1()
    priced = set(all_priced)
    new_ciks = set(filings.loc[filings["date_filed"] > since, "cik"])
    delta = sorted(t for t, cik in zip(cw["ticker"], cw["cik"]) if cik in new_ciks and t in priced)

    if not delta:
        log.info("fundamentals us: no new filings since %s", since.date())
        return
    log.info("fundamentals us: %d ticker(s) with filings since %s", len(delta), since.date())
    sec_fundamentals.collect_fundamentals_us(delta, workers=workers)

    checkpoint.save("us_fundamentals_delta", "refresh",
                     {"last_run": datetime.now(timezone.utc).isoformat()})


def _refresh_br(stages: set[str], full: bool, workers: int) -> None:
    tickers = _active_tickers()
    if "macro" in stages:
        log.info("--- BR macro ---")
        br_collectors.collect_macro("update")

    changed: set[str] = set()
    if "dividends" in stages:
        log.info("--- BR dividends (%d tickers) ---", len(tickers))
        changed = collect_dividends_yf(tickers, "update", workers=workers)

    if "prices" in stages:
        log.info("--- BR prices (%d tickers) ---", len(tickers))
        collect_prices_yf(tickers, "update", workers=workers,
                           full_refetch=None if full else changed)

    if "fundamentals" in stages:
        log.info("--- BR fundamentals (%d tickers) ---", len(tickers))
        collect_fundamentals_yf(tickers, "update", workers=workers)


def _refresh_us(stages: set[str], full: bool, workers: int) -> None:
    tickers = _us_tickers()
    if "macro" in stages:
        log.info("--- US macro ---")
        collect_macro_us(US_MODE)

    changed: set[str] = set()
    if "dividends" in stages:
        log.info("--- US dividends (%d tickers) ---", len(tickers))
        changed = collect_dividends_yf(tickers, US_MODE, dividend_dir=config.US_DIVIDENDS_DIR,
                                        suffix="", floor="1900-01-01", workers=workers)

    if "prices" in stages:
        log.info("--- US prices (%d tickers) ---", len(tickers))
        collect_prices_yf(tickers, US_MODE, price_dir=config.US_PRICES_DIR, suffix="",
                           floor="1900-01-01", workers=workers,
                           full_refetch=None if full else changed)

    if "fundamentals" in stages:
        log.info("--- US fundamentals (delta) ---")
        _refresh_us_fundamentals(tickers, workers)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--market", choices=["br", "us", "both"], default="both")
    p.add_argument("--only", nargs="+", choices=ALL_STAGES, default=ALL_STAGES,
                    help="stages to run (order is always macro, dividends, prices, fundamentals)")
    p.add_argument("--full", action="store_true",
                    help="full-span price re-fetch for every ticker, not just ones with a new "
                         "dividend/split since last run (slow; the old always-on behavior)")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = p.parse_args()

    setup_logging()
    stages = set(args.only)

    if args.market in ("br", "both"):
        log.info("=" * 60)
        log.info("REFRESH: BR")
        log.info("=" * 60)
        _refresh_br(stages, args.full, args.workers)

    if args.market in ("us", "both"):
        log.info("=" * 60)
        log.info("REFRESH: US")
        log.info("=" * 60)
        _refresh_us(stages, args.full, args.workers)

    log.info("DONE.")


if __name__ == "__main__":
    main()
