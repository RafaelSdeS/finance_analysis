"""refresh.py — one-command fast top-up for BR + US: macro, dividends, prices,
fundamentals, tail-only by default. BR's "macro" stage also refreshes
company_info/sectors/corporate_events (free CVM CAD + yfinance sources) --
folded in 2026-08-20 so this "one-command" path doesn't silently skip them
the way it used to (§6, DATA_LAYER_CORRECTNESS_PLAN.md); US has no
equivalent (CVM is BR-only).

Dividends and prices are now ONE pass, not two (2026-08-13): yfinance's
actions=True response already carries the Dividends/Stock Splits columns
alongside OHLCV, so collect_prices_yf(collect_dividends=True) extracts and
writes dividends straight from the SAME fetch collect_prices_yf's own tail
fetch makes for prices -- collect_dividends_yf's separate, dividends-only
request is no longer needed for a ticker with nothing new. Two-pass shape
per market: pass 1 is a tail-only fetch for EVERY ticker (full_refetch=set()),
which also reports which tickers had a new dividend/split THIS window
(collect_prices_yf's return value, the direct replacement for
collect_dividends_yf's old return value); pass 2 forces a full-span
re-fetch (price AND dividend history) for just that subset -- those are the
only tickers whose stored adj_close can actually be stale. `--full` skips
the two-pass split and does one full-span fetch for every ticker instead
(still just one request per ticker, folding dividends in for free).

`--only dividends` (dividends requested WITHOUT prices) is the one case with
no price fetch to fold detection into -- falls back to the old standalone
collect_dividends_yf call, kept only for that combination.

Reuses the existing per-source collectors and their checkpoints as-is (BR
mode "update", US mode "us_full_scale_v2") -- this module adds no new
collection logic, only orchestration + the US fundamentals delta (below).

Usage (from project root):
    python -m src.data_collection.refresh                    # both markets, all stages
    python -m src.data_collection.refresh --market us
    python -m src.data_collection.refresh --only macro prices
    python -m src.data_collection.refresh --full              # force full-span prices
    python -m src.data_collection.refresh --workers 12

    # Long runs: keep the machine awake (even unplugged) so a laptop suspend
    # doesn't freeze the run for hours mid-flight -- systemd-inhibit is native,
    # no wrapper script needed. Deliberate tradeoff: this blocks suspend on
    # battery too, trading the "auto-suspend saves a dying battery" safety net
    # for the run actually finishing; safe because every collector here is
    # checkpoint-resumable per ticker, so a hard power-off just stops the run,
    # it doesn't corrupt anything.
    systemd-inhibit --what=sleep:idle python -m src.data_collection.refresh
"""

import argparse
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import checkpoint, config
from .br import collectors as br_collectors
from .br.pipeline import _active_tickers, _collect, setup_logging
from .cvm import company_info as cvm_company_info
from .cvm import sectors as cvm_sectors
from .sec import crosswalk, fundamentals as sec_fundamentals, universe as sec_universe
from .us.fred_collectors import collect_macro_us
from .yf_collectors import collect_dividends_yf, collect_prices_yf, collect_splits_yf

log = logging.getLogger("refresh")

# 8 triggered a real Yahoo 429 storm on US prices (2026-08-12): each thread paces
# itself via YF_RATE_LIMIT_SLEEP independently, so the combined request rate scales
# with workers. 4 matches collect_dividends_yf's own conservative default; pass
# --workers explicitly for a larger backfill.
DEFAULT_WORKERS = 4
US_MODE = "us_full_scale_v2"  # reuse run_us_full_scale.py's checkpoints, not a fresh mode
ALL_STAGES = ["macro", "dividends", "prices", "fundamentals"]

# First-ever run has no "refresh" checkpoint to diff against -- look back this
# far rather than rebuilding every US company's fundamentals from scratch.
FUNDAMENTALS_LOOKBACK_DAYS = 90


def _us_tickers() -> list[str]:
    return sorted(p.stem for p in config.US_PRICES_DIR.glob("*.parquet"))


def _refresh_us_fundamentals(all_priced: list[str]) -> None:
    """Only companies with a new 10-K/10-Q filing since the last refresh --
    sec/universe.build_filings() is already incrementally cached per quarter
    (a rerun only re-fetches the current in-progress quarter), so this is
    cheap on a warm cache. Deliberately narrower than
    sec.fundamentals.collect_fundamentals_us's default "rebuild every
    currently-priced ticker", which exists for the different case of a
    derivation fix that must reach already-collected companies too.

    Deliberately does NOT forward this module's own `--workers` (tuned for
    Yahoo's throttle, see DEFAULT_WORKERS) into collect_fundamentals_us --
    SEC's real limit is enforced globally by sec/http.py's throttle lock
    regardless of thread count, so collect_fundamentals_us's own workers=8
    default (tuned for that lock) shouldn't be silently downgraded to a
    value picked for a different vendor's different throttle.
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
    sec_fundamentals.collect_fundamentals_us(delta)

    checkpoint.save("us_fundamentals_delta", "refresh",
                     {"last_run": datetime.now(timezone.utc).isoformat()})


def _refresh_prices_and_dividends(tickers: list[str], mode: str, stages: set[str], full: bool,
                                   workers: int, **price_kwargs) -> None:
    """Shared BR/US orchestration for the folded prices+dividends pass (see
    module docstring). `price_kwargs` forwards market-specific overrides
    (price_dir/dividend_dir/suffix/floor) straight to collect_prices_yf.
    """
    if "prices" not in stages:
        if "dividends" in stages:
            # No price fetch to fold detection into -- the one case that still
            # needs the old standalone collector.
            log.info("--- dividends (%d tickers) ---", len(tickers))
            dividend_dir = price_kwargs.get("dividend_dir")
            suffix = price_kwargs.get("suffix")
            floor = price_kwargs.get("floor")
            collect_dividends_yf(tickers, mode, dividend_dir=dividend_dir, suffix=suffix,
                                  floor=floor, workers=workers)
        return

    want_dividends = "dividends" in stages
    log.info("--- prices%s (%d tickers) ---", " + dividends" if want_dividends else "", len(tickers))
    if full:
        collect_prices_yf(tickers, mode, workers=workers, collect_dividends=want_dividends,
                           **price_kwargs)
        return

    changed = collect_prices_yf(tickers, mode, workers=workers, full_refetch=set(),
                                 collect_dividends=want_dividends, **price_kwargs) or set()
    if changed:
        log.info("--- prices full re-fetch for %d changed ticker(s) ---", len(changed))
        collect_prices_yf(sorted(changed), mode, workers=workers, full_refetch=changed,
                           collect_dividends=want_dividends, **price_kwargs)


def _refresh_br(stages: set[str], full: bool, workers: int) -> None:
    tickers = _active_tickers()
    if "macro" in stages:
        log.info("--- BR macro ---")
        br_collectors.collect_macro("update")

        # company_info/sectors/corporate_events are free (CVM CAD + yfinance) and
        # pipeline.py already runs them in every mode ("no BolsAI usage to ration") --
        # folded in under the same "macro" stage here so refresh.py's one-command
        # top-up doesn't skip them (previously only `pipeline.py --mode update` did,
        # so a status change or new split never reached this path -- §6,
        # DATA_LAYER_CORRECTNESS_PLAN.md).
        log.info("--- BR company_info ---")
        cvm_company_info.synthesize_company_info()
        log.info("--- BR sectors ---")
        cvm_sectors.build_sectors()
        log.info("--- BR corporate_events ---")
        all_tickers = sorted(set(tickers) | set(config.BENCHMARK_TICKERS))
        collect_splits_yf(all_tickers, "update")

    _refresh_prices_and_dividends(tickers, "update", stages, full, workers)

    if "fundamentals" in stages:
        log.info("--- BR fundamentals (%d tickers) ---", len(tickers))
        # Routed through pipeline._collect (config.DATA_SOURCE) rather than calling a
        # vendor collector directly -- this module used to hardcode collect_fundamentals_yf,
        # which silently bypassed the CVM rebuild (BUG-1: yfinance's BR financials are
        # wrong in level, not just thin -- see BOLSAI_EXIT_PLAN.md). One dispatcher, one
        # place to flip the source, instead of two independently-tracked switches.
        _collect("fundamentals", tickers, "update")


def _refresh_us(stages: set[str], full: bool, workers: int) -> None:
    tickers = _us_tickers()
    if "macro" in stages:
        log.info("--- US macro ---")
        collect_macro_us(US_MODE)

    _refresh_prices_and_dividends(tickers, US_MODE, stages, full, workers,
                                   price_dir=config.US_PRICES_DIR, dividend_dir=config.US_DIVIDENDS_DIR,
                                   suffix="", floor="1900-01-01")

    if "fundamentals" in stages:
        log.info("--- US fundamentals (delta) ---")
        _refresh_us_fundamentals(tickers)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--market", choices=["br", "us", "both"], default="both")
    p.add_argument("--only", nargs="+", choices=ALL_STAGES, default=ALL_STAGES,
                    help="stages to run (order is always macro, then prices+dividends folded "
                         "into one pass, then fundamentals)")
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
