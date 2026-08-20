"""
pipeline.py — orchestration + CLI for the staged data collection pipeline.

Same code runs prototype and full-scale; only the ticker list and checkpoint
dir change (prototype↔production parity). Stages run in dependency order;
each collector is resumable via its checkpoint.

Usage (from project root):
    python -m src.data_collection.br.pipeline --mode prototype
    python -m src.data_collection.br.pipeline --mode full_scale
    python -m src.data_collection.br.pipeline --mode full_scale --dry-run
    python -m src.data_collection.br.pipeline --mode prototype --tickers PETR4 VALE3
"""

import argparse
import logging
import sys
from datetime import datetime

import pandas as pd

from . import collectors
from .. import config, yf_collectors
from ..cvm import company_info as cvm_company_info
from ..cvm import ratios as cvm_ratios
from ..cvm import sectors as cvm_sectors


def _collect(name: str, tickers: list[str], mode: str):
    """Per-data-type source switch, from config.DATA_SOURCE -- governs every mode
    alike (full_scale/prototype/update) now that the free sources (yfinance prices/
    dividends, CVM fundamentals) match or exceed BolsAI's own depth (see
    BOLSAI_EXIT_PLAN.md). Flip a DATA_SOURCE entry to "bolsai" to opt back into the
    paid path for that data type, in any mode.

    Special handling: YFINANCE_ONLY_TICKERS (e.g. BOVA11) always use yfinance.
    """
    # Split tickers: yfinance-only vs global data source
    yf_only = [t for t in tickers if t in config.YFINANCE_ONLY_TICKERS]
    others = [t for t in tickers if t not in config.YFINANCE_ONLY_TICKERS]

    fn_map = {
        ("prices", "bolsai"): collectors.collect_prices,
        ("prices", "yfinance"): yf_collectors.collect_prices_yf,
        ("fundamentals", "bolsai"): collectors.collect_fundamentals,
        ("fundamentals", "yfinance"): yf_collectors.collect_fundamentals_yf,
        ("fundamentals", "cvm"): cvm_ratios.collect_fundamentals_cvm,
        ("dividends", "bolsai"): collectors.collect_dividends,
        ("dividends", "yfinance"): yf_collectors.collect_dividends_yf,
    }

    if others:
        source = config.DATA_SOURCE.get(name, "bolsai")
        fn = fn_map[(name, source)]
        fn(others, mode)

    # Collect yfinance-only tickers via yfinance
    if yf_only:
        fn = fn_map[(name, "yfinance")]
        fn(yf_only, mode)


def _active_tickers() -> list[str]:
    """Return only tickers with status='ATIVO' (exclude delisted/suspended)."""
    path = config.COMPANY_INFO_PATH
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    return sorted(df[df["status"] == "ATIVO"]["ticker"].dropna().unique().tolist())


def _recover_stale_company_info_tickers(requested: list[str], prices_tickers: set[str]) -> set[str]:
    """--mode update's price collection is yfinance-only (free, no BolsAI dependency --
    collect_prices_yf needs only a valid ticker symbol, nothing from company_info).
    Gating it on BolsAI's company_info ATIVO status is unnecessarily restrictive:
    BolsAI's own /companies/ registry is smaller than this repo's tracked universe
    (523 tickers per CLAUDE.md) and structurally omits some real, still-trading
    distressed/judicial-recovery names -- confirmed 2026-08-16 (AMER3/Americanas,
    LIGT3, and 13 others): re-running collect_company_info() found zero of them in a
    fresh BolsAI pull, ruling out "stale collection" and confirming a genuine vendor
    coverage gap, not a staleness bug.

    A ticker already on disk (collected once, however long ago) is a real,
    previously-confirmed equity -- keep refreshing its free yfinance price series
    even after BolsAI's own registry loses track of it. Tickers with NO existing
    file are deliberately NOT recovered here (that's collect_delisted.py's job,
    which already bypasses this same ATIVO gate for exactly that case, per its own
    docstring).
    """
    on_disk = {p.stem for p in config.PRICES_DIR.glob("*.parquet")}
    return (set(requested) & on_disk) - prices_tickers


def setup_logging():
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = config.LOG_DIR / f"collection-{datetime.now():%Y%m%d-%H%M%S}.log"
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logfile)],
    )
    return logging.getLogger("pipeline")


def run(mode: str, tickers: list[str], dry_run: bool = False):
    log = logging.getLogger("pipeline")

    # DATA_SOURCE now governs every mode (see _collect) -- only warn, don't hard-fail,
    # since a missing key only breaks the specific data type(s) actually routed to
    # "bolsai"; every default entry is free/keyless (BOLSAI_EXIT_PLAN.md Task 5).
    needs_bolsai = any(
        config.DATA_SOURCE.get(k) == "bolsai" for k in ("prices", "fundamentals", "dividends")
    )
    if needs_bolsai and not config.BOLSAI_API_KEY:
        log.warning("BOLSAI_API_KEY not set (add it to .env) — data type(s) explicitly "
                     "routed to bolsai in DATA_SOURCE will fail")

    # Always append benchmark tickers (prices only, for performance comparison)
    all_tickers = sorted(set(tickers) | set(config.BENCHMARK_TICKERS))

    if dry_run:
        log.info("DRY RUN | mode=%s | %d tickers (+%d benchmarks)", mode, len(tickers), len(config.BENCHMARK_TICKERS))
        log.info("tickers: %s", all_tickers[:20] + (["..."] if len(all_tickers) > 20 else []))
        stage_names = ["macro", "company_info", "sectors", "corporate_events", "prices", "fundamentals", "dividends"]
        log.info("would run: %s (source: %s)", ", ".join(stage_names), config.DATA_SOURCE)
        return True

    log.info("=" * 60)
    log.info("DATA COLLECTION | mode=%s | %d tickers (+%d benchmarks)", mode, len(tickers), len(config.BENCHMARK_TICKERS))
    log.info("=" * 60)

    # (name, callable) in dependency order: macro is ticker-independent; prices
    # and fundamentals are the heavy, failure-prone payloads, so they run last.
    # company_info/sectors/corporate_events are free (CVM CAD + yfinance), so they
    # run in every mode now, including update -- no BolsAI usage to ration.
    # collectors.collect_company_info/collect_sectors/collect_corporate_events (the
    # original BolsAI versions) stay importable but unused by default; see
    # BOLSAI_EXIT_PLAN.md Task 5.
    stages = [
        ("macro", lambda: collectors.collect_macro(mode)),
        ("company_info", lambda: cvm_company_info.synthesize_company_info()),
        ("sectors", lambda: cvm_sectors.build_sectors()),
        ("corporate_events", lambda: yf_collectors.collect_splits_yf(all_tickers, mode)),
    ]

    for name, fn in stages:
        log.info("--- stage: %s ---", name)
        try:
            fn()
        except Exception as e:
            log.error("stage %s failed: %s", name, e, exc_info=True)
            return False

    # After company_info, filter to only ATIVO tickers from the requested list.
    # Benchmarks (ETFs) bypass company_info requirement and are always collected for prices.
    active_all = _active_tickers()
    active = [t for t in tickers if t in set(active_all)]
    # Always include requested benchmarks; also include non-requested benchmarks
    requested_benchmarks = [b for b in config.BENCHMARK_TICKERS if b in tickers]
    other_benchmarks = [b for b in config.BENCHMARK_TICKERS if b not in tickers]
    prices_tickers = sorted(set(active) | set(requested_benchmarks) | set(other_benchmarks))

    if mode == "update":
        recovered = _recover_stale_company_info_tickers(tickers, set(prices_tickers))
        if recovered:
            log.info("%d ticker(s) already on disk but missing/non-ATIVO in company_info -- "
                      "including in the free yfinance price refresh anyway: %s",
                      len(recovered), sorted(recovered)[:10])
        prices_tickers = sorted(set(prices_tickers) | recovered)

    log.info("filtered to %d/%d requested tickers (ATIVO) + %d benchmarks for prices",
             len(active), len(tickers), len(requested_benchmarks) + len(other_benchmarks))

    data_stages = [
        ("prices",       lambda: _collect("prices", prices_tickers, mode)),
        ("fundamentals", lambda: _collect("fundamentals", active, mode)),
        ("dividends",    lambda: _collect("dividends", active, mode)),
    ]

    for name, fn in data_stages:
        log.info("--- stage: %s ---", name)
        try:
            fn()
        except Exception as e:
            log.error("stage %s failed: %s", name, e, exc_info=True)
            return False

    log.info("=" * 60)
    log.info("DONE. Next: python tests/data_collection/validate_vs_yfinance.py")
    log.info("=" * 60)
    return True


def main():
    p = argparse.ArgumentParser(description="Staged data collection pipeline")
    p.add_argument("--mode", choices=["prototype", "full_scale", "update"], default="prototype")
    p.add_argument("--tickers", nargs="+", help="override ticker list")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    setup_logging()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.mode == "prototype":
        tickers = config.PROTOTYPE_TICKERS
    elif args.mode == "update":
        tickers = _active_tickers()
    elif args.mode == "full_scale" and not config.BOLSAI_API_KEY:
        # No key: get_all_tickers() needs BolsAI's /stocks/ registry for fresh-ticker
        # discovery. Fall back to the universe already on disk (company_info.parquet,
        # itself CVM-sourced -- see cvm/company_info.py) so full_scale can still refresh
        # every known ticker's history keylessly; new IPOs need a key to discover (S4,
        # BOLSAI_EXIT_PLAN.md -- pre-existing limitation, not a new gap).
        tickers = _active_tickers()
    else:
        tickers = collectors.get_all_tickers()

    ok = run(args.mode, tickers, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
