"""
pipeline.py — orchestration + CLI for the staged data collection pipeline.

Same code runs prototype and full-scale; only the ticker list and checkpoint
dir change (prototype↔production parity). Stages run in dependency order;
each collector is resumable via its checkpoint.

Usage (from project root):
    python -m src.data_collection.pipeline --mode prototype
    python -m src.data_collection.pipeline --mode full_scale
    python -m src.data_collection.pipeline --mode full_scale --dry-run
    python -m src.data_collection.pipeline --mode prototype --tickers PETR4 VALE3
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import collectors, config, yf_collectors


def _collect(name: str, tickers: list[str], mode: str):
    """Per-data-type source switch. Flip config.DATA_SOURCE[name] to fall back to BolsAI.

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
        ("dividends", "bolsai"): collectors.collect_dividends,
        ("dividends", "yfinance"): yf_collectors.collect_dividends_yf,
    }

    # Collect from others using global DATA_SOURCE
    if others:
        source = config.DATA_SOURCE.get(name, "bolsai")
        fn = fn_map[(name, source)]
        fn(others, mode)

    # Collect yfinance-only tickers via yfinance
    if yf_only:
        fn = fn_map[(name, "yfinance")]
        fn(yf_only, mode)


def _tickers_with_company_info() -> list[str]:
    """Return tickers that matched BolsAI company info (exist on the platform)."""
    path = config.COMPANY_DIR / "company_info.parquet"
    if not path.exists():
        return []
    return sorted(pd.read_parquet(path)["ticker"].dropna().unique().tolist())


def _active_tickers() -> list[str]:
    """Return only tickers with status='ATIVO' (exclude delisted/suspended)."""
    path = config.COMPANY_DIR / "company_info.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    return sorted(df[df["status"] == "ATIVO"]["ticker"].dropna().unique().tolist())


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

    # update mode skips company_info and may run prices/fundamentals/dividends
    # entirely via yfinance — only require a BolsAI key if something actually needs it.
    needs_bolsai = mode != "update" or any(
        config.DATA_SOURCE.get(k) == "bolsai" for k in ("prices", "fundamentals", "dividends")
    )
    if needs_bolsai and not config.BOLSAI_API_KEY:
        log.error("BOLSAI_API_KEY not set (add it to .env)")
        return False

    # Always append benchmark tickers (prices only, for performance comparison)
    all_tickers = sorted(set(tickers) | set(config.BENCHMARK_TICKERS))

    if dry_run:
        log.info("DRY RUN | mode=%s | %d tickers (+%d benchmarks)", mode, len(tickers), len(config.BENCHMARK_TICKERS))
        log.info("tickers: %s", all_tickers[:20] + (["..."] if len(all_tickers) > 20 else []))
        stage_names = ["macro"] + ([] if mode == "update" else ["company_info"]) + ["prices", "fundamentals", "dividends"]
        log.info("would run: %s (source: %s)", ", ".join(stage_names), config.DATA_SOURCE if mode == "update" else "bolsai")
        return True

    log.info("=" * 60)
    log.info("DATA COLLECTION | mode=%s | %d tickers (+%d benchmarks)", mode, len(tickers), len(config.BENCHMARK_TICKERS))
    log.info("=" * 60)

    # (name, callable) in dependency order: macro is ticker-independent; prices
    # and fundamentals are the heavy, failure-prone payloads, so they run last.
    # After company_info, narrow to only tickers confirmed to exist on BolsAI —
    # saves ~2x requests by skipping ghost tickers in the per-ticker collectors.
    def _data_tickers():
        matched = _tickers_with_company_info()
        # Exclude benchmarks from company_info/fundamentals/dividends (they're ETFs, not stocks)
        active = [t for t in tickers if t in set(matched)]
        log.info("data stages: %d/%d tickers confirmed on BolsAI", len(active), len(tickers))
        return active

    stages = [("macro", lambda: collectors.collect_macro(mode))]
    if mode != "update":
        # company_info is BolsAI-only and rarely changes; update mode skips it to
        # minimize BolsAI usage. Run `--mode full_scale`/`prototype` manually to
        # pick up new IPOs or status changes.
        stages.append(("company_info", lambda: collectors.collect_company_info(tickers, mode)))

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
    else:
        tickers = collectors.get_all_tickers()

    ok = run(args.mode, tickers, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
