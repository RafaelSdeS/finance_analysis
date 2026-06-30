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

from . import collectors, config


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

    if not config.BOLSAI_API_KEY:
        log.error("BOLSAI_API_KEY not set (add it to .env)")
        return False

    # Always append benchmark tickers (prices only, for performance comparison)
    all_tickers = sorted(set(tickers) | set(config.BENCHMARK_TICKERS))

    if dry_run:
        log.info("DRY RUN | mode=%s | %d tickers (+%d benchmarks)", mode, len(tickers), len(config.BENCHMARK_TICKERS))
        log.info("tickers: %s", all_tickers[:20] + (["..."] if len(all_tickers) > 20 else []))
        log.info("would run: macro, company_info, prices, fundamentals, dividends")
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

    stages = [
        ("macro",        lambda: collectors.collect_macro(mode)),
        ("company_info", lambda: collectors.collect_company_info(tickers, mode)),
        ("prices",       lambda: collectors.collect_prices(all_tickers, mode)),  # all_tickers includes benchmarks
        ("fundamentals", lambda: collectors.collect_fundamentals(_active_tickers(), mode)),  # only ATIVO; exclude benchmarks
        ("dividends",    lambda: collectors.collect_dividends(_active_tickers(), mode)),  # only ATIVO; exclude benchmarks
    ]

    for name, fn in stages:
        log.info("--- stage: %s ---", name)
        try:
            fn()
        except Exception as e:
            # Fail fast for operator visibility; checkpoints let the re-run resume.
            log.error("stage %s failed: %s", name, e, exc_info=True)
            return False

    log.info("=" * 60)
    log.info("DONE. Next: python tests/raw_data/validate_vs_yfinance.py")
    log.info("=" * 60)
    return True


def main():
    p = argparse.ArgumentParser(description="Staged data collection pipeline")
    p.add_argument("--mode", choices=["prototype", "full_scale"], default="prototype")
    p.add_argument("--tickers", nargs="+", help="override ticker list")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    setup_logging()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.mode == "prototype":
        tickers = config.PROTOTYPE_TICKERS
    else:
        tickers = collectors.get_all_tickers()

    ok = run(args.mode, tickers, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
