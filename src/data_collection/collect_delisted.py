"""
collect_delisted.py — Stage 1 price backfill for delisted/never-collected tickers.

pipeline.run() gates the per-ticker collectors behind company_info status=ATIVO,
which by construction excludes every delisted ticker (BolsAI's CANCELADA registry
carries no ticker link, so delisted names never match company_info). This script
bypasses that gate: it enumerates the full /stocks/ universe and calls
collect_prices() directly — collect_prices() itself has no ATIVO dependency.

Suffix-11 tickers are ambiguous (corporate units like SULA11 vs FIIs/ETFs like
HGLG11); only names confirmed as CVM-registered companies by the FCA crosswalk
(cvm_statements.build_crosswalk) are included. Without the crosswalk on disk,
suffix-11 names are skipped entirely.

Usage (from project root):
    python -m src.data_collection.collect_delisted --dry-run
    python -m src.data_collection.collect_delisted
    python -m src.data_collection.collect_delisted --tickers SMLS3 LAME4 HGTX3
"""

import argparse
import re
from concurrent.futures import ThreadPoolExecutor

from . import collectors, config

_STOCK = re.compile(r"^[A-Z0-9]{4}[3-8]$")  # same filter as get_all_tickers
_UNIT = re.compile(r"^[A-Z]{4}11$")         # units (SULA11); funds excluded via crosswalk


def candidate_tickers(all_tickers, existing, crosswalk_tickers=None) -> list[str]:
    """Stock-like tickers with no prices parquet yet.

    Suffix 3-8 pass on shape alone; suffix-11 only if the FCA crosswalk confirms
    a CVM-registered company behind them (filters out FIIs/ETFs).
    """
    cands = {t for t in all_tickers if _STOCK.match(t)}
    if crosswalk_tickers:
        cands |= {t for t in all_tickers if _UNIT.match(t) and t in crosswalk_tickers}
    return sorted(cands - set(existing) - set(config.BENCHMARK_TICKERS))


def main():
    p = argparse.ArgumentParser(description="Backfill prices for delisted tickers")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--tickers", nargs="+", help="override candidate list")
    p.add_argument("--workers", type=int, default=10, help="parallel workers (default 10)")
    args = p.parse_args()

    if args.tickers:
        cands = [t.upper() for t in args.tickers]
    else:
        existing = {f.stem for f in config.PRICES_DIR.glob("*.parquet")}
        crosswalk = set()
        xwalk_path = config.CVM_DIR / "fca_crosswalk.parquet"
        if xwalk_path.exists():
            import pandas as pd
            crosswalk = set(pd.read_parquet(xwalk_path)["ticker"])
        else:
            print("note: no FCA crosswalk on disk — suffix-11 units skipped "
                  "(run cvm_statements --step crosswalk first to include them)")
        cands = candidate_tickers(collectors.get_all_tickers_raw(), existing, crosswalk)

    print(f"{len(cands)} candidate tickers")
    if args.dry_run:
        print(" ".join(cands))
        return

    # Divide into batches and process in parallel (10 workers by default)
    batch_size = max(1, len(cands) // args.workers)
    batches = [cands[i:i + batch_size] for i in range(0, len(cands), batch_size)]

    def process_batch(batch):
        collectors.collect_prices(batch, "full_scale")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        executor.map(process_batch, batches)


if __name__ == "__main__":
    main()
