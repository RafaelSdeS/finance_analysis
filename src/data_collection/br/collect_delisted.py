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
    python -m src.data_collection.br.collect_delisted --dry-run
    python -m src.data_collection.br.collect_delisted
    python -m src.data_collection.br.collect_delisted --tickers SMLS3 LAME4 HGTX3
"""

import argparse
import json
import re

from . import collectors
from .. import config

_STOCK = re.compile(r"^[A-Z0-9]{4}[3-8]$")  # same filter as get_all_tickers
_UNIT = re.compile(r"^[A-Z]{4}11$")         # units (SULA11); funds excluded via crosswalk

_CONTINUITY_PATH = config.BR_RAW_DIR / "reference/ticker_continuity.json"


def _known_renamed_tickers() -> set[str]:
    """Old-side tickers of a rename/merger already spliced in ticker_continuity.json.

    BolsAI resolves a retired code to the live successor and serves its current
    data under the dead symbol, unflagged (docs/BR_DATA_RECONSTRUCTION_PLAN.md §9
    "rename phantom" -- confirmed on KROT3/ELET3). Collecting these here would
    reintroduce a second, uncorrected copy of the surviving entity's own history
    right as apply_ticker_continuity() is about to splice the real one in. Only
    catches renames already known; find_rename_candidates() (Stage 2) is the
    backstop for ones not yet in this file.
    """
    if not _CONTINUITY_PATH.exists():
        return set()
    events = json.loads(_CONTINUITY_PATH.read_text())["events"]
    return {e["old"] for e in events if e["type"] in ("rename", "merger")}


def candidate_tickers(all_tickers, existing, crosswalk_tickers=None) -> list[str]:
    """Stock-like tickers with no prices parquet yet.

    Suffix 3-8 pass on shape alone; suffix-11 only if the FCA crosswalk confirms
    a CVM-registered company behind them (filters out FIIs/ETFs). Excludes known
    rename/merger old-side tickers -- see _known_renamed_tickers().
    """
    cands = {t for t in all_tickers if _STOCK.match(t)}
    if crosswalk_tickers:
        cands |= {t for t in all_tickers if _UNIT.match(t) and t in crosswalk_tickers}
    return sorted(cands - set(existing) - set(config.BENCHMARK_TICKERS) - _known_renamed_tickers())


def main():
    p = argparse.ArgumentParser(description="Backfill prices for delisted tickers")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--tickers", nargs="+", help="override candidate list")
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

    # collect_prices() loads its own checkpoint dict per call and isn't safe to
    # call concurrently across ticker batches (each call's `checkpoint.save()`
    # would overwrite the others' in-memory view, last-writer-wins) -- one
    # call over the full list, relying on its existing per-ticker rate limit
    # (config.RATE_LIMIT_SLEEP), same as how pipeline.py invokes it.
    collectors.collect_prices(cands, "full_scale")


if __name__ == "__main__":
    main()
