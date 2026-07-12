"""
cvm_statements.py — CLI for the CVM open-data collection steps.

BolsAI serves full price history for delisted tickers but 404s on their
fundamentals (only companies still in its live registry resolve — verified
2026-07-11, see DELISTED_UNIVERSE.md). CVM's open-data portal has every
filer's raw statements back to 2010, delisted included. This module rebuilds
fundamentals in the same per-ticker parquet schema collect_fundamentals()
writes, so Stage 2 (load_fundamentals glob-and-concat) needs zero changes.

Steps (all CVM sources free & keyless; caches under data/raw/cvm/), each in
its own module under cvm/:
  crosswalk     FCA valor_mobiliario: ticker -> cnpj (verified 3/3 on
                SMLS3/LAME4/HGTX3), cvm_code joined from filing_dates.parquet
  filing_dates  ITR/DFP registers -> real CVM receipt date (DT_RECEB) per
                quarter, used by Stage 2 to avoid lookahead bias
  statements    DFP/ITR DRE+BPA+BPP -> one wide quarterly frame per cnpj
  shares        FRE capital_social  -> shares-outstanding timeline per cnpj
  fundamentals  BolsAI-named ratio columns; per-ticker parquet written ONLY
                where no BolsAI file exists (BolsAI stays source of truth
                for active tickers)
  company_info  CANCELADA registry rows (sector, cvm_code — ticker-less on
                BolsAI) joined to tickers via the crosswalk, appended to
                company_info.parquet with status=CANCELADA

Usage (from project root):
    python -m src.data_collection.cvm_statements                  # all steps
    python -m src.data_collection.cvm_statements --step crosswalk
    python -m src.data_collection.cvm_statements --step filing_dates
    python -m src.data_collection.cvm_statements --step fundamentals --tickers SMLS3
"""

import argparse
import logging

from .cvm.company_info import synthesize_company_info
from .cvm.crosswalk import build_crosswalk
from .cvm.filing_dates import collect_filing_dates
from .cvm.ratios import build_fundamentals
from .cvm.shares import collect_shares
from .cvm.statements import collect_statements

log = logging.getLogger("cvm_statements")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="CVM open-data collection steps")
    p.add_argument("--step", choices=["crosswalk", "filing_dates", "statements", "shares",
                                      "fundamentals", "company_info", "all"],
                   default="all")
    p.add_argument("--tickers", nargs="+", help="restrict build_fundamentals")
    args = p.parse_args()

    steps = {
        # filing_dates before crosswalk: build_crosswalk() reads filing_dates.parquet
        # to fill cvm_code, so a fresh `--step all` run needs it collected first.
        "filing_dates": collect_filing_dates,
        "crosswalk": build_crosswalk,
        "statements": collect_statements,
        "shares": collect_shares,
        "fundamentals": lambda: build_fundamentals(
            [t.upper() for t in args.tickers] if args.tickers else None),
        "company_info": synthesize_company_info,
    }
    order = list(steps) if args.step == "all" else [args.step]
    for name in order:
        log.info("=== step: %s ===", name)
        steps[name]()


if __name__ == "__main__":
    main()
