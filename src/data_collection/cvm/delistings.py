"""cvm/delistings.py — CVM's own cancellation registry (cad_cia_aberta.csv):
delist date, reason, and current registry status per company, joined to
tickers via the FCA crosswalk.

Distinct from company_info.py's synthesize_company_info(), which sources
sector/status from BolsAI's CANCELADA registry for company_info.parquet rows.
This module sources the CANCELLATION REASON (MOTIVO_CANCEL) from CVM
directly -- BolsAI's registry doesn't carry it -- for
build_dataset/terminal_events.py to turn a delisted ticker's forward-return
label into a realized payoff instead of an unexplained NaN.

cad_cia_aberta.csv is CVM's static company master (not a yearly zip like
every other cvm/ source): one row per company ever registered, active or
cancelled, with CNPJ/name/sector/status/cancellation date+reason. Verified
live 2026-08-15: no ticker field at all, hence the join through the FCA
crosswalk rather than reading tickers from here directly.

Usage: python -m src.data_collection.br.cvm_statements --step delistings
"""

import io
import logging

import pandas as pd

from .. import config
from . import http
from .crosswalk import CROSSWALK_PATH

log = logging.getLogger("cvm")

CAD_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv"
OUTPUT_PATH = config.CVM_DIR / "delist_events.parquet"


def build_delist_events() -> pd.DataFrame:
    """ticker -> cnpj, delist_date, motivo_cancel, sit -- joined via the FCA
    crosswalk (CROSSWALK_PATH), since cad_cia_aberta.csv itself has no ticker.

    sit == 'ATIVO' rows are tickers whose own price series stopped but the
    COMPANY is still registered active -- an unspliced rename/share-class
    consolidation, not a real delisting. build_dataset/terminal_events.py
    routes those to find_rename_candidates() instead of a terminal payoff.
    """
    text = http.fetch_csv_url(CAD_URL)
    if text is None:
        raise RuntimeError(f"failed to fetch CVM company registry from {CAD_URL}")
    cad = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
    cad["cnpj"] = cad["CNPJ_CIA"].str.replace(r"\D", "", regex=True)
    cad["delist_date"] = pd.to_datetime(cad["DT_CANCEL"], errors="coerce")

    # cad_cia_aberta.csv carries one row per registration EPISODE, not one per
    # company -- a cnpj can have several (a stale pre-1978-rule registration
    # superseded by a real cancellation, or a still-open registration
    # alongside an old closed one). Verified live 2026-08-15: 140/2530 cnpjs
    # duplicated, 34 with genuinely different SIT/DT_CANCEL across their rows
    # (e.g. Vibra Energia: one stale 2003 CANCELADA row + a current ATIVO
    # row -- naive drop_duplicates without this ordering silently picks
    # whichever row happens to sort first, at real risk of reading a still-
    # active company as delisted). Keep the row reflecting the company's
    # ACTUAL current state: prefer any ATIVO row, else the latest DT_CANCEL.
    cad = (
        cad.assign(_is_ativo=cad["SIT"] == "ATIVO")
           .sort_values(["_is_ativo", "delist_date"], na_position="first")
           .drop_duplicates("cnpj", keep="last")
    )

    xwalk = pd.read_parquet(CROSSWALK_PATH)[["ticker", "cnpj"]].dropna(subset=["cnpj"])
    merged = xwalk.merge(
        cad[["cnpj", "delist_date", "MOTIVO_CANCEL", "SIT"]], on="cnpj", how="inner"
    )
    out = pd.DataFrame({
        "ticker": merged["ticker"],
        "cnpj": merged["cnpj"],
        "delist_date": merged["delist_date"],
        "motivo_cancel": merged["MOTIVO_CANCEL"],
        "sit": merged["SIT"],
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)
    log.info("delist_events: %d tickers resolved (%d with a cancellation reason) -> %s",
              len(out), out["motivo_cancel"].notna().sum(), OUTPUT_PATH)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(message)s")
    build_delist_events()
