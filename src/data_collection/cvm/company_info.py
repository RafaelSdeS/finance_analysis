"""cvm/company_info.py — CANCELADA (delisted) company_info rows from BolsAI's
registry, joined to tickers via the FCA crosswalk (BolsAI's CANCELADA
registry itself carries no ticker link)."""

import logging

import pandas as pd

from .. import client, config
from . import http
from .crosswalk import CROSSWALK_PATH

log = logging.getLogger("cvm")


def synthesize_company_info() -> None:
    """Append COMPANY_FIELDS rows for delisted tickers to company_info.parquet.
    Sector/cvm_code come from BolsAI's CANCELADA registry (ticker-less there);
    the ticker link comes from the FCA crosswalk, joined on cnpj."""
    xwalk = pd.read_parquet(CROSSWALK_PATH)

    c = client.make_client(config.BOLSAI_BASE, config.BOLSAI_API_KEY)
    try:
        regs, offset = [], 0
        while True:
            d = client.get_json(c, "/companies/",
                                {"status": "CANCELADA", "limit": 500, "offset": offset})
            batch = d.get("data", [])
            if not batch:
                break
            regs += batch
            offset += len(batch)
            if len(batch) < 500:
                break
    finally:
        c.close()

    reg = pd.DataFrame(regs)
    reg["cnpj"] = reg["cnpj"].map(http.digits)
    reg = reg.drop_duplicates("cnpj")

    merged = xwalk.merge(reg[["cnpj", "cvm_code", "sector", "corporate_name",
                              "trade_name", "status"]],
                         on="cnpj", how="inner", suffixes=("_fca", ""))
    rows = pd.DataFrame({
        "ticker": merged["ticker"],
        "ticker_primary": merged["ticker"],
        "corporate_name": merged["corporate_name"],
        "trade_name": merged["trade_name"],
        "cvm_code": merged["cvm_code"],
        "cnpj": merged["cnpj"],
        "sector": merged["sector"],
        "status": merged["status"],
    })

    path = config.COMPANY_DIR / "company_info.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=rows.columns)
    out = (pd.concat([existing, rows], ignore_index=True)
             .drop_duplicates("ticker", keep="first"))  # existing (BolsAI ATIVO) wins
    out.to_parquet(path, index=False)
    log.info("company_info: +%d delisted rows -> %d total", len(out) - len(existing), len(out))
