"""cvm/sectors.py — sector classification from CVM's own CAD registry
(SETOR_ATIV field), replacing BolsAI's /companies/sectors endpoint.

Separate small fetch from cvm/delistings.py's build_delist_events() rather than
plumbing SETOR_ATIV through that function's narrower return shape -- CAD is one
lightweight static CSV (not a per-year zip), so a second GET here is cheap and
keeps each module's job single-purpose (delistings.py: cancellation reason for
terminal_events.py; this module: sector taxonomy).
"""

import io
import logging

import pandas as pd

from .. import config, validate
from . import http
from .crosswalk import CROSSWALK_PATH
from .delistings import CAD_URL

log = logging.getLogger("cvm")

SECTORS_PATH = config.COMPANY_DIR / "sectors.parquet"


def _fetch_cad_sectors() -> pd.DataFrame:
    """cnpj -> sector, sit. One row per company: same "prefer ATIVO, else
    latest DT_CANCEL" dedup as delistings.py's build_delist_events(), for the
    same reason -- cad_cia_aberta.csv carries one row per registration episode,
    not one per company."""
    text = http.fetch_csv_url(CAD_URL)
    if text is None:
        raise RuntimeError(f"failed to fetch CVM company registry from {CAD_URL}")
    cad = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
    cad["cnpj"] = cad["CNPJ_CIA"].str.replace(r"\D", "", regex=True)
    cad["delist_date"] = pd.to_datetime(cad["DT_CANCEL"], errors="coerce")
    cad = (
        cad.assign(_is_ativo=cad["SIT"] == "ATIVO")
           .sort_values(["_is_ativo", "delist_date"], na_position="first")
           .drop_duplicates("cnpj", keep="last")
    )
    return cad[["cnpj", "SETOR_ATIV", "SIT"]].rename(
        columns={"SETOR_ATIV": "sector", "SIT": "sit"})


def sector_by_ticker() -> pd.Series:
    """ticker -> sector, for filling company_info.parquet's `sector` column."""
    xwalk = pd.read_parquet(CROSSWALK_PATH)[["ticker", "cnpj"]]
    cad = _fetch_cad_sectors()
    merged = xwalk.merge(cad[["cnpj", "sector"]], on="cnpj", how="inner")
    return merged.dropna(subset=["sector"]).set_index("ticker")["sector"]


def build_sectors() -> None:
    """Canonical sector names + ATIVO company counts -> sectors.parquet, same
    (name, count) schema BolsAI's /companies/sectors produced."""
    cad = _fetch_cad_sectors()
    active = cad[(cad["sit"] == "ATIVO") & cad["sector"].notna()]
    df = (active.groupby("sector").size()
                .reset_index(name="count")
                .rename(columns={"sector": "name"})
                .sort_values("count", ascending=False)
                .reset_index(drop=True))
    vr = validate.validate_sectors(df)
    if not vr.passed:
        log.error("sectors: validation FAILED: %s", vr.errors)
        return
    SECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SECTORS_PATH, index=False)
    log.info("sectors: %d sectors total -> %s", len(df), SECTORS_PATH)
