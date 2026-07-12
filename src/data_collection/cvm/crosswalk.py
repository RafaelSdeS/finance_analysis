"""cvm/crosswalk.py — FCA valor_mobiliario: ticker -> cnpj/cvm_code/corporate_name."""

import logging
import re
from datetime import date

import pandas as pd

from .. import config
from . import http
from .filing_dates import OUTPUT_PATH as FILING_DATES_PATH

log = logging.getLogger("cvm")

CROSSWALK_PATH = config.CVM_DIR / "fca_crosswalk.parquet"

_TICKER = re.compile(r"^[A-Z0-9]{4}(?:[3-8]|11)$")
_FCA_COLS = ["ticker", "cnpj", "corporate_name", "end_trading", "year"]


def build_crosswalk() -> pd.DataFrame:
    """ticker -> cnpj, cvm_code, corporate_name, end_trading. Latest FCA wins per ticker.
    Per-year FCA rows cached to data/raw/cvm/fca_{year}.parquet; only the current
    year is re-downloaded on rerun (new filings arrive all year)."""
    config.CVM_DIR.mkdir(parents=True, exist_ok=True)
    current = date.today().year
    frames = []
    for year in range(http.START_YEAR, current + 1):
        cache = config.CVM_DIR / f"fca_{year}.parquet"
        if cache.exists() and year < current:
            frames.append(pd.read_parquet(cache))
            continue
        zf = http.fetch_zip("FCA", year)
        if zf is None:
            continue
        rows = []
        for r in http.read_csv(zf, f"fca_cia_aberta_valor_mobiliario_{year}.csv"):
            ticker = (r.get("Codigo_Negociacao") or "").strip().upper()
            if not _TICKER.match(ticker):
                continue
            rows.append({
                "ticker": ticker,
                "cnpj": http.digits(r.get("CNPJ_Companhia")),
                "corporate_name": (r.get("Nome_Empresarial") or "").strip(),
                "end_trading": (r.get("Data_Fim_Negociacao") or "").strip() or None,
                "year": year,
            })
        df_y = pd.DataFrame(rows, columns=_FCA_COLS)  # empty years cached too (2010 has no codes)
        df_y.to_parquet(cache, index=False)
        frames.append(df_y)
        log.info("FCA %d: %d ticker rows", year, len(df_y))

    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_FCA_COLS)
    if all_rows.empty:
        raise RuntimeError("no FCA data downloaded — CVM portal unreachable?")
    df = (all_rows
          .sort_values("year")
          .drop_duplicates("ticker", keep="last")
          .drop(columns="year"))

    # cvm_code via filing_dates (already on disk, cnpj+cvm_code per filing)
    if FILING_DATES_PATH.exists():
        fd = pd.read_parquet(FILING_DATES_PATH)[["cnpj", "cvm_code"]].drop_duplicates("cnpj")
        df = df.merge(fd, on="cnpj", how="left")
    else:
        df["cvm_code"] = None
        log.warning("filing_dates.parquet missing — cvm_code left null "
                    "(run: python -m src.data_collection.cvm_statements --step filing_dates)")

    df["end_trading"] = pd.to_datetime(df["end_trading"], errors="coerce")
    df.to_parquet(CROSSWALK_PATH, index=False)
    log.info("crosswalk: %d tickers (%d with cvm_code) -> %s",
             len(df), df["cvm_code"].notna().sum(), CROSSWALK_PATH)
    return df
