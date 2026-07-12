"""cvm/shares.py — FRE capital_social -> shares-outstanding timeline per cnpj."""

import logging
from datetime import date

import pandas as pd

from .. import config
from . import http

log = logging.getLogger("cvm")

SHARES_PATH = config.CVM_DIR / "shares.parquet"
_FRE_COLS = ["cnpj", "effective_date", "shares"]


def collect_shares() -> pd.DataFrame:
    """Per-cnpj timeline of total shares: (cnpj, effective_date, shares).
    Per-year FRE rows cached to data/raw/cvm/fre_{year}.parquet (zips are ~10 MB
    each); only the current year is re-downloaded on rerun."""
    config.CVM_DIR.mkdir(parents=True, exist_ok=True)
    current = date.today().year
    frames = []
    for year in range(http.START_YEAR, current + 1):
        cache = config.CVM_DIR / f"fre_{year}.parquet"
        if cache.exists() and year < current:
            frames.append(pd.read_parquet(cache))
            continue
        zf = http.fetch_zip("FRE", year)
        if zf is None:
            continue
        rows = []
        for r in http.read_csv(zf, f"fre_cia_aberta_capital_social_{year}.csv"):
            if r.get("Tipo_Capital") != "Capital Integralizado":
                continue
            try:
                shares = int(r.get("Quantidade_Total_Acoes", "") or 0)
            except ValueError:
                continue
            if shares <= 0:
                continue
            rows.append({
                "cnpj": http.digits(r.get("CNPJ_Companhia")),
                "effective_date": r.get("Data_Autorizacao_Aprovacao") or r.get("Data_Referencia"),
                "shares": shares,
            })
        df_y = pd.DataFrame(rows, columns=_FRE_COLS)
        df_y.to_parquet(cache, index=False)
        frames.append(df_y)
        log.info("FRE %d: %d rows", year, len(df_y))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_FRE_COLS)
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    df = (df.dropna(subset=["effective_date"])
            .sort_values("effective_date")
            .drop_duplicates(["cnpj", "effective_date"], keep="last")
            .reset_index(drop=True))
    df.to_parquet(SHARES_PATH, index=False)
    log.info("shares: %d rows, %d companies -> %s", len(df), df["cnpj"].nunique(), SHARES_PATH)
    return df
