"""
macro_refresh.py
================

Downloads macroeconomic series from BCB SGS with full historical coverage.

Fixes:
- Handles BCB 10-year window limitation for daily series
- Fetches full history via chunking
- Saves clean parquet files for ML pipelines

Output:
    data/raw/macro/selic.parquet
    data/raw/macro/cdi.parquet
    data/raw/macro/ipca.parquet

Usage:
    python macro_refresh.py
    python macro_refresh.py --start 1995-01-01
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

import httpx
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BCB_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"

SERIES = {
    "selic": 432,
    "cdi": 12,
    "ipca": 433,
}

MAX_YEARS_PER_REQUEST = 10

OUTPUT_DIR = Path("data/raw/macro")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger(__name__)


# ============================================================
# DATE UTIL
# ============================================================

def chunk_date_range(start: str, end: str, years: int = 10):
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    chunks = []

    while start_dt <= end_dt:

        chunk_end = min(
            datetime(start_dt.year + years, start_dt.month, start_dt.day),
            end_dt
        )

        chunks.append((
            start_dt.strftime("%d/%m/%Y"),
            chunk_end.strftime("%d/%m/%Y"),
        ))

        start_dt = chunk_end + timedelta(days=1)

    return chunks


# ============================================================
# BCB FETCH (ROBUST)
# ============================================================

def fetch_bcb_series(series_id: int, start: str, end: str):

    url = f"{BCB_BASE}.{series_id}/dados"

    chunks = chunk_date_range(start, end)

    all_rows = []

    for i, (ini, fim) in enumerate(chunks):

        log.info(f"[BCB {series_id}] chunk {i+1}: {ini} → {fim}")

        try:
            r = httpx.get(
                url,
                params={
                    "formato": "json",
                    "dataInicial": ini,
                    "dataFinal": fim,
                },
                timeout=60,
            )
        except Exception as e:
            log.error(f"request failed: {e}")
            continue

        if r.status_code != 200:
            log.error(f"HTTP {r.status_code}: {r.text[:200]}")
            continue

        data = r.json()

        if data:
            all_rows.extend(data)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    df["date"] = pd.to_datetime(df["data"], dayfirst=True)
    df["value"] = pd.to_numeric(df["valor"], errors="coerce")

    df = (
        df[["date", "value"]]
        .dropna()
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# SAVE
# ============================================================

def save(df: pd.DataFrame, name: str):

    path = OUTPUT_DIR / f"{name}.parquet"

    df.to_parquet(path, index=False)

    log.info(
        f"[{name}] saved -> {path} "
        f"({df.date.min().date()} → {df.date.max().date()})"
    )


# ============================================================
# PIPELINE
# ============================================================

def run(start: str, end: str):

    log.info("=" * 60)
    log.info(f"MACRO REFRESH START | {start} → {end}")
    log.info("=" * 60)

    for name, sid in SERIES.items():

        log.info(f"[{name}] fetching series {sid}")

        df = fetch_bcb_series(sid, start, end)

        if not df.empty:
            save(df, name)

    log.info("=" * 60)
    log.info("DONE")
    log.info("=" * 60)


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default="2026-01-01")

    args = parser.parse_args()

    run(args.start, args.end)