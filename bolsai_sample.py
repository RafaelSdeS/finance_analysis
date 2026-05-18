"""
bolsai_data_pipeline.py
=======================

Objective
---------
This script DOES NOT build the final ML dataset yet.

Instead, it builds a PROFESSIONAL RAW DATA PIPELINE:

1. Downloads and saves DAILY PRICES
2. Downloads and saves QUARTERLY FUNDAMENTALS
3. Downloads and saves FINANCIAL STATEMENTS
4. Downloads and saves MACRO DATA
5. Saves EVERYTHING separately in parquet files

This is the correct architecture for ML projects because:
- raw data remains immutable
- datasets can be rebuilt later
- features can be regenerated
- easier debugging
- avoids corruption
- scalable

Folder structure created:
-------------------------

data/
│
├── raw/
│   ├── prices/
│   │   ├── PETR4.parquet
│   │   ├── VALE3.parquet
│   │   └── ...
│   │
│   ├── fundamentals/
│   │   ├── PETR4.parquet
│   │   └── ...
│   │
│   ├── financials/
│   │   ├── PETR4.parquet
│   │   └── ...
│   │
│   └── macro/
│       ├── selic.parquet
│       ├── ipca.parquet
│       └── cdi.parquet

Later:
------
Another script will:
- load raw data
- perform as-of joins
- generate features
- create train/test datasets

Usage
-----
python bolsai_data_pipeline.py --api-key sk_YOUR_KEY
"""

import argparse
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import httpx
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

BASE_URL = "https://api.usebolsai.com/api/v1"

MAX_HIST = 80

REPORT_TYPES = ["DFP", "ITR"]

STATEMENT_TYPES = [
    "BPA",
    "BPP",
    "DRE",
    "DFC_MI",
    "DVA",
]

SAMPLE_TICKERS = [
    "PETR4",
    "VALE3",
    "WEGE3",
    "PRIO3",
]

MACRO_ENDPOINTS = {
    "selic": "/macro/selic",
    "ipca": "/macro/ipca",
    "cdi": "/macro/cdi",
}


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger(__name__)


# =============================================================================
# FOLDERS
# =============================================================================

RAW_DIR = Path("data/raw")

PRICES_DIR = RAW_DIR / "prices"
FUND_DIR = RAW_DIR / "fundamentals"
FIN_DIR = RAW_DIR / "financials"
MACRO_DIR = RAW_DIR / "macro"

for directory in [PRICES_DIR, FUND_DIR, FIN_DIR, MACRO_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HTTP CLIENT
# =============================================================================

class BolsaiClient:

    def __init__(self, api_key: str, delay: float = 0.2):

        self.delay = delay

        self.http = httpx.Client(
            timeout=30,
            headers={"X-API-Key": api_key},
            follow_redirects=True,
        )

    def get(self, path: str, params: dict = None) -> Optional[dict]:

        url = f"{BASE_URL}/{path.lstrip('/')}"

        for attempt in range(1, 4):

            try:

                r = self.http.get(url, params=params or {})

                if r.status_code == 429:

                    wait = int(r.headers.get("Retry-After", 60))

                    log.warning(f"Rate limit hit. Waiting {wait}s")

                    time.sleep(wait)

                    continue

                if r.status_code in [404, 422, 500]:
                    return None

                r.raise_for_status()

                time.sleep(self.delay)

                return r.json()

            except Exception as e:

                if attempt == 3:

                    log.error(f"Request failed: {url}")
                    log.error(str(e))

                    return None

                time.sleep(2 ** attempt)

        return None

    def close(self):
        self.http.close()


# =============================================================================
# STEP 1 — DAILY PRICES
# =============================================================================

def fetch_prices(client: BolsaiClient, ticker: str) -> pd.DataFrame:

    log.info(f"[{ticker}] Fetching prices...")

    all_prices = []

    start = "2000-01-01"

    while True:

        raw = client.get(
            f"/stocks/{ticker}/history",
            params={
                "start": start,
                "limit": MAX_HIST,
            }
        )

        if not raw:
            break

        page = raw.get("prices", [])

        if not page:
            break

        all_prices.extend(page)

        if len(page) < MAX_HIST:
            break

        last_date = page[-1]["trade_date"]

        next_start = (
            datetime.strptime(last_date, "%Y-%m-%d")
            + timedelta(days=1)
        ).strftime("%Y-%m-%d")

        if next_start <= start:
            break

        start = next_start

    if not all_prices:
        return pd.DataFrame()

    df = pd.DataFrame(all_prices)

    df["trade_date"] = pd.to_datetime(df["trade_date"])

    df.insert(0, "ticker", ticker)

    df.drop_duplicates(
        subset=["ticker", "trade_date"],
        keep="last",
        inplace=True
    )

    df.sort_values("trade_date", inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df


def save_prices(df: pd.DataFrame, ticker: str):

    output = PRICES_DIR / f"{ticker}.parquet"

    df.to_parquet(output, index=False)

    log.info(f"[{ticker}] Prices saved → {output}")


# =============================================================================
# STEP 2 — FUNDAMENTALS
# =============================================================================

def fetch_fundamentals(
    client: BolsaiClient,
    ticker: str
) -> pd.DataFrame:

    log.info(f"[{ticker}] Fetching fundamentals...")

    raw = client.get(
        f"/fundamentals/{ticker}/history",
        params={"limit": MAX_HIST}
    )

    if not raw:
        return pd.DataFrame()

    history = raw.get("history", [])

    if not history:
        return pd.DataFrame()

    df = pd.DataFrame(history)

    df["reference_date"] = pd.to_datetime(df["reference_date"])

    df.insert(0, "ticker", ticker)

    company_info = {
        "corporate_name": raw.get("corporate_name"),
        "sector": raw.get("sector"),
        "subsector": raw.get("subsector"),
        "segment": raw.get("segment"),
        "listing_segment": raw.get("listing_segment"),
        "stock_type": raw.get("type"),
    }

    for col, value in company_info.items():
        df[col] = value

    df.sort_values("reference_date", inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df


def save_fundamentals(df: pd.DataFrame, ticker: str):

    output = FUND_DIR / f"{ticker}.parquet"

    df.to_parquet(output, index=False)

    log.info(f"[{ticker}] Fundamentals saved → {output}")


# =============================================================================
# STEP 3 — FINANCIAL STATEMENTS
# =============================================================================

def fetch_financials(
    client: BolsaiClient,
    ticker: str
) -> pd.DataFrame:

    log.info(f"[{ticker}] Fetching financial statements...")

    frames = []

    for report_type in REPORT_TYPES:

        for statement_type in STATEMENT_TYPES:

            raw = client.get(
                f"/financials/{ticker}",
                params={
                    "report_type": report_type,
                    "statement_type": statement_type,
                    "limit": MAX_HIST,
                }
            )

            if not raw:
                continue

            statements = raw.get("statements", [])

            if not statements:
                continue

            df = pd.DataFrame(statements)

            df["report_type"] = report_type

            df["statement_type"] = statement_type

            frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    df["reference_date"] = pd.to_datetime(df["reference_date"])

    df.insert(0, "ticker", ticker)

    df.sort_values(
        ["reference_date", "statement_type"],
        inplace=True
    )

    df.reset_index(drop=True, inplace=True)

    return df


def save_financials(df: pd.DataFrame, ticker: str):

    output = FIN_DIR / f"{ticker}.parquet"

    df.to_parquet(output, index=False)

    log.info(f"[{ticker}] Financials saved → {output}")


# =============================================================================
# STEP 4 — MACRO DATA
# =============================================================================

def fetch_macro_series(
    client: BolsaiClient,
    name: str,
    endpoint: str,
) -> pd.DataFrame:

    log.info(f"[MACRO] Fetching {name}...")

    raw = client.get(endpoint, params={"limit": 500})

    if not raw:
        return pd.DataFrame()

    data = raw.get("data", [])

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    df["date"] = pd.to_datetime(df["date"])

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    df.rename(
        columns={
            "date": "reference_date",
            "value": name,
        },
        inplace=True
    )

    df.sort_values("reference_date", inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df


def save_macro(df: pd.DataFrame, name: str):

    output = MACRO_DIR / f"{name}.parquet"

    df.to_parquet(output, index=False)

    log.info(f"[MACRO] {name} saved → {output}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    api_key: str,
    tickers: list[str],
):

    client = BolsaiClient(api_key=api_key)

    # =========================================================================
    # PRICES
    # =========================================================================

    for ticker in tickers:

        df_prices = fetch_prices(client, ticker)

        

        if not df_prices.empty:
            save_prices(df_prices, ticker)

    # =========================================================================
    # FUNDAMENTALS
    # =========================================================================

    for ticker in tickers:

        df_fund = fetch_fundamentals(client, ticker)

        if not df_fund.empty:
            save_fundamentals(df_fund, ticker)

    # =========================================================================
    # FINANCIALS
    # =========================================================================

    for ticker in tickers:

        df_fin = fetch_financials(client, ticker)

        if not df_fin.empty:
            save_financials(df_fin, ticker)

    # =========================================================================
    # MACRO
    # =========================================================================

    for name, endpoint in MACRO_ENDPOINTS.items():

        df_macro = fetch_macro_series(
            client,
            name,
            endpoint
        )

        if not df_macro.empty:
            save_macro(df_macro, name)

    client.close()

    log.info("")
    log.info("=" * 60)
    log.info("RAW DATA PIPELINE FINISHED")
    log.info("=" * 60)


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--api-key",
        required=True,
    )

    parser.add_argument(
        "--tickers",
        nargs="*",
        default=SAMPLE_TICKERS,
    )

    args = parser.parse_args()

    run_pipeline(
        api_key=args.api_key,
        tickers=[t.upper() for t in args.tickers],
    )