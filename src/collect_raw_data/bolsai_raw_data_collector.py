"""
bolsai_data_pipeline.py
=======================

RAW DATA PIPELINE — fetches and saves all raw data using only the Bolsai API.

Sources:
    - Bolsai API    → daily prices  (paginated, 80 records/page)
    - Bolsai API    → fundamentals  (quarterly, up to 80 quarters ~20 years)
    - Bolsai API    → macro: selic, ipca, cdi

Folder structure:
-----------------
data/
└── raw/
    ├── prices/
    │   ├── PETR4.parquet
    │   └── ...
    ├── fundamentals/
    │   ├── PETR4.parquet
    │   └── ...
    └── macro/
        ├── selic.parquet
        ├── ipca.parquet
        └── cdi.parquet

Usage:
------
    python bolsai_data_pipeline.py --api-key sk_YOUR_KEY
    python bolsai_data_pipeline.py --api-key sk_YOUR_KEY --start 2015-01-01
    python bolsai_data_pipeline.py --api-key sk_YOUR_KEY --tickers PETR4 VALE3
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

SAMPLE_TICKERS = [
    "PETR4",
    "VALE3",
    "WEGE3",
    "PRIO3",
]

MAX_PRICES       = 80   # Bolsai hard limit per page for price history
MAX_FUNDAMENTALS = 80   # Bolsai hard limit for fundamentals history

MACRO_ENDPOINTS = {
    "selic": "/macro/selic",
    "ipca":  "/macro/ipca",
    "cdi":   "/macro/cdi",
}

BASE_URL = "https://api.usebolsai.com/api/v1"

DEFAULT_START = "2005-01-01"


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

def _find_project_root(start):
    markers = (".git", "pyproject.toml", "requirements.txt")
    for p in [start, *start.parents]:
        if any((p / m).exists() for m in markers):
            return p
    return start

ROOT_DIR   = _find_project_root(Path(__file__).resolve().parent)
RAW_DIR    = ROOT_DIR / "data" / "raw"
PRICES_DIR = RAW_DIR / "prices"
FUND_DIR   = RAW_DIR / "fundamentals"
MACRO_DIR  = RAW_DIR / "macro"

for d in [PRICES_DIR, FUND_DIR, MACRO_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# BOLSAI HTTP CLIENT
# =============================================================================

class BolsaiClient:

    def __init__(self, api_key: str, delay: float = 0.25):
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
                    log.warning(f"Rate limit — waiting {wait}s")
                    time.sleep(wait)
                    continue
                if r.status_code in (404, 422, 500):
                    log.debug(f"[{r.status_code}] {url}")
                    return None
                r.raise_for_status()
                time.sleep(self.delay)
                return r.json()
            except Exception as e:
                if attempt == 3:
                    log.error(f"Failed: {url} — {e}")
                    return None
                time.sleep(2 ** attempt)
        return None

    def close(self):
        self.http.close()


# =============================================================================
# STEP 1 — DAILY PRICES via Bolsai (paginated forward from start date)
# =============================================================================

def fetch_prices_bolsai(client: BolsaiClient, ticker: str, start: str) -> pd.DataFrame:
    """
    Downloads full OHLCV history from Bolsai, paginating forward from `start`.
    The API returns at most MAX_PRICES (80) records per call, so we advance
    the window by setting `start` to the day after the last received record.
    """
    log.info(f"[{ticker}] Fetching prices via Bolsai (start={start})...")

    all_pages: list[pd.DataFrame] = []
    cursor = start

    while True:
        raw = client.get(
            f"/stocks/{ticker}/history",
            params={"start": cursor, "limit": MAX_PRICES},
        )
        if not raw:
            break

        page = raw.get("prices", [])
        if not page:
            break

        df_page = pd.DataFrame(page)
        all_pages.append(df_page)

        # If we got fewer records than the page limit, we've reached the end.
        if len(page) < MAX_PRICES:
            break

        # Advance cursor to the day after the last record on this page.
        last_date = page[-1]["trade_date"]
        next_start = (
            datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")

        # Safety guard: prevent infinite loop if the API echoes the same date.
        if next_start <= cursor:
            break

        cursor = next_start

    if not all_pages:
        log.warning(f"[{ticker}] Bolsai returned no price data.")
        return pd.DataFrame()

    df = pd.concat(all_pages, ignore_index=True)

    # Standardise schema
    df.insert(0, "ticker", ticker)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    rename_map = {
        "adjusted_open":   "adj_open",
        "adjusted_high":   "adj_high",
        "adjusted_low":    "adj_low",
        "adjusted_close":  "adj_close",
        "adjusted_volume": "volume_adjusted",
        "traded_amount":   "traded_amount",
        "num_trades":      "num_trades",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    df = df[df["volume"] > 0] if "volume" in df.columns else df
    df = df.drop_duplicates(subset=["ticker", "trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)

    log.info(
        f"[{ticker}] {len(df)} trading days "
        f"({df['trade_date'].min().date()} → {df['trade_date'].max().date()})"
    )

    return df


def save_prices(df: pd.DataFrame, ticker: str):
    path = PRICES_DIR / f"{ticker}.parquet"
    df.to_parquet(path, index=False)
    log.info(f"[{ticker}] Prices saved → {path}")


# =============================================================================
# STEP 2 — QUARTERLY FUNDAMENTALS via Bolsai (up to 80 quarters ≈ 20 years)
# =============================================================================

def fetch_fundamentals(client: BolsaiClient, ticker: str) -> pd.DataFrame:
    log.info(f"[{ticker}] Fetching fundamentals via Bolsai...")

    raw = client.get(
        f"/fundamentals/{ticker}/history",
        params={"limit": MAX_FUNDAMENTALS},
    )

    if not raw:
        log.warning(f"[{ticker}] No fundamentals returned.")
        return pd.DataFrame()

    history = raw.get("history", [])
    if not history:
        log.warning(f"[{ticker}] Fundamentals history is empty.")
        return pd.DataFrame()

    df = pd.DataFrame(history)
    df.insert(0, "ticker", ticker)
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    df = df.sort_values("reference_date").reset_index(drop=True)

    log.info(
        f"[{ticker}] {len(df)} quarters "
        f"({df['reference_date'].min().date()} → {df['reference_date'].max().date()})"
    )

    return df


def save_fundamentals(df: pd.DataFrame, ticker: str):
    path = FUND_DIR / f"{ticker}.parquet"
    df.to_parquet(path, index=False)
    log.info(f"[{ticker}] Fundamentals saved → {path}")


# =============================================================================
# STEP 3 — MACRO DATA via Bolsai (selic, ipca, cdi)
# =============================================================================

def fetch_macro(client: BolsaiClient, name: str, endpoint: str) -> pd.DataFrame:
    log.info(f"[MACRO] Fetching {name}...")

    raw = client.get(endpoint, params={"limit": 500})

    if not raw:
        log.warning(f"[MACRO] No data for {name}.")
        return pd.DataFrame()

    data = raw.get("data", [])
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["reference_date"] = pd.to_datetime(df["date"])
    df[name] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["reference_date", name]]
    df = df.sort_values("reference_date").reset_index(drop=True)

    log.info(
        f"[MACRO] {name}: {len(df)} records "
        f"({df['reference_date'].min().date()} → {df['reference_date'].max().date()})"
    )

    return df


def save_macro(df: pd.DataFrame, name: str):
    path = MACRO_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    log.info(f"[MACRO] {name} saved → {path}")


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(api_key: str, tickers: list[str], start: str):

    client = BolsaiClient(api_key=api_key)

    log.info("=" * 60)
    log.info(f"PIPELINE START — {len(tickers)} ticker(s), start={start}")
    log.info("=" * 60)

    # ── Prices (Bolsai, paginated) ────────────────────────────────────────────
    log.info("")
    log.info("── PRICES (Bolsai) ──────────────────────────────────────")
    for ticker in tickers:
        df = fetch_prices_bolsai(client, ticker, start)
        if not df.empty:
            save_prices(df, ticker)

    # ── Fundamentals (Bolsai) ────────────────────────────────────────────────
    log.info("")
    log.info("── FUNDAMENTALS (Bolsai) ────────────────────────────────")
    for ticker in tickers:
        df = fetch_fundamentals(client, ticker)
        if not df.empty:
            save_fundamentals(df, ticker)

    # ── Macro (Bolsai) ───────────────────────────────────────────────────────
    log.info("")
    log.info("── MACRO (Bolsai) ───────────────────────────────────────")
    for name, endpoint in MACRO_ENDPOINTS.items():
        df = fetch_macro(client, name, endpoint)
        if not df.empty:
            save_macro(df, name)

    client.close()

    log.info("")
    log.info("=" * 60)
    log.info("PIPELINE FINISHED")
    log.info(f"  Prices       → {PRICES_DIR}")
    log.info(f"  Fundamentals → {FUND_DIR}")
    log.info(f"  Macro        → {MACRO_DIR}")
    log.info("=" * 60)


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Download raw market data for ML pipeline (Bolsai only)"
    )
    parser.add_argument("--api-key", required=True, help="Bolsai API key")
    parser.add_argument(
        "--tickers", nargs="*", default=SAMPLE_TICKERS,
        help="Tickers to fetch (default: PETR4 VALE3 WEGE3 PRIO3)"
    )
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help=f"Price history start date (default: {DEFAULT_START})"
    )

    args = parser.parse_args()

    run_pipeline(
        api_key=args.api_key,
        tickers=[t.upper() for t in args.tickers],
        start=args.start,
    )