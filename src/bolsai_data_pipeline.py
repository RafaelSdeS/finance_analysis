"""
bolsai_data_pipeline.py
=======================

RAW DATA PIPELINE — fetches and saves all raw data separately.

Sources:
    - yfinance      → daily prices (full history, free, no API limit)
    - Bolsai API    → fundamentals (quarterly, up to 80 quarters ~20 years)
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
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
import yfinance as yf


# =============================================================================
# CONFIG
# =============================================================================

SAMPLE_TICKERS = [
    "PETR4",
    "VALE3",
    "WEGE3",
    "PRIO3",
]

# Bolsai hard limit for fundamentals endpoint
MAX_FUNDAMENTALS = 80

MACRO_ENDPOINTS = {
    "selic": "/macro/selic",
    "ipca":  "/macro/ipca",
    "cdi":   "/macro/cdi",
}

BASE_URL = "https://api.usebolsai.com/api/v1"

# Default price history start date
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

RAW_DIR       = Path("../data/raw")
PRICES_DIR    = RAW_DIR / "prices"
FUND_DIR      = RAW_DIR / "fundamentals"
MACRO_DIR     = RAW_DIR / "macro"

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
# STEP 1 — DAILY PRICES via yfinance (full history, no API cost)
# =============================================================================

def fetch_prices_yfinance(ticker: str, start: str) -> pd.DataFrame:
    """
    Downloads full OHLCV history from Yahoo Finance.
    Brazilian tickers need the .SA suffix (e.g. PETR4 → PETR4.SA).
    Returns columns matching the Bolsai price schema where possible.
    """
    log.info(f"[{ticker}] Fetching prices via yfinance (start={start})...")

    yf_ticker = f"{ticker}.SA"

    raw = yf.download(
    yf_ticker,
    start=start,
    auto_adjust=False,
    progress=False,
)

    # flatten multi-level columns if present (older yfinance versions)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if raw.empty:
        log.warning(f"[{ticker}] yfinance returned no data.")
        return pd.DataFrame()

    raw = raw.reset_index()
    raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

    # Rename to match Bolsai schema
    raw = raw.rename(columns={
        "date":       "trade_date",
        "adj_close":  "adjusted_close",
    })

    raw["ticker"]     = ticker
    raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.tz_localize(None)

    # Keep only the columns we care about
    keep = [
        "ticker", "trade_date",
        "open", "high", "low", "close", "adjusted_close",
        "volume",
    ]
    raw = raw[[c for c in keep if c in raw.columns]]

    raw = raw[raw["volume"] > 0]

    raw = raw.drop_duplicates(subset=["ticker", "trade_date"])
    raw = raw.sort_values("trade_date").reset_index(drop=True)

    log.info(
        f"[{ticker}] {len(raw)} trading days "
        f"({raw['trade_date'].min().date()} → {raw['trade_date'].max().date()})"
    )

    return raw


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

    # ── Prices (yfinance — no API cost) ──────────────────────────────────────
    log.info("")
    log.info("── PRICES (yfinance) ────────────────────────────────────")
    for ticker in tickers:
        df = fetch_prices_yfinance(ticker, start)
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
        description="Download raw market data for ML pipeline"
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