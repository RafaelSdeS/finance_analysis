"""
config.py — shared configuration for the data collection pipeline.

Loads .env (stdlib parser, no python-dotenv dependency), defines tickers,
paths, API endpoints, and collection constants. Prototype vs full-scale
differ ONLY in the ticker list and checkpoint dir — everything else is shared.
"""

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def load_env(path: Path = PROJECT / ".env") -> None:
    """Minimal .env loader. ponytail: 4 lines beats a python-dotenv dependency."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


load_env()

# --- Secrets ---
BOLSAI_API_KEY = os.environ.get("BOLSAI_API_KEY")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# --- Endpoints ---
BOLSAI_BASE = "https://api.usebolsai.com/api/v1"
BCB_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"

# --- Tickers ---
# Prototype: small representative sample (validated against yfinance).
PROTOTYPE_TICKERS = ["PETR4", "VALE3", "WEGE3"]
# Full-scale: fetched dynamically from BolsAI /stocks/ (see collectors.get_all_tickers).
# Benchmarks: prices only (no fundamentals/dividends); used for performance comparison.
BENCHMARK_TICKERS = ["BOVA11"]  # iShares Bovespa ETF (IBOV index proxy)

# --- BCB macro series IDs (confirmed against existing data units) ---
# selic=11 (daily rate ~0.0534), NOT 432 (annual meta target 14.50); cdi=12; ipca=433
BCB_SERIES = {"selic": 11, "cdi": 12, "ipca": 433}

# --- Collection limits ---
PRICE_LIMIT = 5000          # API hard cap per request (6000 -> 422)
# ponytail: API rejects limit >= 90; 80 grabs all ~62 quarters available today.
# If a ticker ever exceeds 80 quarters, paginate via start/end (confirmed working).
FUND_LIMIT = 80
PRICE_CHUNK_YEARS = 10      # ~250 trading days/yr * 10 = 2500 rows < cap
START_DATE = "2000-01-01"   # backfill floor; API returns what it has

# --- HTTP retry/backoff ---
MAX_RETRIES = 3
BACKOFF_BASE = 1            # seconds; wait = min(BACKOFF_BASE * 2**attempt, BACKOFF_MAX)
BACKOFF_MAX = 30
HTTP_TIMEOUT = 60
RATE_LIMIT_SLEEP = 0.3      # polite pause between per-ticker calls

# --- Paths ---
RAW_DIR = PROJECT / "data/raw"
PRICES_DIR = RAW_DIR / "prices"
FUND_DIR = RAW_DIR / "fundamentals"
MACRO_DIR = RAW_DIR / "macro"
COMPANY_DIR = RAW_DIR / "company_info"
DIVIDENDS_DIR = RAW_DIR / "dividends"
CORP_EVENTS_DIR = RAW_DIR / "corporate_events"
CHECKPOINT_ROOT = PROJECT / "artifacts/checkpoints"
LOG_DIR = PROJECT / "artifacts/logs/collection"

# --- Collection limits ---
DIVIDENDS_YEARS = 20  # API max; covers full history

# --- yfinance update pipeline ---
# Flip any entry to "yfinance" to fall back to the free collector for that data type.
DATA_SOURCE = {"prices": "bolsai", "fundamentals": "bolsai", "dividends": "bolsai"}
YF_SUFFIX = ".SA"
YF_RETRIES = 3
YF_RETRY_SLEEP = 2          # seconds; doubles each retry
TICKER_ALIASES: dict[str, str] = {}  # old_ticker -> new_yf_ticker, hand-maintained on B3 renames
YFINANCE_ONLY_TICKERS = {"BOVA11"}  # ETFs/benchmarks not in BolsAI; always fetch from yfinance
