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
# SEC asks (not enforced, but good practice re throttling) for a descriptive UA
# identifying the requester -- set a real contact in .env to be a good citizen.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "finance-analysis-research contact@example.com")

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
# selic=11, NOT 432 (annual meta target 14.50); cdi=12; ipca=433.
#
# Units (heterogeneous -- confirmed once here so it isn't re-derived or
# mis-assumed downstream; a unit mismatch here previously produced a Critical
# audit finding, see docs/PIPELINE_FORENSIC_AUDIT_2026-07-23.md Issue 1):
#   selic, cdi: DAILY rate, in percent (e.g. 0.0534 means 0.0534%/day).
#   ipca:       MONTHLY rate, in percent (e.g. 0.62 means 0.62% for that month).
# Consumers converting to decimal/log-return terms: build_dataset/features.py
# (compute_macro_features, earnings_yield_vs_selic) and
# build_dataset/merge.py (merge_macro's selic_trend_20d, IPCA_PUBLICATION_LAG_DAYS).
BCB_SERIES = {"selic": 11, "cdi": 12, "ipca": 433}

# --- FRED (US macro) series IDs ---
# Keyless (fredgraph.csv), verified 2026-07-28: CPIAUCNS returns 1913-01-01 -> present.
# Frequency/unit documented once here, same reasoning as BCB_SERIES above (a mismatch
# there was already a Critical audit finding) — never assume before combining series:
#   FEDFUNDS    monthly,   percent (effective fed funds rate)
#   DGS2/10/30  daily,     percent (constant-maturity Treasury yield)
#   CPIAUCSL/NS monthly,   index (1982-84=100) -- a LEVEL, not a rate; pct_change downstream
#   PPIACO      monthly,   index
#   UNRATE      monthly,   percent
#   GDPC1       quarterly, billions of chained 2017 dollars
#   INDPRO      monthly,   index (2017=100)
#   T10Y2Y      daily,     percentage points (10y-2y spread, already differenced)
#   VIXCLS      daily,     index points
#   DTWEXBGS    daily,     index (broad trade-weighted USD)
#   M2SL        monthly,   billions of dollars
FRED_BASE = "https://fred.stlouisfed.org/graph"
FRED_SERIES = {
    "fed_funds": "FEDFUNDS", "treasury_2y": "DGS2", "treasury_10y": "DGS10",
    "treasury_30y": "DGS30", "cpi_sa": "CPIAUCSL", "cpi_nsa": "CPIAUCNS",
    "ppi": "PPIACO", "unemployment": "UNRATE", "real_gdp": "GDPC1",
    "industrial_production": "INDPRO", "term_spread_10y2y": "T10Y2Y",
    "vix": "VIXCLS", "dollar_index": "DTWEXBGS", "m2": "M2SL",
}

# --- Collection limits ---
PRICE_LIMIT = 5000          # API hard cap per request (6000 -> 422)
# ponytail: API rejects limit >= 90; 80 grabs all ~62 quarters available today.
# If a ticker ever exceeds 80 quarters, paginate via start/end (confirmed working).
FUND_LIMIT = 80
PRICE_CHUNK_YEARS = 10      # ~250 trading days/yr * 10 = 2500 rows < cap
START_DATE = "2000-01-01"   # backfill floor; API returns what it has

# --- HTTP retry/backoff ---
MAX_RETRIES = 1             # fail fast on first error; skip-list catches no-data tickers
BACKOFF_BASE = 1            # seconds; wait = min(BACKOFF_BASE * 2**attempt, BACKOFF_MAX)
BACKOFF_MAX = 30
HTTP_TIMEOUT = 60
RATE_LIMIT_SLEEP = 0.3      # polite pause between per-ticker calls (BolsAI, a paid/permissive API)

# yfinance is an unofficial scraper, not a rate-limited-but-documented API like BolsAI --
# confirmed 2026-07-29 at ~2,462-ticker US scale: RATE_LIMIT_SLEEP's 0.3s pace triggered
# sustained Yahoo-side throttling (widespread "possibly delisted; no price data found" on
# real, actively-traded tickers like EQH/EQIX) a few hundred tickers in. Re-probing the
# same tickers directly minutes later returned instantly with correct data -- a rate-based
# throttle, not an IP ban, so a slower per-ticker pace should avoid retriggering it. Kept
# separate from RATE_LIMIT_SLEEP (not just raised globally) so BolsAI backfill runs aren't
# slowed down by a fix that has nothing to do with them.
#
# Re-tested 0.5s deliberately (2026-07-29), since 1.0s was picked without probing anything
# between 0.3s (broken) and 1.0s (safe) -- 330 cold tickers across two isolated runs (a
# scale comparable to where 0.3s actually failed: "a few hundred tickers in"), empty-rate
# DECREASING across the batch (3/83 -> 1/83 -> 0/83), the opposite of the throttle-onset
# signature above -- not the flat/rising pattern a real throttle would show. Actual
# per-ticker cost (~2.1s) is dominated by yfinance's own network+processing time, not this
# sleep, so the real gain is modest (~35%) but genuine, not guessed.
YF_RATE_LIMIT_SLEEP = 0.5

# --- Paths ---
RAW_DIR = PROJECT / "data/raw"
PRICES_DIR = RAW_DIR / "prices"
FUND_DIR = RAW_DIR / "fundamentals"
MACRO_DIR = RAW_DIR / "macro"
COMPANY_DIR = RAW_DIR / "company_info"
DIVIDENDS_DIR = RAW_DIR / "dividends"
CORP_EVENTS_DIR = RAW_DIR / "corporate_events"
CVM_DIR = RAW_DIR / "cvm"           # CVM open-data caches (crosswalk, statements, shares)
US_RAW_DIR = RAW_DIR / "us"         # US-market raw data root (prices/fundamentals/macro)
US_MACRO_DIR = US_RAW_DIR / "macro"
US_PRICES_DIR = US_RAW_DIR / "prices"
US_SEC_DIR = US_RAW_DIR / "sec"      # EDGAR full-index cache, universe roster, CIK<->ticker crosswalk
US_COMPANY_INFO_PATH = US_SEC_DIR / "company_info.parquet"  # SIC code/description per CIK
US_FUNDAMENTALS_DIR = US_RAW_DIR / "fundamentals"
US_DIVIDENDS_DIR = US_RAW_DIR / "dividends"

# US prices: pure yfinance (no BolsAI counterpart), no exchange suffix needed.
# Verified 2026-07-28: GE/KO/IBM/XOM/PG all return 16,249 rows from 1962-01-02.
US_PROTOTYPE_TICKERS = ["AAPL", "GE", "KO", "IBM", "XOM", "PG"]
CHECKPOINT_ROOT = PROJECT / "artifacts/checkpoints"
LOG_DIR = PROJECT / "artifacts/logs/collection"

# --- Collection limits ---
DIVIDENDS_YEARS = 20  # API max; covers full history

# --- yfinance update pipeline ---
# Flip any entry to "yfinance" to fall back to the free collector for that data type.
DATA_SOURCE = {"prices": "yfinance", "fundamentals": "yfinance", "dividends": "yfinance"}
YF_SUFFIX = ".SA"
YF_RETRIES = 3
YF_RETRY_SLEEP = 2          # seconds; doubles each retry
TICKER_ALIASES: dict[str, str] = {}  # old_ticker -> new_yf_ticker, hand-maintained on B3 renames
YFINANCE_ONLY_TICKERS = {"BOVA11"}  # ETFs/benchmarks not in BolsAI; always fetch from yfinance
