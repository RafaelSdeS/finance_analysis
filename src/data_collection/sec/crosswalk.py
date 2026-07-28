"""sec/crosswalk.py — CIK <-> ticker mapping.

Tier 1 ONLY (this pass): SEC's company_tickers.json, current listings. Covers
every CIK still associated with a ticker today. SEC clears the ticker field
once a company stops filing (verified 2026-07-28: Enron/Lehman/Twitter all
return tickers=[] from their submissions.json), so this tier alone cannot
resolve a ticker for any dead company. Tiers 2-4 (XBRL dei:TradingSymbol on
2009+ cover pages, pre-2009 cover-page text, submissions.json formerNames for
renames) are the plan's §4.3 dead-company recovery ladder -- not implemented
here; compute_coverage() in universe.py treats this as a lower bound, not the
final survivorship-bias number, for exactly this reason.
"""

import json
import logging

import pandas as pd

from .. import config
from . import http

log = logging.getLogger("sec")

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
CROSSWALK_PATH = config.US_SEC_DIR / "cik_ticker_crosswalk.parquet"


def build_crosswalk_tier1() -> pd.DataFrame:
    """CIK -> ticker for every currently-listed company (survivors only)."""
    resp = http.get(TICKERS_URL)
    data = json.loads(resp.text)
    df = pd.DataFrame(data.values())  # columns: cik_str, ticker, title
    df = df.rename(columns={"cik_str": "cik", "title": "company_name"})
    df["cik"] = df["cik"].astype("Int64")
    df["tier"] = 1
    config.US_SEC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CROSSWALK_PATH, index=False)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(message)s")
    df = build_crosswalk_tier1()
    log.info("tier-1 crosswalk: %d CIK->ticker rows (current listings only)", len(df))
