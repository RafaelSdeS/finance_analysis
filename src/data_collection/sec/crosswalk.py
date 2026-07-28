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

# SEC's company_tickers.json occasionally points a ticker at a newly-created
# holding-company shell CIK with zero (or near-zero) filing history, while
# the real, decades-long filing history stays under the OLD CIK indefinitely
# (the old entity gets renamed/demoted but never refiles under the new one).
# Confirmed on two cases (2026-07-28), both verified via submissions.json's
# formerNames -- distinguished from a genuinely NEW company (spinoff/IPO/
# merger) by the OLD entity's decades of real filings continuing under a
# diminished name, not a merger-shell placeholder name that was always
# destined to become the new public entity (e.g. Apollo's "Tango Holdings,
# Inc." or TKO's "New Whale Inc." -- normal, legitimate, NOT this bug):
#   XOM: ticker -> CIK 2115436 ("ExxonMobil Holdings Corp", 0 filings, 0 XBRL
#        concepts); real filer is CIK 34088 ("Exxon Mobil Corporation", 133
#        10-K/10-Q filings, 438 XBRL concepts).
#   BLK: ticker -> CIK 2012383 (created 2024-02 as "BlackRock Funding, Inc.",
#        renamed "BlackRock, Inc." 2024-10, only 1 real filing); real filer
#        is CIK 1364742 ("BlackRock Finance, Inc.", 73 10-K/10-Q filings back
#        to 2006, 557 XBRL concepts).
# A one-off patch of the on-disk crosswalk parquet would be silently lost
# the next time build_crosswalk_tier1() refetches from SEC -- this override
# survives that. A systematic scan of all 500 top-cap tickers (companies
# with <20 filings and an earliest filing after 2020) found 21 candidates;
# the other 19 were verified as genuinely new entities (real spinoffs,
# IPOs, mergers, or a 20-F-to-10-K filer-type transition), not more
# instances of this bug.
CIK_OVERRIDES = {
    "XOM": 34088,
    "BLK": 1364742,
}


def build_crosswalk_tier1() -> pd.DataFrame:
    """CIK -> ticker for every currently-listed company (survivors only)."""
    resp = http.get(TICKERS_URL)
    data = json.loads(resp.text)
    df = pd.DataFrame(data.values())  # columns: cik_str, ticker, title
    df = df.rename(columns={"cik_str": "cik", "title": "company_name"})
    df["cik"] = df["cik"].astype("Int64")
    df["tier"] = 1
    for ticker, cik in CIK_OVERRIDES.items():
        df.loc[df["ticker"] == ticker, "cik"] = cik
    config.US_SEC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CROSSWALK_PATH, index=False)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(message)s")
    df = build_crosswalk_tier1()
    log.info("tier-1 crosswalk: %d CIK->ticker rows (current listings only)", len(df))
