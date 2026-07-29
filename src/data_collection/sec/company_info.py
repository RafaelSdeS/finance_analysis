"""sec/company_info.py — SIC code/description per company, the free US analog
of BR's company_info sector field.

SEC's `submissions.json` per CIK carries `sic`/`sicDescription` (Standard
Industrial Classification -- an older, coarser scheme than GICS, but real,
free, and already keyed on CIK, same data.sec.gov domain/throttle every other
sec/ module goes through). Static, current-day metadata, same caveat CLAUDE.md
already documents for BR's `status`/`sector`: fine for point-in-time universe
construction, not something to train on as if it were known at every historical
row's own date.
"""

import json
import logging

import pandas as pd

from .. import config
from . import crosswalk, http

log = logging.getLogger("sec")

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


def collect_company_info(tickers: list[str]) -> pd.DataFrame:
    """ticker -> CIK (tier-1 crosswalk) -> submissions.json -> sic/sicDescription.
    Skips tickers the crosswalk can't resolve or whose fetch fails, logging why
    rather than crashing the whole batch over one bad ticker."""
    cw = pd.read_parquet(crosswalk.CROSSWALK_PATH) if crosswalk.CROSSWALK_PATH.exists() \
        else crosswalk.build_crosswalk_tier1()
    ticker_to_cik = dict(zip(cw["ticker"], cw["cik"]))

    rows = []
    for i, ticker in enumerate(tickers, 1):
        cik = ticker_to_cik.get(ticker)
        if cik is None:
            log.info("company_info %s: no CIK in tier-1 crosswalk, skipping", ticker)
            continue
        cik = int(cik)
        resp = http.get(SUBMISSIONS_URL.format(cik=cik))
        if resp is None:
            log.warning("company_info %s (CIK %s): fetch failed, skipping", ticker, cik)
            continue
        data = json.loads(resp.text)
        rows.append({
            "ticker": ticker,
            "cik": cik,
            "sic": data.get("sic") or None,
            "sic_description": data.get("sicDescription") or None,
        })
        if i % 500 == 0:
            log.info("company_info: %d/%d tickers processed (%d resolved so far)",
                      i, len(tickers), len(rows))

    df = pd.DataFrame(rows, columns=["ticker", "cik", "sic", "sic_description"])
    config.US_SEC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.US_COMPANY_INFO_PATH, index=False)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(message)s")
    cw = pd.read_parquet(crosswalk.CROSSWALK_PATH) if crosswalk.CROSSWALK_PATH.exists() \
        else crosswalk.build_crosswalk_tier1()
    df = collect_company_info(cw["ticker"].tolist())
    log.info("company_info: %d/%d tickers resolved, %d distinct SIC codes",
              df["sic"].notna().sum(), len(df), df["sic"].nunique())
