"""sec/fundamentals.py — combine the XBRL (2007+), EX-27 (usably 1995-2000),
and Item 6 (2001-2006 gap) tiers into one point-in-time fundamentals table
per company.

Item 6 is ANNUAL only (one row per fiscal year, not per quarter like the
other two tiers) -- its `fiscal_year` is mapped to `end` = that year's
Dec-31, a simplification for calendar-fiscal-year companies (Item 6's own
tables don't reliably expose an exact fiscal year-end date the way XBRL/EX-27
do). Flagged via `fundamentals_tier == "item6"` rather than silently blended
in as if it were a real quarterly observation.
"""

import logging

import pandas as pd

from .. import config
from . import companyfacts, crosswalk, fds, item6, universe

log = logging.getLogger("sec")

# XBRL preferred over EX-27/Item 6 on any overlapping fiscal period (richer,
# more reliable -- see plan §2.0); EX-27 preferred over Item 6 (quarterly,
# fuller line-item set, vs. Item 6's annual-only, narrower set). In practice
# the tiers shouldn't overlap (EX-27 usably ends 2000, Item 6 covers
# 2001-2006, XBRL starts ~2006-2007), but resolve deterministically rather
# than leave a silent duplicate `end` if one does.
_TIER_PRIORITY = {"xbrl": 0, "ex27": 1, "item6": 2}


def build_company_fundamentals(cik: int, filings: pd.DataFrame) -> pd.DataFrame:
    """One CIK's combined fundamentals across all three built tiers, one row
    per fiscal period, each stamped with `fundamentals_tier` and a real
    `fundamentals_available_date` (never the period end -- plan §5.2)."""
    frames = []

    facts = companyfacts.fetch_companyfacts(cik)
    if facts is not None:
        line_items = companyfacts.extract_line_items(facts)
        if not line_items.empty:
            xbrl = companyfacts.compute_us_ratios(line_items)
            xbrl["fundamentals_tier"] = "xbrl"
            frames.append(xbrl)

    ex27 = fds.build_cik_history(cik, filings)
    if not ex27.empty:
        ex27 = ex27.rename(columns={"fds_period_end": "end"})
        ex27["fundamentals_tier"] = "ex27"
        frames.append(ex27)

    gap = item6.build_cik_history(cik, filings)
    if not gap.empty:
        gap["end"] = pd.to_datetime(gap["fiscal_year"].astype(str) + "-12-31")
        gap["fundamentals_tier"] = "item6"
        frames.append(gap)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["cik"] = cik
    combined["_priority"] = combined["fundamentals_tier"].map(_TIER_PRIORITY)
    return (combined.sort_values(["end", "_priority"])
                     .drop_duplicates(subset="end", keep="first")
                     .drop(columns="_priority")
                     .sort_values("fundamentals_available_date")
                     .reset_index(drop=True))


def collect_fundamentals_us(tickers: list[str], fund_dir=None) -> None:
    """Batch driver: ticker -> CIK (tier-1 crosswalk) -> build_company_fundamentals()
    -> data/raw/us/fundamentals/{ticker}.parquet. Skips tickers the tier-1 crosswalk
    can't resolve (dead companies -- see plan §4.3) or that yield no fundamentals at
    all from either tier, logging why rather than silently continuing.
    """
    fund_dir = fund_dir or config.US_FUNDAMENTALS_DIR
    fund_dir.mkdir(parents=True, exist_ok=True)
    cw = pd.read_parquet(crosswalk.CROSSWALK_PATH) if crosswalk.CROSSWALK_PATH.exists() \
        else crosswalk.build_crosswalk_tier1()
    ticker_to_cik = dict(zip(cw["ticker"], cw["cik"]))
    filings = pd.read_parquet(universe.FILINGS_PATH)

    for ticker in tickers:
        cik = ticker_to_cik.get(ticker)
        if cik is None:
            log.info("fundamentals %s: no CIK in tier-1 crosswalk, skipping", ticker)
            continue
        try:
            df = build_company_fundamentals(int(cik), filings)
        except Exception as e:
            log.warning("fundamentals %s (CIK %s): skipping after error: %s", ticker, cik, e)
            continue
        if df.empty:
            log.info("fundamentals %s (CIK %s): no data from either tier", ticker, cik)
            continue
        df.to_parquet(fund_dir / f"{ticker}.parquet", index=False)
        log.info("fundamentals %s: %d rows (%s)", ticker, len(df),
                  df["fundamentals_tier"].value_counts().to_dict())


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(message)s")
    all_priced = sorted(p.stem for p in config.US_PRICES_DIR.glob("*.parquet"))
    collect_fundamentals_us(all_priced)
