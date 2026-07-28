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
        # Non-calendar-fiscal-year companies break the Dec-31 assumption above,
        # producing an impossible end > fundamentals_available_date -- confirmed
        # on ADP (real FYE is June 30; its Aug-filed 10-K got labeled with a
        # Dec-31 "end" that hadn't happened yet at filing time).
        impossible = gap["end"] > gap["fundamentals_available_date"]
        if impossible.any():
            # Derive this company's real fiscal quarter-end (month/day) from
            # whichever row proves the naive Dec-31 guess impossible, then
            # apply it to EVERY row for this CIK -- a company's fiscal
            # year-end doesn't change year to year. Real bug, found auditing
            # ADP's already-collected data (2026-07-28): only the CURRENT
            # year of a filing ever trips this check (its Dec-31 postdates
            # the filing); OLDER comparative years bundled in the SAME Item 6
            # table kept the wrong naive Dec-31 uncorrected, because it still
            # safely precedes the (much later) filing date -- silently
            # mis-dated by up to 6 months, not caught by a per-row-only fix.
            approx = gap.loc[impossible, "fundamentals_available_date"].iloc[0] - pd.DateOffset(months=2)
            q_end = approx.to_period("Q").end_time.normalize()
            gap["end"] = pd.to_datetime(gap["fiscal_year"].astype(str)
                                         + f"-{q_end.month:02d}-{q_end.day:02d}")
        gap["fundamentals_tier"] = "item6"
        frames.append(gap)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["cik"] = cik
    combined["_priority"] = combined["fundamentals_tier"].map(_TIER_PRIORITY)
    # Cluster 'end' across tiers before dedup, don't rely on exact equality.
    # Item6's Dec-31-style rounding and xbrl/ex27's exact fiscal-calendar
    # dates (e.g. "2007-09-29") can describe the SAME real period a few days
    # apart -- an exact-date dedup misses this and keeps both as separate
    # rows. Real bug, found scaling to ~465 companies (2026-07-28): 40 such
    # tier-boundary duplicates (AAPL, INTC, JNJ, MAR, CSX...). Reuses the
    # same tolerance-clustering already applied intra-tier in
    # companyfacts.py (same shape of bug, different tier boundary). The
    # winning row keeps its OWN 'end' (not the cluster midpoint) -- the
    # cluster only decides which duplicate to drop.
    combined["_end_cluster"] = combined["end"].map(companyfacts.cluster_period_ends(combined["end"]))
    return (combined.sort_values(["_end_cluster", "_priority"])
                     .drop_duplicates(subset="_end_cluster", keep="first")
                     .drop(columns=["_priority", "_end_cluster"])
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
