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
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .. import config, validate
from . import companyfacts, crosswalk, fds, item6, universe

log = logging.getLogger("sec")

# XBRL preferred over EX-27/Item 6 on any overlapping fiscal period (richer,
# more reliable -- see plan §2.0); EX-27 preferred over Item 6 (quarterly,
# fuller line-item set, vs. Item 6's annual-only, narrower set). In practice
# the tiers shouldn't overlap (EX-27 usably ends 2000, Item 6 covers
# 2001-2006, XBRL starts ~2006-2007), but resolve deterministically rather
# than leave a silent duplicate `end` if one does.
_TIER_PRIORITY = {"xbrl": 0, "ex27": 1, "item6": 2}

# A handful of CIKs are genuine corporate spinoffs/split-offs/post-bankruptcy
# successors whose SEC companyfacts XBRL data includes PRE-SEPARATION
# comparative financials describing a legally different predecessor entity --
# not this company's own standalone history. Unlike CIK_OVERRIDES
# (crosswalk.py's shell-CIK fix), the CIK here is already the right one for
# the ticker; the problem is upstream SEC XBRL data blending a predecessor's
# books into the new registrant's own facts, an inverse-shaped version of the
# same "wrong entity's numbers under this ticker" risk. Found auditing a
# "751/1,848 tickers have a fundamentals gap" symptom (2026-07-29): most of
# that gap turned out to be the item6.py cascade bug (see item6.py's
# _FISCAL_YEAR_MIN/_MAX), but 40 of the remaining xbrl-tier cases are this,
# confirmed via SEC's own filing-type signal (Form 10-12B/G = share
# distribution to existing shareholders, the standard spin-off registration
# mechanism) or, for the smaller remainder registered via an S-4 exchange
# offer instead, independently confirmed the pre-separation business still
# trades separately today under its own ticker. Many more candidates were
# checked and rejected: redomiciliations/tax inversions (same single company,
# re-incorporated abroad), pre-IPO holdco insertions and PE-backed re-IPOs
# (same company, no second entity created), and mergers where neither
# original party still trades separately (no live double-counting risk, just
# an unresolvable "whose history is this" question not worth guessing at) --
# see docs/US_EQUITIES_EXPANSION_PLAN.md's Phase 7 section for the full
# category breakdown and the ~60 rejected candidates.
#
# Cutoff = the date the NEW entity actually came into existence (its CIK's
# earliest filing or formerName); fundamentals rows with `end` before this
# are dropped, not blended -- there is no correct predecessor value to keep.
PREDECESSOR_CUTOFFS = {
    1675149: "2016-06-29",  # AA: Alcoa Corp, spun from Alcoa Inc (now Arconic)
    2035989: "2024-09-06",  # AMRZ: Amrize, spun from Holcim
    1501585: "2010-10-15",  # HII: Huntington Ingalls, spun from Northrop Grumman
    1524472: "2011-07-11",  # XYL: Xylem, spun from ITT Corporation
    1560385: "2012-10-19",  # FWONA: Liberty Media tracking-stock spinoff vehicle
    2064953: "2025-05-01",  # SOLS: Solstice Advanced Materials, spun from Honeywell
    1856437: "2021-04-16",  # VSXY: Victoria's Secret & Co, spun from L Brands (Bath & Body Works)
    2011286: "2024-03-07",  # AMTM: Amentum, reverse-Morris-trust spin/merger with Jacobs Engineering
    1710366: "2017-07-11",  # CNR: Core Natural Resources, ex-CONSOL Mining spinoff lineage
    1795250: "2019-12-03",  # SPHR: Sphere Entertainment, spun from MSG Entertainment
    1727263: "2018-01-23",  # FTDR: Frontdoor, spun from ServiceMaster (now Terminix)
    1965040: "2023-02-13",  # FTRE: Fortrea, spun from Labcorp
    1751788: "2018-09-07",  # DOW: Dow Inc, DowDuPont 3-way split (siblings DD/CTVA)
    1996810: "2023-10-27",  # GEV: GE Vernova, spun from General Electric
    2058873: "2025-04-24",  # Q: Qnity Electronics, spun from DuPont
    2041385: "2024-12-17",  # RAL: Ralliant, spun from Fortive
    1670541: "2016-04-26",  # ADNT: Adient, spun from Johnson Controls
    1754301: "2018-10-09",  # FOXA: Fox Corp, split from 21st Century Fox (rest acquired by Disney)
    1571123: "2013-03-07",  # SAIC: "new" SAIC, spun from original SAIC (kept name Leidos)
    1519751: "2011-05-06",  # FBIN: Fortune Brands Innovations, spun from Fortune Brands Inc
    1624794: "2014-12-02",  # CSW: CSW Industrials, spun from Capital Southwest Corporation
    1627223: "2014-12-18",  # CC: Chemours, spun from DuPont
    1932393: "2022-07-29",  # GEHC: GE HealthCare, spun from General Electric
    1679049: "2016-07-15",  # INSW: International Seaways, spun from Overseas Shipholding Group
    1673358: "2016-05-03",  # YUMC: Yum China, spun from Yum! Brands
    1636519: "2015-03-27",  # MSGS: Madison Square Garden Sports, spun from MSG Entertainment
    1868275: "2021-07-26",  # CEG: Constellation Energy, spun from Exelon
    1564708: "2012-12-21",  # NWSA: "new" News Corp, split from original News Corporation (-> 21CF/Disney)
    1740332: "2018-06-14",  # REZI: Resideo, spun from Honeywell
    1929561: "2022-06-01",  # RXO: RXO Inc, spun from XPO Logistics
    1735707: "2018-05-01",  # GTX: Garrett Motion, spun from Honeywell
    1603923: "2014-04-02",  # WFRD: Weatherford International, 2019 Ch.11 successor entity
    1935979: "2022-07-01",  # BHVN: "new" Biohaven, remainder after main business sold to Pfizer
    1921963: "2022-04-21",  # ATMU: Atmus Filtration, carved out of Cummins
    2074176: "2025-06-30",  # VNOM: Viper Energy, Diamondback Energy mineral-rights tracking entity
    1889539: "2021-12-21",  # CRBG: Corebridge Financial, carved out of AIG
    1803696: "2020-02-18",  # ADEA: Adeia, spun from Xperi (rest continues as "new" Xperi)
    1944048: "2022-08-30",  # KVUE: Kenvue, spun from Johnson & Johnson
    1637459: "2015-03-25",  # KHC: Kraft Heinz -- predates Kraft Foods Group's OWN 2012 spinoff from Mondelez
    1895262: "2021-12-20",  # NE: Noble Corp, 2021 Chapter 11 successor (old Noble legally dissolved)
}


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
            # Derive this company's real fiscal quarter-end from whichever row
            # proves the naive Dec-31 guess impossible: the latest standard
            # calendar quarter-end STRICTLY BEFORE that row's own filing date
            # (QuarterEnd subtraction guarantees this by construction). Apply
            # the same (month, day, year-offset-from-fiscal_year) template to
            # EVERY row for this CIK -- a company's fiscal year-end doesn't
            # change year to year. The year OFFSET matters, not just
            # month/day: confirmed on CRM/NTAP (real FYE Jan/Apr) -- their
            # true fiscal-year-end quarter falls in the CALENDAR YEAR BEFORE
            # the fiscal_year label (e.g. "fiscal 2005" ends Jan-2005, whose
            # nearest safe quarter-end is Dec-2004), so reusing bare
            # month/day against each row's own fiscal_year (the prior fix)
            # produced a DIFFERENT still-impossible date (e.g. "2005-12-31"
            # for a company whose year rolls into the prior calendar year).
            # An earlier version of this derivation subtracted ~2 months then
            # rounded UP to the CONTAINING quarter -- for filings less than
            # ~2 months past their own quarter boundary that rounds FORWARD
            # past the filing date itself. Confirmed on CRM, NTAP, LRCX,
            # ADSK (2026-07-28): e.g. CRM filed 2005-03-25, "-2mo" gives
            # 2005-01-25, rounding up to Q1's end (2005-03-31) -- 6 days
            # AFTER the filing that supposedly reported it.
            flagged = gap.loc[impossible].iloc[0]
            q_end = flagged["fundamentals_available_date"] - pd.offsets.QuarterEnd(n=1, startingMonth=3)
            year_offset = q_end.year - flagged["fiscal_year"]
            gap["end"] = pd.to_datetime((gap["fiscal_year"] + year_offset).astype(str)
                                         + f"-{q_end.month:02d}-{q_end.day:02d}")
            # Belt-and-suspenders: the shared template can still overshoot a
            # SPECIFIC row's own (shorter-lag) filing -- clamp any remaining
            # violator to ITS OWN safe quarter-end rather than leave an
            # impossible ordering in the data.
            still_bad = gap["end"] > gap["fundamentals_available_date"]
            if still_bad.any():
                gap.loc[still_bad, "end"] = (gap.loc[still_bad, "fundamentals_available_date"]
                                              - pd.offsets.QuarterEnd(n=1, startingMonth=3))
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
    result = (combined.sort_values(["_end_cluster", "_priority"])
                       .drop_duplicates(subset="_end_cluster", keep="first")
                       .drop(columns=["_priority", "_end_cluster"])
                       .sort_values("fundamentals_available_date")
                       .reset_index(drop=True))

    # Last line of defense, not a derivation: every fix above targets a KNOWN
    # shape of bug, but the source data itself occasionally has one too --
    # confirmed on WMT's real XBRL: a CashAndCashEquivalentsAtCarryingValue
    # fact tagged end=2012-12-31 (not even one of WMT's real Jan/Apr/Jul/Oct
    # fiscal quarter-ends) filed 2012-03-27, nine months before the period it
    # claims to describe -- a genuine upstream tagging error, not anything
    # our derivation logic could "fix" correctly, since there's no right
    # answer to derive. Rather than chase every possible upstream anomaly
    # shape, enforce the invariant itself at the one point all three tiers
    # converge: drop (and log) whatever still violates it.
    bad = result["end"] > result["fundamentals_available_date"]
    if bad.any():
        log.warning("fundamentals CIK %s: dropping %d row(s) with end > fundamentals_available_date "
                    "(source data anomaly) -- %s", cik, bad.sum(),
                    result.loc[bad, "end"].dt.date.tolist())
        result = result[~bad].reset_index(drop=True)

    cutoff = PREDECESSOR_CUTOFFS.get(cik)
    if cutoff:
        predecessor = result["end"] < pd.Timestamp(cutoff)
        if predecessor.any():
            log.warning("fundamentals CIK %s: dropping %d predecessor-entity row(s) before %s -- %s",
                        cik, predecessor.sum(), cutoff, result.loc[predecessor, "end"].dt.date.tolist())
            result = result[~predecessor].reset_index(drop=True)
    return result


def collect_fundamentals_us(tickers: list[str], fund_dir=None, workers: int = 8,
                             skip_existing: bool = False) -> None:
    """Batch driver: ticker -> CIK (tier-1 crosswalk) -> build_company_fundamentals()
    -> data/raw/us/fundamentals/{ticker}.parquet. Skips tickers the tier-1 crosswalk
    can't resolve (dead companies -- see plan §4.3) or that yield no fundamentals at
    all from either tier, logging why rather than silently continuing.

    Runs `workers` tickers concurrently (I/O-bound: mostly waiting on SEC HTTP
    responses). This does NOT lift SEC's 10 req/s cap -- http._throttle() is a
    single lock shared by every thread, so total request rate is unchanged.
    The speedup instead comes from overlapping each ticker's own per-request
    latency/backoff with other tickers' work, rather than one ticker's full
    multi-request build finishing before the next one even starts.

    `skip_existing` (default off) is for resuming a crashed/killed run only --
    it skips a ticker whose output parquet already exists rather than rebuilding
    it. Leave off for a real rebuild (e.g. after a derivation fix like the item6
    cascade or a CONCEPT_MAP addition): those need every already-collected
    company redone, not just new ones, or the fix silently never reaches rows
    already on disk.
    """
    fund_dir = fund_dir or config.US_FUNDAMENTALS_DIR
    fund_dir.mkdir(parents=True, exist_ok=True)
    cw = pd.read_parquet(crosswalk.CROSSWALK_PATH) if crosswalk.CROSSWALK_PATH.exists() \
        else crosswalk.build_crosswalk_tier1()
    ticker_to_cik = dict(zip(cw["ticker"], cw["cik"]))
    filings = pd.read_parquet(universe.FILINGS_PATH)

    def _one(ticker: str) -> None:
        if skip_existing and (fund_dir / f"{ticker}.parquet").exists():
            return
        cik = ticker_to_cik.get(ticker)
        if cik is None:
            log.info("fundamentals %s: no CIK in tier-1 crosswalk, skipping", ticker)
            return
        try:
            df = build_company_fundamentals(int(cik), filings)
        except Exception as e:
            log.warning("fundamentals %s (CIK %s): skipping after error: %s", ticker, cik, e)
            return
        if df.empty:
            log.info("fundamentals %s (CIK %s): no data from either tier", ticker, cik)
            return
        vr = validate.validate_us_fundamentals(df)
        for w in vr.warnings:
            log.warning("fundamentals %s (CIK %s): %s", ticker, cik, w)
        df.to_parquet(fund_dir / f"{ticker}.parquet", index=False)
        log.info("fundamentals %s: %d rows (%s)", ticker, len(df),
                  df["fundamentals_tier"].value_counts().to_dict())

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_one, tickers))


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(message)s")
    all_priced = sorted(p.stem for p in config.US_PRICES_DIR.glob("*.parquet"))
    collect_fundamentals_us(all_priced)
