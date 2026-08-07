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

from ..yf_collectors import compute_ratios
from .. import config, validate
from . import companyfacts, crosswalk, fds, selected_financial_data, tenq, universe

log = logging.getLogger("sec")

# XBRL preferred over EX-27/tenq/Item 6 on any overlapping fiscal period
# (richer, more reliable -- see plan §2.0); EX-27 above tenq (structured
# tag-value data beats HTML table scraping, and they only overlap in early
# 2001); tenq above item6 (real quarterly resolution, vs. item6's annual-only,
# most parse-fragile tier in this pipeline). In practice the tiers shouldn't
# overlap much (EX-27 usably ends 2000, tenq covers 2001-2006, item6 covers
# the same 2001-2006 window as a fallback, XBRL starts ~2006-2007), but
# resolve deterministically rather than leave a silent duplicate `end` if one does.
_TIER_PRIORITY = {"xbrl": 0, "ex27": 1, "tenq": 2, "item6": 3}


def _derive_annual_q4(quarters: pd.DataFrame, annual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Q4 = item6's annual FY total - sum(tenq's Q1+Q2+Q3), for fiscal years
    where tenq covers exactly 3 real quarters that plausibly nest inside
    item6's FY window (docs/US_QUARTERLY_BACKFILL_PLAN.md Phase 4). Returns
    (quarters + derived Q4 rows, annual rows NOT consumed) -- a consumed
    annual row is REMOVED here, not left for the general tier-priority dedup
    to resolve later: item6's `end` is only a Dec-31-ish guess (see the
    non-calendar-FYE correction above), so relying on cluster_period_ends's
    10-day tolerance to catch the collision could silently ship BOTH the
    annual (12mo) and derived (3mo) row for the same real period if the
    guess misses by more than that -- the exact schema defect this project
    exists to fix, reintroduced at the fix site.

    Guards, all -> no Q4 derived, the annual row is left untouched (never
    fabricate): exactly 3 quarters must nest in (fy_end-370d, fy_end-20d];
    consecutive quarter spacing 60-120 days; Q3 within 60-120 days of the FY
    end; and the derived Q4's revenue must be a plausible share of the FY
    total (0-60%) -- the one genuinely INDEPENDENT check available here
    (Q1+Q2+Q3+Q4==FY is circular once Q4 is *defined* as the residual), and
    it catches a unit-multiplier mismatch between the item6 table and tenq's
    table, a wrong fiscal-year match, or a mid-year restatement, all at once,
    with no per-company tuning -- the same shape of check as fundamentals.py's
    own _FLOORS, just relative instead of absolute.

    The derived row keeps EVERY OTHER item6 column as-is (total_assets,
    equity, eps_basic, dividends_per_share, item6_form/filename...) -- those
    describe the exact same real date (fy_end) whether the flow columns are
    annual or Q4-only, same convention as companyfacts._derive_q4's own
    instant-columns-pass-through design.
    """
    if quarters.empty or annual.empty:
        return quarters, annual
    quarters = quarters.sort_values("end").reset_index(drop=True)
    flow_cols = [c for c in ("net_revenue", "net_income", "cost_of_revenue")
                 if c in annual.columns and c in quarters.columns]
    derived_rows, consumed_idx = [], []
    for idx, fy in annual.iterrows():
        fy_end = fy["end"]
        window = quarters[(quarters["end"] > fy_end - pd.Timedelta(days=370))
                           & (quarters["end"] <= fy_end - pd.Timedelta(days=20))].sort_values("end")
        if len(window) != 3:
            continue
        gaps = window["end"].diff().dt.days.dropna()
        if not gaps.between(60, 120).all():
            continue
        if not (60 <= (fy_end - window["end"].iloc[-1]).days <= 120):
            continue
        fy_revenue, q_revenue_sum = fy.get("net_revenue"), window["net_revenue"].sum()
        if pd.isna(fy_revenue) or pd.isna(q_revenue_sum):
            continue
        derived_revenue = fy_revenue - q_revenue_sum
        if not (0 <= derived_revenue <= 0.60 * fy_revenue):
            continue

        row = fy.to_dict()
        for c in flow_cols:
            fy_val, q_sum = row.get(c), window[c].sum()
            row[c] = (fy_val - q_sum) if pd.notna(fy_val) and pd.notna(q_sum) else float("nan")
        row["period_months"] = 3
        row["flows_derived"] = 1
        row["flows_defined"] = 1
        derived_rows.append(row)
        consumed_idx.append(idx)

    if not derived_rows:
        return quarters, annual
    derived = pd.DataFrame(derived_rows)
    ratios = derived.apply(lambda r: compute_ratios(r.to_dict(), unit_scale=1), axis=1, result_type="expand")
    derived[ratios.columns] = ratios
    quarters = (pd.concat([quarters, derived], ignore_index=True, sort=False)
                  .sort_values("end").reset_index(drop=True))
    annual = annual.drop(index=consumed_idx).reset_index(drop=True)
    return quarters, annual

# A handful of CIKs are genuine corporate spinoffs/split-offs/post-bankruptcy
# successors whose SEC companyfacts XBRL data includes PRE-SEPARATION
# comparative financials describing a legally different predecessor entity --
# not this company's own standalone history. Unlike CIK_OVERRIDES
# (crosswalk.py's shell-CIK fix), the CIK here is already the right one for
# the ticker; the problem is upstream SEC XBRL data blending a predecessor's
# books into the new registrant's own facts, an inverse-shaped version of the
# same "wrong entity's numbers under this ticker" risk. Found auditing a
# "751/1,848 tickers have a fundamentals gap" symptom (2026-07-29): most of
# that gap turned out to be the selected_financial_data.py cascade bug (see its
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

    xbrl = pd.DataFrame()
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
        # NOT appended to frames yet -- infer_multiplier_from_trusted_tiers
        # below needs quarters/gap (tenq/item6) built first as a reference.

    quarters = tenq.build_cik_history(cik, filings)
    if not quarters.empty:
        quarters["fundamentals_tier"] = "tenq"

    gap = selected_financial_data.build_cik_history(cik, filings)
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

    # ex27 rows with NO explicit <MULTIPLIER> anywhere in this CIK's own ex27
    # history (fds._fill_missing_multipliers had no same-tier sibling to
    # borrow from at all -- confirmed real on AEO/ATNI/AUSI/... 2026-08-06)
    # get one more chance against the other, individually more reliable,
    # tiers built above. See fds.infer_multiplier_from_trusted_tiers.
    if not ex27.empty:
        trusted = pd.concat([f for f in (xbrl, quarters, gap) if not f.empty], ignore_index=True, sort=False)
        ex27 = fds.infer_multiplier_from_trusted_tiers(ex27, trusted)
        frames.append(ex27)

    # tenq/item6 tags must already be set above -- _derive_annual_q4's
    # derived Q4 rows inherit the item6 row's own fundamentals_tier ("item6",
    # since most of the derived row's columns still come from it), and must
    # not be clobbered by a blanket post-hoc tag assignment on `quarters`.
    if not quarters.empty and not gap.empty:
        quarters, gap = _derive_annual_q4(quarters, gap)
    if not quarters.empty:
        frames.append(quarters)
    if not gap.empty:
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

    # Absolute-floor rejection: a company that cleared the universe gate
    # cannot genuinely have $20 of total assets. Confirmed 2026-08-01: CVBF's
    # 2006/2007 item6 rows read total_assets 0.0/20.0 against a real ~$6.5B
    # (its very next xbrl-tier row); BPOP's item6 net_income reads as low as
    # 740 against other years' ~$100M+ -- a per-filing unit-multiplier
    # ((in thousands)/(in millions)) misapplication, not a uniform tier
    # offset (see selected_financial_data.detect_unit_multiplier / fds.py's
    # MULTIPLIER handling -- both tiers DO implement scaling, just not always
    # correctly per-filing). NaN'd, never guessed (loaders.load_dividends'
    # convention) -- deliberately NOT companyfacts._reject_sequential_outliers
    # here: that seeds from a ticker's DOMINANT magnitude cluster, right for
    # shares_outstanding but wrong for a company that legitimately grew 100x
    # over 30 years of real filings.
    # ponytail: floors, not a parser fix -- the underlying item6 year-label
    # misparse (some rows' `end` traces back to a filing >10y later, see
    # audit doc §3) is a separate, unattempted project.
    _FLOORS = {"total_assets": 1e5, "net_revenue": 1e4, "equity": 1e4}
    for col, floor in _FLOORS.items():
        if col not in result.columns:
            continue
        bad_scale = result[col].abs().between(0, floor, inclusive="left") & result[col].notna()
        if bad_scale.any():
            log.warning("fundamentals CIK %s: rejecting %d implausible %s value(s) "
                        "(< %.0f, below any real public filer's floor) -- now NaN",
                        cik, bad_scale.sum(), col, floor)
            result.loc[bad_scale, col] = float("nan")

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
