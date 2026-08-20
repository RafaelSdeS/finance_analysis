"""sec/companyfacts.py — SEC XBRL Company Facts -> tidy, as-first-reported fundamentals.

The 2007/2009+ tier (plan §3.2/§3.3): data.sec.gov's companyfacts API returns
every XBRL-tagged fact for a CIK, each carrying its own `filed` date (when
that specific filing reported the figure) alongside the fiscal `start`/`end`
period it describes. Point-in-time correctness rule, verified 2026-07-28 on
AAPL FY2008 NetIncomeLoss (first filed 2009-10-27 at $4.834B, restated to
$6.119B a filing later): take min(filed) per (concept, start, end) -- the
value nobody could have seen before that filing date. Never key on period end.

Tag heterogeneity (the real cost of this tier, per the plan's §2.0 comparison):
revenue alone appears as Revenues, SalesRevenueNet,
RevenueFromContractWithCustomerExcludingAssessedTax, etc. depending on filer
and era -- CONCEPT_MAP below is an ordered fallback list per raw line item,
resolved PER PERIOD (_resolve_item), not once per company: confirmed on AAPL,
whose revenue tag alone moved SalesRevenueNet (2008-2018) -> "Revenues"
(2016-2018, an 8-period transition label) -> RevenueFromContractWithCustomer...
(2017-2026). An earlier version of this picked one winning concept for the
whole company and silently lost 2/3 of the available history.
"""

import json
import logging

import numpy as np
import pandas as pd

from . import http
from ..ratios import compute_ratios

log = logging.getLogger("sec")

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Nothing in this pipeline claims fundamentals data before this floor (it's
# fds.py's own EX-27 tier boundary; XBRL itself didn't exist yet) -- any XBRL
# fact with an `end` this old is a garbage/placeholder context, not real
# financial data. Real bug, confirmed on NG/CLSK/TENX (2026-07-30): each has
# a genuine fact in its own companyfacts (e.g. NG's
# StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest,
# end=1984-12-04; TENX's CashAndCashEquivalentsAtCarryingValue,
# end=1967-05-25/08-25) -- every one carries val=0, a filer-side XBRL-tooling
# artifact, not something this repo introduces. item6.py already has an
# equivalent last-line-of-defense year bound (_FISCAL_YEAR_MIN/MAX) for its
# own version of this exact failure shape; instant (balance-sheet) concepts
# have no `start` and so no duration filter to catch this at all here, unlike
# item6's table-derived rows -- this bound is the only guard for them.
_MIN_PLAUSIBLE_END = pd.Timestamp("1995-01-01")

# raw line item (compute_ratios' expected key) -> ordered XBRL concept fallback list.
# First concept present in a filer's facts wins; verified present across a 10-company
# sample (AAPL/MSFT/KO/INTC/XOM/JNJ/WMT/CAT/HD/NKE, 2026-07-28) at the rates noted.
#
# Also includes ifrs-full concept names (marked below), for foreign private issuers
# that file 20-F under IFRS instead of 10-K under US-GAAP -- verified against real
# HSBC/RIO/TECK/SAN companyfacts (2026-07-28). A domestic filer never has any
# ifrs-full facts at all, so these extra entries are harmless no-ops for it (the
# lookup just comes up empty); a 20-F filer typically has ONLY ifrs-full data, so
# these are additive, not competing, in the overwhelming majority of real cases.
# Known remaining gap: some foreign issuers (e.g. CYATY) have ZERO tagged data
# under ANY taxonomy in companyfacts -- not recoverable via this API at all.
CONCEPT_MAP = {
    "net_revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                    "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax",
                    "Revenue", "RevenueFromContractsWithCustomers"],  # ifrs-full
    # Known, deliberately unfilled gap (measured 2026-07-29 across all 1,848 collected
    # tickers): 170 (9.2%) have no net_revenue -- 163 with no column at all (none of the
    # above tags ever appear in their XBRL history) plus 7 more (AB, CFNB, COLB, CVBF, DX,
    # GS, NLY) where the column exists from an ex27/item6-tier row but is 100%-NaN in the
    # xbrl tier. All 170 cluster in banks/thrifts/mortgage REITs/BDCs/GSEs (confirmed via
    # ABCB/AGNC/ARCC/AGM/GS/NLY/COLB's raw companyfacts): they report
    # InterestIncomeExpenseNet (interest income NET of interest expense) + NoninterestIncome
    # instead of a single gross top-line figure. Deliberately NOT added as a fallback here --
    # net interest income is already a spread, not comparable to industrial companies'
    # gross Revenues, and conflating them would silently corrupt every revenue-based ratio
    # (P/S, revenue CAGR, margins) for this whole sector cluster. Same shape of gap as
    # total_debt's acknowledged banks gap below; leave NaN, don't fabricate a number.
    "net_income": ["NetIncomeLoss", "ProfitLoss"],  # ProfitLoss doubles as the ifrs-full tag
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
               "Equity", "EquityAttributableToOwnersOfParent"],  # ifrs-full
    "total_assets": ["Assets"],  # identical literal tag name in both taxonomies
    "current_assets": ["AssetsCurrent", "CurrentAssets"],  # ifrs-full
    "current_liabilities": ["LiabilitiesCurrent", "CurrentLiabilities"],  # ifrs-full
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
             "CashAndCashEquivalents"],  # ifrs-full
    "ebit": ["OperatingIncomeLoss", "ProfitLossFromOperatingActivities"],  # ifrs-full; ~80% coverage (financials often lack this subtotal)
    "gross_profit_reported": ["GrossProfit"],         # ~80% coverage; gross_margin derived if absent
    "total_debt": ["LongTermDebt", "LongTermDebtNoncurrent", "DebtLongtermAndShorttermCombinedAmount",
                   "Borrowings"],  # ifrs-full; banks (HSBC/SAN) lack a clean equivalent, same known gap as us-gaap financials
    "shares_outstanding": ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"],
    "cashflow_ops": ["NetCashProvidedByUsedInOperatingActivities",
                     "CashFlowsFromUsedInOperatingActivities"],  # ifrs-full
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}

# Items whose `end` is NOT a fiscal period-end and must not define/join a period
# cluster on equal footing with the rest. `shares_outstanding`'s dei concept
# (EntityCommonStockSharesOutstanding) is explicitly "as of the cover page date"
# -- confirmed on Coca-Cola: its `end` floats ~3 weeks from the real quarter end
# (e.g. 2009-07-24 vs. the quarter's actual 2009-07-03), which used to create a
# spurious extra cluster/row every quarter. Attached via nearest-match instead
# of being allowed to anchor a cluster.
_ATTACHED_ITEMS = {"shares_outstanding"}

# Concepts whose SEC XBRL "units" key is genuinely "shares", not "USD" -- the
# two concepts CONCEPT_MAP maps to shares_outstanding today. _facts_to_frame
# restricts these to exactly that unit key (see its docstring): a structural
# guard against ever admitting a fact mistakenly tagged under a different
# unit. Deliberately NOT extended to "USD only" for the other (dollar-
# denominated) concepts -- whether every ifrs-full foreign filer's dollar
# facts are uniformly tagged "USD" is unverified, and the current "any unit
# key" behavior for those is unchanged/working; scoping this to the one
# concept class where the expected unit is unambiguous and confirmed.
_SHARES_UNIT_CONCEPTS = {"CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"}


def fetch_companyfacts(cik: int) -> dict | None:
    resp = http.get(COMPANYFACTS_URL.format(cik=cik))
    if resp is None:
        return None
    return json.loads(resp.text)


def _facts_to_frame(facts: dict, concept: str) -> pd.DataFrame:
    """One concept's raw fact list (any taxonomy) -> tidy (start, end, val, filed, form, accn).

    Checks ifrs-full alongside us-gaap/dei -- foreign private issuers filing 20-F
    report under IFRS, tagged under a separate top-level taxonomy key that earlier
    versions of this function never looked at at all (confirmed 2026-07-28: HSBC/
    RIO/TECK/SAN each have 350-450 populated ifrs-full concepts, silently ignored).

    For a shares-denominated concept (_SHARES_UNIT_CONCEPTS), only the "shares"
    units key is accepted -- this used to iterate `units.values()` unconditionally,
    admitting a fact under ANY unit key with no check at all. Real bug, confirmed
    2026-07-31: at least 27 real US tickers' shares_outstanding is inflated by
    ~800x-1,000,000x for isolated filings (e.g. BTI's FY2019 reads 2.46 quadrillion
    instead of ~2.46 billion) -- this alone doesn't prove a wrong-unit-key origin
    for those specific rows (root cause of the bad SEC-side value stays genuinely
    uncertain), but it closes a real, previously-unvalidated gap either way, and
    is paired with reject_sequential_outliers below as a second, independent
    layer of defense.
    """
    rows = []
    expected_units = {"shares"} if concept in _SHARES_UNIT_CONCEPTS else None
    for taxonomy in ("us-gaap", "ifrs-full", "dei"):
        units = facts.get("facts", {}).get(taxonomy, {}).get(concept, {}).get("units", {})
        for unit, unit_facts in units.items():
            if expected_units is not None and unit not in expected_units:
                continue
            for fact in unit_facts:
                rows.append({**fact, "_taxonomy": taxonomy})
    if not rows:
        return pd.DataFrame(columns=["start", "end", "val", "filed", "form", "accn", "_taxonomy"])
    df = pd.DataFrame(rows)
    keep = [c for c in ("start", "end", "val", "filed", "form", "accn", "_taxonomy") if c in df.columns]
    return df[keep]


def _quarterly_only(df: pd.DataFrame) -> pd.DataFrame:
    """For duration concepts (have a `start`), keep only ~1-quarter (60-100 day) periods.

    XBRL tags the same fiscal `end` with quarterly, half-year, 9-month, AND annual
    durations at once (a 10-Q's current-quarter + YTD comparatives; a 10-K's full
    year) -- confirmed on AAPL: 96 of NetIncomeLoss's periods share an `end` with a
    different-duration sibling (e.g. end=2009-09-26 has both a 90-day Q4 figure and
    a 363-day full-year figure). Without this filter, merging line items on `end`
    collides multiple duration variants into one period. Instant (balance-sheet)
    concepts have no `start` and no such ambiguity.

    ifrs-full rows are exempt from this filter -- real gap, found extending this
    tier to 20-F filers (2026-07-28): foreign private issuers are exempt from
    quarterly reporting entirely (no 10-Q equivalent), so EVERY one of their
    duration facts is ~365 days. Applying the 60-100 day window to them dropped
    100% of their revenue/income/cashflow data, not just the annual duplicates
    the filter exists to remove for us-gaap filers -- there's no quarterly
    sibling to prefer over, so nothing needs filtering out.
    """
    if "start" not in df.columns or df.empty:
        return df
    dur = (pd.to_datetime(df["end"], errors="coerce") - pd.to_datetime(df["start"], errors="coerce")).dt.days
    is_ifrs = df["_taxonomy"] == "ifrs-full" if "_taxonomy" in df.columns else False
    return df[is_ifrs | dur.between(60, 100)]


def _annual_only(df: pd.DataFrame) -> pd.DataFrame:
    """The full fiscal-year (300-380 day) durations _quarterly_only drops -- kept
    separately so a missing standalone Q4 duration tag can be derived as FY total
    minus Q1+Q2+Q3 (see _derive_q4). ifrs-full rows are excluded here: they're
    already fully present in _quarterly_only's output (exempted from its filter
    since foreign private issuers only ever report annually), so re-including them
    as "annual" would double up rather than fill a real gap.
    """
    if "start" not in df.columns or df.empty:
        return df
    dur = (pd.to_datetime(df["end"], errors="coerce") - pd.to_datetime(df["start"], errors="coerce")).dt.days
    is_ifrs = df["_taxonomy"] == "ifrs-full" if "_taxonomy" in df.columns else False
    return df[~is_ifrs & dur.between(300, 380)]


def as_first_reported(facts: dict, concept: str, annual: bool = False) -> pd.DataFrame:
    """A concept's facts, restricted to quarterly duration (if applicable) and deduped
    to the EARLIEST filing per (start, end) period -- the as-first-reported value
    (plan §3.3), not whatever the latest restatement holds. `annual=True` restricts
    to full fiscal-year durations instead (see _annual_only) -- used only to derive
    a missing Q4 duration, not as a general-purpose alternate view.

    All four pd.to_datetime() calls in this function/its two duration helpers use
    errors="coerce" -- real bug, confirmed on MIND (CIK 926423, 2026-07-30): a raw
    XBRL fact's `start` was literally "0202-02-01" (a year-digit typo in the source
    filing, almost certainly meant "2002"), which pandas can't represent at
    nanosecond resolution and raised OutOfBoundsDatetime uncaught -- discarding
    MIND's ENTIRE fundamentals build (every tier, not just the one bad fact), the
    same "one bad filing shouldn't lose everything else" failure class already
    fixed for item6's pd.read_html crash. Coercing to NaT lets dur.between(...)
    correctly exclude just that one malformed fact (NaN comparisons are False),
    same principle as this file's cluster_period_ends already relying on NaT-safe
    comparisons elsewhere.
    """
    df = _facts_to_frame(facts, concept)
    if df.empty:
        return df
    df = _annual_only(df) if annual else _quarterly_only(df)
    if df.empty:
        return df
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df = df[df["end"] >= _MIN_PLAUSIBLE_END]
    if df.empty:
        return df
    key = ["start", "end"] if "start" in df.columns else ["end"]
    return (df.sort_values("filed")
              .drop_duplicates(subset=key, keep="first")
              .reset_index(drop=True))


def _derive_q4(quarterly: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """Fill in a missing standalone Q4 duration as FY total minus Q1+Q2+Q3.

    Most 10-K filers never tag a discrete ~90-day Q4 duration at all -- they tag
    the full fiscal year instead (10-Qs cover Q1-Q3; the 10-K's own duration fact
    is the full year) -- so _quarterly_only leaves every fiscal year-end NaN for
    flow items even though the balance sheet (instant concepts) has a real data
    point there. Confirmed on a 120-ticker sample of the real collected dataset
    (2026-07-28): net_revenue NaN 22.9% overall, but 58.7% of those NaN rows land
    in December vs 26.4% of all rows -- one missing quarter per fiscal year,
    almost exactly (median NaN count per ticker = 0.93 x rows/4).

    Only derives where EXACTLY 3 quarterly periods nest inside the FY's own
    [start, end] window -- if a company's quarterly history has its own gaps,
    the subtraction isn't safe, so it's left NaN as before rather than guessing.
    The derived row keeps the FY total's own `filed` date (conservative: the Q4
    figure isn't computable, by definition, before the FY total itself was
    filed). A filer that DOES tag a real standalone Q4 (rare) is left alone --
    derivation only fills an `end` with no existing quarterly value.
    """
    if annual.empty or quarterly.empty:
        return quarterly
    if "start" not in quarterly.columns or "start" not in annual.columns:
        # Real bug, confirmed on EPWKF (CIK 1900720, 2026-07-30): a concept can
        # have facts with no `start` at all (an instant-shaped tag used for a
        # nominally flow item, or similar filer-side tagging oddity) -- both
        # _quarterly_only and _annual_only already return such a frame
        # UNCHANGED rather than crash (their own "start" not in df.columns
        # guard), but this function never mirrored that, so a non-empty
        # start-less frame reached quarterly["start"] and raised KeyError,
        # discarding the whole company's fundamentals build over one concept.
        return quarterly
    have_ends = set(quarterly["end"])
    rows = []
    for _, fy in annual.iterrows():
        if fy["end"] in have_ends:
            continue  # a real standalone Q4 fact already covers this FY end
        nested = quarterly[(quarterly["start"] >= fy["start"]) & (quarterly["end"] < fy["end"])]
        if len(nested) != 3:
            continue
        rows.append({"start": nested["end"].max(), "end": fy["end"],
                     "val": fy["val"] - nested["val"].sum(), "filed": fy["filed"],
                     "_derived": True})
    if not rows:
        return quarterly
    derived = pd.DataFrame(rows)
    return pd.concat([quarterly, derived], ignore_index=True).sort_values("end").reset_index(drop=True)


# Flow/duration concepts where a fiscal year is the sum of its 4 quarters, so a
# missing Q4 can be derived via _derive_q4. Instant (balance-sheet) concepts --
# total_assets, current_assets/liabilities, cash, total_debt, equity,
# shares_outstanding -- are as-of-a-date snapshots, not additive across
# quarters, so deriving a "Q4" for them the same way would be meaningless.
_FLOW_ITEMS = {"net_revenue", "net_income", "ebit", "gross_profit_reported", "cashflow_ops", "capex"}

_FYE_TOLERANCE_DAYS = 20  # 52/53-week retail calendars can drift the FYE anchor by ~1-2 weeks


def ytd_to_discrete(df: pd.DataFrame, flow_cols: list[str] | None = None) -> pd.DataFrame:
    """Cumulative year-to-date flow figures (EX-27's 3/6/9/12-MOS exhibits, or a
    10-Q's YTD-only cash-flow statement) -> discrete ~3-month figures, via
    consecutive differencing within a fiscal year. Shared by the ex27 and tenq
    tiers -- see docs/US_QUARTERLY_BACKFILL_PLAN.md.

    Requires `df` to carry `period_months` (3/6/9/12) and `end`, one row per
    period. Instant (balance-sheet) columns in `flow_cols` are never present by
    construction (caller passes only its own flow-item names) and everything
    else in `df` passes through untouched.

    Never guesses: a period whose reconstruction isn't safe gets its flow
    columns NaN'd and `flows_defined=0`, mirroring this repo's existing
    informative-NaN convention (features.py's cagr_earnings_defined, merge.py's
    has_dividends) rather than shipping a possibly-wrong number. Differences
    are always taken against the RAW YTD row two periods back, never against an
    already-differenced value -- one missing link costs only the quarters that
    touch it (e.g. a missing Q1: Q2 is unrecoverable, but Q3 = raw9mo-raw6mo
    and Q4 = raw12mo-raw9mo both still reconstruct fine).
    """
    if df.empty:
        return df
    if "period_months" not in df.columns:
        raise ValueError("ytd_to_discrete requires a period_months column")
    flow_cols = ([c for c in _FLOW_ITEMS if c in df.columns] if flow_cols is None
                 else [c for c in flow_cols if c in df.columns])

    df = df.sort_values("end").reset_index(drop=True)
    months = df["period_months"]
    end_dt = pd.to_datetime(df["end"])
    implied_start = pd.Series(
        [e - pd.DateOffset(months=int(m)) if pd.notna(m) else pd.NaT for e, m in zip(end_dt, months)],
        index=df.index)

    # Fiscal-year grouping, robust to an FYE change mid-history: reset on the
    # normal 3->6->9->12->3 cycle rollover, OR when this row's implied start
    # has drifted too far from the group's own anchor (a same-direction but
    # differently-anchored YTD sequence a plain "months resets" check can't see).
    group = np.zeros(len(df), dtype=int)
    gid, group_start = 0, implied_start.iloc[0]
    for i in range(1, len(df)):
        m, prev_m, s = months.iloc[i], months.iloc[i - 1], implied_start.iloc[i]
        reset = pd.isna(m) or pd.isna(prev_m) or m <= prev_m
        drift = pd.isna(s) or pd.isna(group_start) or abs((s - group_start).days) > _FYE_TOLERANCE_DAYS
        if reset or drift:
            gid += 1
            group_start = s
        group[i] = gid

    flows_defined = pd.Series(1, index=df.index, dtype="int8")
    flows_derived = pd.Series(0, index=df.index, dtype="int8")
    out_flows = df[flow_cols].copy()

    for _, idx in pd.Series(group, index=df.index).groupby(group).groups.items():
        idx = list(idx)
        first = idx[0]
        if months.loc[first] != 3:
            out_flows.loc[first, :] = np.nan
            flows_defined.loc[first] = 0
        prev_idx = first
        for cur in idx[1:]:
            step = months.loc[cur] - months.loc[prev_idx]
            ok = (not pd.isna(step)) and step == 3
            if ok:
                diff = df.loc[cur, flow_cols] - df.loc[prev_idx, flow_cols]
                # Negative YTD-over-YTD revenue is the direct signature of a
                # restatement-basis mismatch (this quarter's YTD figure rests
                # on a different restated basis than the prior filing's) --
                # NaN rather than ship a wrong number. net_income is NOT
                # sign-checked: a genuine loss quarter must survive.
                if "net_revenue" in flow_cols and pd.notna(diff.get("net_revenue")) and diff["net_revenue"] < 0:
                    ok = False
            if ok:
                out_flows.loc[cur, :] = diff.values
                flows_derived.loc[cur] = 1
            else:
                out_flows.loc[cur, :] = np.nan
                flows_defined.loc[cur] = 0
            prev_idx = cur  # always the RAW row, never the just-computed discrete value

    result = df.drop(columns=flow_cols).copy()
    result[flow_cols] = out_flows
    result["flows_defined"] = flows_defined
    result["flows_derived"] = flows_derived
    result["period_months"] = pd.array([3] * len(result), dtype="Int8")
    return result.sort_values("end").reset_index(drop=True)


def _resolve_item(facts: dict, concepts: list[str], annual: bool = False) -> pd.DataFrame:
    """Union as_first_reported() across a line item's WHOLE fallback list, keyed by
    `end` -- NOT "first non-empty concept wins for the whole company", which silently
    truncates coverage. Confirmed on AAPL's revenue: SalesRevenueNet covers 2008-2018
    (40 periods), "Revenues" only 2016-2018 (8 periods, a transition-year label), and
    RevenueFromContractWithCustomerExcludingAssessedTax covers 2017-2026 (29 periods)
    -- a filer can change concepts mid-history, so each `end` must independently pick
    its best available concept, not inherit one company-wide choice. Where more than
    one concept reports the same `end` (transition-year overlap), the earlier concept
    in the priority list wins for that specific end.
    """
    frames = []
    for priority, concept in enumerate(concepts):
        df = as_first_reported(facts, concept, annual=annual)
        if not df.empty:
            df = df.copy()
            df["_priority"] = priority
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return (combined.sort_values(["end", "_priority"])
                     .drop_duplicates(subset="end", keep="first")
                     .drop(columns="_priority")
                     .reset_index(drop=True))


# Real stock splits/reverse-splits essentially never exceed ~10-20x in a single
# event; every real shares_outstanding scale-corruption case measured in this
# dataset (2026-07-31, 27 real tickers incl. BTI/YUM/LTM/PCG/WRB/CNA/...) was
# ~800x or larger -- wide safety margin on both sides of a genuine split.
_MAX_PLAUSIBLE_RATIO = 20.0


def reject_sequential_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """NaN out any `col` value that doesn't belong to the ticker's own dominant
    order of magnitude, walking outward (both directions) from a seed picked
    from the MAJORITY cluster -- never blindly the chronologically-first
    value. Adds a boolean f"{col}_rejected_outlier" flag column -- a rejected
    value becomes NaN, never a guessed/reconstructed number (this repo's own
    convention, see loaders.load_dividends's implausible value_per_share
    drop-and-log).

    Real bug this guards against, confirmed 2026-07-31: at least 27 real US
    tickers have one or more SEC XBRL shares_outstanding facts inflated by
    ~800x-1,000,000x relative to that same ticker's own other filings (e.g.
    BTI's FY2019 reads 2.46 quadrillion instead of ~2.46 billion), corrupting
    market_cap and every ratio derived from it downstream.

    A single forward-only pass comparing each value only to the LAST ACCEPTED
    one (not merely the previous raw value -- one ticker, LTM, has been wrong
    for 4 CONSECUTIVE fiscal years, so a rejected value must never become the
    new baseline) turned out to have its own real failure mode, found the hard
    way on a live recollection: CCI and TFC's OWN FIRST-EVER XBRL-era values
    (2008-2009, when this tier begins) are themselves the corrupted ones. A
    naive forward walk anchors on that first (bad) value and then rejects
    every one of the ~67 genuinely correct quarters that follow it, because
    each looks "implausible" relative to the wrong baseline -- turning a
    2-row bug into a 67-row one. Fixed by seeding the walk from the ticker's
    MAJORITY magnitude cluster (values grouped by rounded log10) instead of
    index 0, then walking forward from the seed AND separately backward from
    the seed (two independent last-accepted trackers) -- correctly flags an
    isolated bad quarter surrounded by good ones (BTI/YUM-shaped), a
    persistent bad run following good history (LTM-shaped, walking forward
    from a seed in the good cluster), AND a persistent bad run at the START of
    history followed by good data (CCI/TFC-shaped, walking backward from a
    seed in the good cluster now correctly flags the early bad values instead
    of the good majority).

    Generic over `col` rather than hardcoded to shares_outstanding: applies to
    every current/future member of _ATTACHED_ITEMS uniformly (see its caller),
    not just today's one member. Deliberately NOT applied to flow/dollar
    concepts (net_income, equity, total_assets, ...) -- those legitimately
    have far higher period-over-period volatility for smaller/cyclical/growth
    companies and would need a fundamentally different plausibility model
    (relative to a slower-moving anchor, not a fixed ratio threshold); a
    follow-up opportunity, not attempted here.

    Known, accepted limitation: an exact tie in cluster SIZE (e.g. LTM's real
    4-good/4-bad split) has no principled winner from magnitude alone -- the
    tiebreak (earliest-occurring cluster wins) is deterministic but not
    guaranteed semantically correct. Rare; not otherwise observed across the
    27 real cases this was built against. A single ticker with only ONE ever
    value has no basis to judge plausibility either way and is kept as-is.
    """
    df = df.reset_index(drop=True)
    vals = df[col]
    rejected = pd.Series(False, index=df.index)
    rejected[vals.notna() & (vals <= 0)] = True

    valid_idx = vals[vals.notna() & (vals > 0)].index
    if len(valid_idx) > 1:
        buckets = np.log10(vals[valid_idx]).round().astype(int)
        counts = buckets.value_counts()
        candidates = counts[counts == counts.max()].index
        # tie-break: the candidate cluster whose earliest member is
        # chronologically first (buckets' index is already chronological --
        # df was end-sorted and reset to a plain RangeIndex before this point)
        majority_bucket = min(candidates, key=lambda b: (buckets == b).idxmax())
        majority_idx = list(buckets[buckets == majority_bucket].index)
        seed = majority_idx[0]
        seed_pos = list(valid_idx).index(seed)

        def _walk(order):
            last_good = vals[seed]
            for i in order:
                val = vals[i]
                ratio = val / last_good
                if ratio > _MAX_PLAUSIBLE_RATIO or ratio < 1 / _MAX_PLAUSIBLE_RATIO:
                    rejected.at[i] = True
                else:
                    last_good = val

        _walk(valid_idx[seed_pos + 1:])    # forward from the seed
        _walk(valid_idx[:seed_pos][::-1])  # backward from the seed

    df[f"{col}_rejected_outlier"] = rejected
    df.loc[rejected, col] = np.nan
    return df


_CLUSTER_TOL_DAYS = 10  # real quarters are ~90 days apart -- 9x margin below that


def cluster_period_ends(dates) -> dict:
    """Group period-end dates within `_CLUSTER_TOL_DAYS` of each other into one
    cluster, mapped to a single representative date.

    Real bug, found scaling past a single company's summary view (2026-07-28):
    different XBRL concepts for the SAME fiscal quarter can carry slightly
    different `end` dates. Confirmed on Coca-Cola: NetIncomeLoss tags Q2 2008 as
    ending 2008-06-27 (the last business day of its actual fiscal period), while
    StockholdersEquity tags "the same" quarter 2008-06-28. An exact-date merge
    (the original approach) fragments one real quarter into several near-empty
    rows -- confirmed directly: KO's XBRL era produced 148 rows this way,
    versus ~70 real quarters once fixed. Greedy chaining is safe here only
    because real distinct quarters are ~90 days apart, 9x this tolerance --
    it will NOT falsely merge two genuinely different quarters.
    """
    uniq = sorted(set(dates))
    if not uniq:
        return {}
    clusters, current = [], [uniq[0]]
    for d in uniq[1:]:
        if (d - current[-1]).days <= _CLUSTER_TOL_DAYS:
            current.append(d)
        else:
            clusters.append(current)
            current = [d]
    clusters.append(current)
    return {d: c[len(c) // 2] for c in clusters for d in c}


_LEGAL_PERIOD_MONTHS = np.array([3, 6, 9, 12])


def _period_months(start: pd.Series, end: pd.Series) -> pd.Series:
    """Period length in months implied by (end - start), snapped to the nearest
    SEC-legal length (3/6/9/12) rather than a blind round() -- real quarters vary
    +-10 days from 91 (52/53-week retail calendars), which a plain round() can
    misclassify at the edges of _quarterly_only's 60-100 day admission window.
    Distinguishes quarterly flow magnitudes from annual ones in the same
    columns; see docs/US_QUARTERLY_BACKFILL_PLAN.md."""
    idx = start.index if hasattr(start, "index") else end.index
    months = (pd.to_datetime(end, errors="coerce") - pd.to_datetime(start, errors="coerce")).dt.days / 30.44
    result = pd.Series(pd.NA, index=idx, dtype="Int8")
    valid = months.notna()
    if valid.any():
        m = months[valid].to_numpy()
        snap = np.abs(m[:, None] - _LEGAL_PERIOD_MONTHS[None, :]).argmin(axis=1)
        result.loc[valid] = _LEGAL_PERIOD_MONTHS[snap]
    return result


def extract_line_items(facts: dict) -> pd.DataFrame:
    """One row per fiscal quarter (period-end dates clustered via
    cluster_period_ends, not merged on exact equality), every CONCEPT_MAP line
    item resolved via its fallback list (each as-first-reported, per-period per
    _resolve_item). Each item keeps its own `{item}_filed` date plus an overall
    `fundamentals_available_date` = MAX across populated items' filed dates --
    the conservative (never-early) bundling date merge_asof downstream must key
    on (plan §5.2): using the max, not the min, guarantees no single item in the
    row is ever exposed before it was genuinely public, even though different
    concepts for the same quarter can individually become known at different
    times (e.g. a balance-sheet item appearing as a prior-period comparative
    before the income-statement figure for the same quarter is filed)."""
    resolved = {}
    for item, concepts in CONCEPT_MAP.items():
        df = _resolve_item(facts, concepts)
        if item in _FLOW_ITEMS and not df.empty:
            df = _derive_q4(df, _resolve_item(facts, concepts, annual=True))
        if df.empty:
            continue
        df = df.rename(columns={"val": item, "filed": f"{item}_filed"})
        keep = ["end", item, f"{item}_filed"]
        if item in _FLOW_ITEMS:
            # Per-item period length + derived-flag, carried through the merge
            # below and collapsed to one row-level period_months/flows_derived
            # pair afterward -- see docs/US_QUARTERLY_BACKFILL_PLAN.md.
            start = df["start"] if "start" in df.columns else pd.Series(pd.NaT, index=df.index)
            df[f"{item}_period_months"] = _period_months(start, df["end"])
            # NaN-safe equality (not fillna+astype, which pandas warns is a
            # deprecated silent downcast on this mixed object-dtype column):
            # NaN == True evaluates False, exactly the "not derived" default wanted.
            df[f"{item}_derived"] = (df["_derived"] == True) if "_derived" in df.columns \
                else pd.Series(False, index=df.index)  # noqa: E712
            keep += [f"{item}_period_months", f"{item}_derived"]
        resolved[item] = df[keep]
    if not resolved:
        return pd.DataFrame()

    period_items = {k: v for k, v in resolved.items() if k not in _ATTACHED_ITEMS}
    attached_items = {k: v for k, v in resolved.items() if k in _ATTACHED_ITEMS}
    if not period_items:
        return pd.DataFrame()

    all_ends = pd.concat([df["end"] for df in period_items.values()])
    cluster_map = cluster_period_ends(all_ends)

    out = None
    for item, df in period_items.items():
        df = df.copy()
        df["end"] = df["end"].map(cluster_map)
        # A cluster could hold >1 row for one item only if two of ITS OWN
        # distinct periods fell within the tolerance window -- shouldn't
        # happen (an item's own periods are naturally ~90 days apart), but
        # keep="first" (already sorted by end) degrades safely if it did.
        df = df.drop_duplicates(subset="end", keep="first")
        out = df if out is None else out.merge(df, on="end", how="outer")
    out = out.sort_values("end").reset_index(drop=True)

    # Row-level period_months (median across flow items agreeing on this
    # cluster's period -- a single mistagged item shouldn't swing the whole
    # row) + flows_derived (true if ANY flow item on this row came from
    # _derive_q4's subtraction, not a directly filed fact). xbrl never NaNs a
    # flow for being "unsafe to reconstruct" (_derive_q4 only fires when
    # exactly 3 quarters nest safely) -- so flows_defined is always 1 here;
    # it earns its keep on the ex27/tenq tiers, which do attempt and can
    # reject risky reconstructions (see companyfacts.ytd_to_discrete).
    pm_cols = [c for c in out.columns if c.endswith("_period_months")]
    derived_cols = [c for c in out.columns if c.endswith("_derived")]
    out["period_months"] = (out[pm_cols].median(axis=1, skipna=True).round().astype("Int8")
                             if pm_cols else pd.Series(pd.NA, index=out.index, dtype="Int8"))
    out["flows_derived"] = (out[derived_cols].any(axis=1).astype("int8")
                             if derived_cols else pd.Series(0, index=out.index, dtype="int8"))
    out["flows_defined"] = pd.Series(1, index=out.index, dtype="int8")
    out = out.drop(columns=pm_cols + derived_cols)

    # Attached items (shares_outstanding): nearest-match onto the real period
    # grid rather than joined on exact/clustered end -- their own `end` is a
    # different kind of date entirely (see _ATTACHED_ITEMS), not a competing
    # fiscal quarter. Outlier-reject BEFORE the nearest-match attach, on the
    # item's own true chronological sequence (see reject_sequential_outliers).
    for item, df in attached_items.items():
        df = df.sort_values("end")
        df = reject_sequential_outliers(df, item)
        out = pd.merge_asof(out.sort_values("end"), df, on="end",
                             direction="nearest", tolerance=pd.Timedelta(days=45))

    filed_cols = [c for c in out.columns if c.endswith("_filed")]
    out["fundamentals_available_date"] = out[filed_cols].max(axis=1)
    return out.sort_values("end").reset_index(drop=True)


def compute_us_ratios(line_items: pd.DataFrame, close_price_by_date: pd.Series | None = None) -> pd.DataFrame:
    """Row-wise compute_ratios(unit_scale=1) over extracted line items. `close_price_by_date`
    (a Series indexed by `end`, e.g. from the ticker's own price history) fills market-cap-
    dependent ratios (pl/pvp/ev_*); left NaN if not supplied -- this function only needs the
    line items to compute margins/ROE/ROA/debt_equity/current_ratio etc."""
    rows = []
    for _, r in line_items.iterrows():
        raw = r.to_dict()
        if close_price_by_date is not None:
            raw["close_price"] = close_price_by_date.get(r["end"], np.nan)
        raw.setdefault("ebitda", np.nan)  # XBRL has no single EBITDA tag; not derived here
        ratios = compute_ratios(raw, unit_scale=1)
        rows.append({**raw, **ratios})
    return pd.DataFrame(rows)
