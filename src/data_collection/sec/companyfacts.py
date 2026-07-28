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
from ..yf_collectors import compute_ratios

log = logging.getLogger("sec")

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# raw line item (compute_ratios' expected key) -> ordered XBRL concept fallback list.
# First concept present in a filer's facts wins; verified present across a 10-company
# sample (AAPL/MSFT/KO/INTC/XOM/JNJ/WMT/CAT/HD/NKE, 2026-07-28) at the rates noted.
CONCEPT_MAP = {
    "net_revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                    "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "ebit": ["OperatingIncomeLoss"],                 # ~80% coverage (financials often lack this subtotal)
    "gross_profit_reported": ["GrossProfit"],         # ~80% coverage; gross_margin derived if absent
    "total_debt": ["LongTermDebt", "LongTermDebtNoncurrent", "DebtLongtermAndShorttermCombinedAmount"],
    "shares_outstanding": ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"],
    "cashflow_ops": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}


def fetch_companyfacts(cik: int) -> dict | None:
    resp = http.get(COMPANYFACTS_URL.format(cik=cik))
    if resp is None:
        return None
    return json.loads(resp.text)


def _facts_to_frame(facts: dict, concept: str) -> pd.DataFrame:
    """One concept's raw fact list (any taxonomy) -> tidy (start, end, val, filed, form, accn)."""
    rows = []
    for taxonomy in ("us-gaap", "dei"):
        units = facts.get("facts", {}).get(taxonomy, {}).get(concept, {}).get("units", {})
        for unit_facts in units.values():
            rows.extend(unit_facts)
    if not rows:
        return pd.DataFrame(columns=["start", "end", "val", "filed", "form", "accn"])
    df = pd.DataFrame(rows)
    keep = [c for c in ("start", "end", "val", "filed", "form", "accn") if c in df.columns]
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
    """
    if "start" not in df.columns or df.empty:
        return df
    dur = (pd.to_datetime(df["end"]) - pd.to_datetime(df["start"])).dt.days
    return df[dur.between(60, 100)]


def as_first_reported(facts: dict, concept: str) -> pd.DataFrame:
    """A concept's facts, restricted to quarterly duration (if applicable) and deduped
    to the EARLIEST filing per (start, end) period -- the as-first-reported value
    (plan §3.3), not whatever the latest restatement holds.
    """
    df = _facts_to_frame(facts, concept)
    if df.empty:
        return df
    df = _quarterly_only(df)
    if df.empty:
        return df
    df["filed"] = pd.to_datetime(df["filed"])
    key = ["start", "end"] if "start" in df.columns else ["end"]
    return (df.sort_values("filed")
              .drop_duplicates(subset=key, keep="first")
              .reset_index(drop=True))


def _resolve_item(facts: dict, concepts: list[str]) -> pd.DataFrame:
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
        df = as_first_reported(facts, concept)
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


def extract_line_items(facts: dict) -> pd.DataFrame:
    """One row per fiscal `end`, every CONCEPT_MAP line item resolved via its fallback
    list (each as-first-reported, per-period per _resolve_item). Each item keeps its own
    `{item}_filed` date plus an overall `fundamentals_available_date` = MAX across
    populated items' filed dates -- the conservative (never-early) bundling date
    merge_asof downstream must key on (plan §5.2): using the max, not the min,
    guarantees no single item in the row is ever exposed before it was genuinely
    public, even though different concepts for the same quarter can individually
    become known at different times (e.g. a balance-sheet item appearing as a
    prior-period comparative before the income-statement figure for the same
    quarter is filed)."""
    resolved = {}
    for item, concepts in CONCEPT_MAP.items():
        df = _resolve_item(facts, concepts)
        if not df.empty:
            resolved[item] = df.rename(columns={"val": item, "filed": f"{item}_filed"})[
                ["end", item, f"{item}_filed"]]
    if not resolved:
        return pd.DataFrame()

    out = None
    for item, df in resolved.items():
        out = df if out is None else out.merge(df, on="end", how="outer")

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
