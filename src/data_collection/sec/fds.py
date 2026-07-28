"""sec/fds.py — EX-27 "Financial Data Schedule" parser (1994-2001 tier).

Structured tag-value financials the SEC required inside pre-XBRL filings,
eliminated in 2001 (File No. S7-05-00). Real prevalence, measured against a
224-filing random sample across 1994-2001 (2026-07-28, not the earlier
8-filing spot-check) -- IMPORTANT correction to the plan's original "1994-2001"
framing:

    1994: 10.7% (11%, even restricted to primary non-amendment 10-Ks: 8%)
    1995: 82.1%    1996: 64.3%    1997: 78.6%
    1998: 67.9%    1999: 71.4%    2000: 75.0%
    2001: 0.0% (full-year sample; the earlier Q1-only spot-check found some
                early-2001 filings still carrying it, before elimination)

The USABLE window is 1995-2000, not 1994-2001. 1994 was EX-27's first, low-
adoption year (~90% miss rate even excluding amendments) and 2001 is
essentially post-elimination. Both edges are parsed anyway (whatever exists
is used), but coverage should not be assumed there -- measure it per year,
same as everywhere else in this pipeline.

Overall prevalence across the full sample: 126/224 = 56.3% -- clears the
plan's Phase 0 ~50% gate, but only because 1995-2000 pulls the average up;
judge the tier by its 1995-2000 core, not the blended number.

Only ARTICLE 5 (commercial/industrial) is mapped to compute_ratios' schema:
83% of EX-27-bearing filings in the sample (105/126). Articles 6/7/9/UT
(investment companies/insurance/banks/utilities) have entirely different tag
vocabularies -- flagged via `fds_article` rather than silently misparsed
against the wrong schema; extending to them is future work, not attempted.
"""

import logging
import re

import numpy as np
import pandas as pd

from . import http
from ..yf_collectors import compute_ratios

log = logging.getLogger("sec")

_EX27_BLOCK = re.compile(r"<TYPE>EX-27[^\n]*\n(.*?)(?=<TYPE>|</DOCUMENT>|\Z)", re.S)
_TAG_VALUE = re.compile(r"<([A-Z][A-Z0-9&-]*)>\s*([^\n<]*)")

# Article 5 (commercial/industrial) -> compute_ratios' raw-item schema. Verified
# against Coca-Cola's real FY1994 EX-27 (2026-07-28): TOTAL-ASSETS 13,873 *
# MULTIPLIER 1,000,000 = $13.873B, matching Coca-Cola's published 1994 10-K.
ARTICLE_5_MAP = {
    "net_income": "NET-INCOME",
    "net_revenue": "TOTAL-REVENUES",
    "total_assets": "TOTAL-ASSETS",
    "current_assets": "CURRENT-ASSETS",
    "current_liabilities": "CURRENT-LIABILITIES",
    "cash": "CASH",
    "total_debt": "BONDS",
    "cost_of_revenue": "CGS",
}
# NOT in EX-27 Article 5 at all (unlike XBRL/EX-27's other tiers): shares_outstanding,
# ebitda, equity (COMMON+OTHER-SE approximates it, mapped separately below since it's
# a sum of two tags, not a single one), cash-flow statement. Left NaN, not derived.


def fetch_filing_text(filename: str) -> str | None:
    resp = http.get(f"https://www.sec.gov/Archives/{filename}")
    return resp.text if resp is not None else None


def parse_fds(text: str) -> list[dict]:
    """ALL EX-27 exhibits' raw tag-value dicts in this filing -- a single filing can
    bundle MULTIPLE (current year + restated prior-year comparatives). Confirmed on
    Coca-Cola's real 1998-03-09 10-K (2026-07-28): it carries three EX-27 exhibits at
    once -- EX-27.1 (FY1995, restated), EX-27.2 (FY1996, restated), EX-27.3 (FY1997,
    the actual current year this filing exists to report). An earlier version of this
    used .search() (first match only), which silently kept EX-27.1's restated FY1995
    figures and discarded FY1996 and FY1997 entirely -- losing the current year's data
    on every filing that bundles comparatives this way. Empty list if none present.
    """
    return [{tag: val.strip() for tag, val in _TAG_VALUE.findall(m.group(1))}
            for m in _EX27_BLOCK.finditer(text)]


def _to_number(s: str) -> float:
    s = (s or "").replace(",", "").strip()
    if not s or s.startswith("<"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def extract_line_items(tags: dict) -> dict:
    """Article-5 raw line items, scaled by <MULTIPLIER>. Non-Article-5 filings
    return just {fds_article, fds_multiplier} -- caller can see why nothing else
    is populated rather than getting silently wrong numbers under Article 5's tags.
    """
    article = (tags.get("ARTICLE") or "").strip()
    multiplier = _to_number(tags.get("MULTIPLIER", "1"))
    multiplier = 1.0 if np.isnan(multiplier) or multiplier == 0 else multiplier
    out = {"fds_article": article, "fds_multiplier": multiplier}
    if article != "5":
        return out

    for item, tag in ARTICLE_5_MAP.items():
        out[item] = _to_number(tags.get(tag)) * multiplier
    common = _to_number(tags.get("COMMON"))
    other_se = _to_number(tags.get("OTHER-SE"))
    out["equity"] = (np.nan_to_num(common) + np.nan_to_num(other_se)) * multiplier \
        if not (np.isnan(common) and np.isnan(other_se)) else np.nan
    return out


def _parse_fds_date(s: str | None) -> pd.Timestamp:
    """EX-27's <FISCAL-YEAR-END> is "DEC-31-1994"-style, not ISO -- pandas parses it
    fine with dayfirst=False, but NaT on anything malformed rather than raising."""
    return pd.to_datetime(s, format="%b-%d-%Y", errors="coerce") if s else pd.NaT


def extract_and_compute(text: str) -> list[dict]:
    """One filing's text -> a list of results, ONE PER EX-27 exhibit (a filing can
    bundle several -- see parse_fds's docstring). Each result is line items +
    compute_ratios(unit_scale=1) (already full-dollar via <MULTIPLIER>) + its own
    `fds_period_end` (from that exhibit's <FISCAL-YEAR-END>). Empty list if no
    EX-27 exhibit exists in this filing."""
    results = []
    for tags in parse_fds(text):
        items = extract_line_items(tags)
        period_end = _parse_fds_date(tags.get("FISCAL-YEAR-END"))
        if items.get("fds_article") != "5":
            results.append({**items, "fds_period_end": period_end})
            continue
        results.append({**items, **compute_ratios(items, unit_scale=1), "fds_period_end": period_end})
    return results


def build_cik_history(cik: int, filings: pd.DataFrame) -> pd.DataFrame:
    """Every qualifying (Article-5) EX-27 exhibit for one CIK across the Phase 3
    filings table -- including every comparative exhibit a single filing bundles,
    not just its first -- each stamped with `fundamentals_available_date` = the
    filing's real EDGAR date_filed -- NOT the fiscal period end, per the plan's
    §5.2 point-in-time rule (identical reasoning to companyfacts.py's `filed`).
    Where a period is reported by more than one filing (a later 10-K restating an
    earlier year as a comparative, or a 10-K/A amendment), the EARLIEST filing
    wins -- same as-first-reported rule as the XBRL tier (§3.3).
    """
    cik_filings = filings[(filings["cik"] == cik) & (filings["form_type"].str.startswith("10-K"))]
    rows = []
    for row in cik_filings.itertuples():
        text = fetch_filing_text(row.filename)
        if text is None:
            continue
        for result in extract_and_compute(text):
            if result.get("fds_article") != "5":
                continue
            rows.append({**result, "cik": cik, "fundamentals_available_date": row.date_filed,
                         "fds_form": row.form_type, "fds_filename": row.filename})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # as-first-reported: whichever filing disclosed a given fiscal period EARLIEST
    # wins, whether that's the period's own original filing or a later filing's
    # bundled comparative exhibit reporting it first for some other reason.
    return (df.sort_values("fundamentals_available_date")
              .drop_duplicates(subset="fds_period_end", keep="first")
              .sort_values("fds_period_end")
              .reset_index(drop=True))


def measure_prevalence(filings: pd.DataFrame, years=range(1994, 2002), sample_per_year=28,
                        random_state=42) -> pd.DataFrame:
    """Phase 0's gate check: real EX-27 prevalence across a random sample of 10-K
    variant filings, per year. `filings` is the Phase 3 filings table (or any
    frame with cik/form_type/date_filed/filename columns)."""
    tenk = filings[filings["form_type"].str.startswith("10-K")]
    rows = []
    for year in years:
        grp = tenk[tenk["date_filed"].dt.year == year]
        if grp.empty:
            continue
        sample = grp.sample(min(sample_per_year, len(grp)), random_state=random_state)
        for row in sample.itertuples():
            text = fetch_filing_text(row.filename)
            tags = parse_fds(text) if text else None
            rows.append({"year": year, "has_ex27": tags is not None,
                         "article": (tags or {}).get("ARTICLE")})
    return pd.DataFrame(rows)
