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


def parse_fds(text: str) -> dict | None:
    """One filing's raw EX-27 tag-value dict, or None if no EX-27 present."""
    m = _EX27_BLOCK.search(text)
    if m is None:
        return None
    return {tag: val.strip() for tag, val in _TAG_VALUE.findall(m.group(1))}


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


def extract_and_compute(text: str) -> dict | None:
    """One filing's text -> line items + compute_ratios(unit_scale=1) (EX-27 values
    are already scaled to full dollars by extract_line_items' multiplier). None if
    no EX-27 present."""
    tags = parse_fds(text)
    if tags is None:
        return None
    items = extract_line_items(tags)
    if items.get("fds_article") != "5":
        return items
    return {**items, **compute_ratios(items, unit_scale=1)}


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
