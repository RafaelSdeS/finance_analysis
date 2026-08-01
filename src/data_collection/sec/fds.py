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

Quarterly (Phase 2, docs/US_QUARTERLY_BACKFILL_PLAN.md): 10-Qs in this same
window carry an EX-27 too (3-MOS/6-MOS/9-MOS), reporting cumulative YTD flow
figures -- build_cik_history turns these into discrete ~3-month figures via
companyfacts.ytd_to_discrete. The prevalence numbers above were measured on
10-Ks only; quarterly coverage is not separately re-measured, same caveat as
every other tier's "verified on" column in the plan doc.
"""

import logging
import re

import numpy as np
import pandas as pd

from . import companyfacts, http
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


_QUARTERLY_PERIOD_TYPES = {"3-MOS": 3, "6-MOS": 6, "9-MOS": 9}


def extract_line_items(tags: dict) -> dict:
    """Article-5 raw line items, scaled by <MULTIPLIER>. Non-Article-5 filings
    return just {fds_article, fds_multiplier} -- caller can see why nothing else
    is populated rather than getting silently wrong numbers under Article 5's tags.

    Accepts <PERIOD-TYPE> YEAR (annual, from a 10-K) and 3-MOS/6-MOS/9-MOS
    (quarterly, cumulative YTD, from a 10-Q) -- see docs/US_QUARTERLY_BACKFILL_PLAN.md
    Phase 2. Any other/missing PERIOD-TYPE is skipped, same as before.

    <FISCAL-YEAR-END> is only reliable as the exhibit's OWN period end when the
    exhibit covers the full year -- real bug, found scaling to ~250 companies
    (2026-07-28). Confirmed on ADP's real 1998-09-23 10-K: it bundles an
    Article-5 exhibit with PERIOD-TYPE=6-MOS but FISCAL-YEAR-END=DEC-31-1998 --
    the company's eventual full-year cutoff (likely a fiscal-year-transition
    stub filing), not the ~1998-06-30 the 6-month figures actually describe.
    Using it as-is produced a filing DATED BEFORE its own claimed period end (a
    fundamentals_available_date earlier than fds_period_end -- the exact class
    of bug this whole pipeline exists to prevent). `_fds_period_end` (below)
    is where this is actually enforced: <PERIOD-END> for a quarterly exhibit,
    never <FISCAL-YEAR-END> as a fallback.
    """
    article = (tags.get("ARTICLE") or "").strip()
    # <MULTIPLIER> is genuinely OPTIONAL per SEC's EX-27 schema -- confirmed on WMT's
    # real filings (2026-07-30): its 1995/1996 10-Ks tag <MULTIPLIER> 1,000,000
    # explicitly, but 1997-2000 omit the tag entirely (not malformed -- simply absent
    # from the exhibit), even though the raw figures are STILL reported at the same
    # implicit millions scale (1997's TOTAL-ASSETS=39,604 is Walmart's real ~$39.6B,
    # not $39,604). Silently defaulting an absent tag to 1.0 understated net_revenue/
    # total_assets/etc. by up to 10^6 for exactly the filings that omit it -- a
    # different failure mode than a malformed/zero tag (which legitimately means "no
    # scaling"). fds_multiplier_explicit lets build_cik_history tell the two apart and
    # borrow a sibling exhibit's real multiplier for this CIK when the tag is missing.
    multiplier_explicit = "MULTIPLIER" in tags
    multiplier = _to_number(tags.get("MULTIPLIER", "1"))
    multiplier = 1.0 if np.isnan(multiplier) or multiplier == 0 else multiplier
    out = {"fds_article": article, "fds_multiplier": multiplier,
           "fds_multiplier_explicit": multiplier_explicit}
    period_type = (tags.get("PERIOD-TYPE") or "").strip().upper()
    if article != "5" or period_type not in ({"YEAR"} | _QUARTERLY_PERIOD_TYPES.keys()):
        return out
    out["period_months"] = 12 if period_type == "YEAR" else _QUARTERLY_PERIOD_TYPES[period_type]

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


def _fds_period_end(tags: dict) -> pd.Timestamp:
    """This exhibit's own period end. <FISCAL-YEAR-END> only for a full-year
    (PERIOD-TYPE=YEAR) exhibit, where the two tags coincide -- for a quarterly
    exhibit <FISCAL-YEAR-END> is the eventual full-year cutoff, not this
    exhibit's own interim end (the ADP bug, see extract_line_items). <PERIOD-END>
    is mandatory for a quarterly exhibit, never a fallback to <FISCAL-YEAR-END>;
    a missing/unparseable one yields NaT, and build_cik_history already drops
    NaT-period rows rather than guess."""
    period_type = (tags.get("PERIOD-TYPE") or "").strip().upper()
    tag = "FISCAL-YEAR-END" if period_type == "YEAR" else "PERIOD-END"
    return _parse_fds_date(tags.get(tag))


def extract_and_compute(text: str) -> list[dict]:
    """One filing's text -> a list of results, ONE PER EX-27 exhibit (a filing can
    bundle several -- see parse_fds's docstring). Each result is line items +
    compute_ratios(unit_scale=1) (already full-dollar via <MULTIPLIER>) + its own
    `fds_period_end` (see _fds_period_end). Empty list if no EX-27 exhibit exists
    in this filing. Ratios computed here are on RAW (possibly cumulative-YTD)
    values -- build_cik_history recomputes them after ytd_to_discrete."""
    results = []
    for tags in parse_fds(text):
        items = extract_line_items(tags)
        period_end = _fds_period_end(tags)
        if items.get("fds_article") != "5":
            results.append({**items, "fds_period_end": period_end})
            continue
        results.append({**items, **compute_ratios(items, unit_scale=1), "fds_period_end": period_end})
    return results


EX27_ERA_END = "2002-12-31"  # one-year buffer past EX-27's 2001 elimination (plan §2.0)

# The dollar-valued fields a <MULTIPLIER> actually scales -- everything
# extract_line_items populates under Article 5 except the fds_*/ratio outputs.
_DOLLAR_FIELDS = [*ARTICLE_5_MAP.keys(), "equity"]


def _fill_missing_multipliers(df: pd.DataFrame) -> pd.DataFrame:
    """Borrow this CIK's own multiplier from a sibling exhibit for any row whose
    <MULTIPLIER> tag was absent (see extract_line_items's docstring on WMT's real
    1997-2000 filings). A company's own EX-27 scale convention is consistent across
    its own filings even when one year's exhibit happens to omit the declaring tag
    -- confirmed on WMT (1995/1996 explicit at 1,000,000; 1997-2000 all implicitly
    the same scale, just undeclared) and SWK (1994-98 explicit at 1,000; 1999
    undeclared, same scale). Rows with NO sibling exhibit anywhere in this CIK's own
    history to borrow from (confirmed on TXT: every single collected exhibit omits
    the tag) are left as-is -- fds_multiplier_explicit stays False so this remains
    visible/auditable rather than silently indistinguishable from a confirmed "no
    scaling" declaration.
    """
    explicit = df[df["fds_multiplier_explicit"]]
    if explicit.empty:
        return df
    canonical = explicit["fds_multiplier"].mode().iloc[0]
    missing = ~df["fds_multiplier_explicit"] & (df["fds_multiplier"] != canonical)
    if not missing.any():
        return df
    df = df.copy()
    factor = canonical / df.loc[missing, "fds_multiplier"]
    for col in _DOLLAR_FIELDS:
        if col in df.columns:
            df.loc[missing, col] = df.loc[missing, col] * factor
    df.loc[missing, "fds_multiplier"] = canonical
    corrected = df.loc[missing]
    ratios = corrected.apply(lambda r: compute_ratios(r.to_dict(), unit_scale=1), axis=1, result_type="expand")
    df.loc[missing, ratios.columns] = ratios
    return df


def build_cik_history(cik: int, filings: pd.DataFrame) -> pd.DataFrame:
    """Every qualifying (Article-5) EX-27 exhibit for one CIK across the Phase 3
    filings table -- including every comparative exhibit a single filing bundles,
    not just its first -- each stamped with `fundamentals_available_date` = the
    filing's real EDGAR date_filed -- NOT the fiscal period end, per the plan's
    §5.2 point-in-time rule (identical reasoning to companyfacts.py's `filed`).
    Where a period is reported by more than one filing (a later 10-K restating an
    earlier year as a comparative, or a 10-K/A amendment), the EARLIEST filing
    wins -- same as-first-reported rule as the XBRL tier (§3.3).

    Covers both 10-K (annual) and 10-Q (quarterly) filings -- EX-27 exhibits
    were required on both until the 2001 elimination. 10-Q exhibits report
    cumulative YTD flow figures, turned into discrete ~3-month figures by
    ytd_to_discrete before returning (docs/US_QUARTERLY_BACKFILL_PLAN.md).

    Filtered to filings up to EX27_ERA_END: an earlier version fetched EVERY
    10-K a CIK ever filed (including decades of post-2001 filings that
    structurally cannot contain an EX-27, per this tier's own prevalence
    measurement), making batch collection needlessly slow -- confirmed while
    scaling past a handful of companies, 2026-07-28.
    """
    cik_filings = filings[(filings["cik"] == cik)
                           & filings["form_type"].str.startswith(("10-K", "10-Q"))
                           & (filings["date_filed"] <= EX27_ERA_END)]
    rows = []
    for row in cik_filings.itertuples():
        text = fetch_filing_text(row.filename)
        if text is None:
            continue
        for result in extract_and_compute(text):
            # "total_assets" is only present when extract_line_items actually
            # populated the exhibit (article==5 AND an accepted PERIOD-TYPE) --
            # covers both the wrong-article and wrong-period-type cases (see
            # extract_line_items) with one check.
            if "total_assets" not in result:
                continue
            rows.append({**result, "cik": cik, "fundamentals_available_date": row.date_filed,
                         "fds_form": row.form_type, "fds_filename": row.filename})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = _fill_missing_multipliers(df)
    # A row with no resolvable period end (missing/malformed <PERIOD-END> or
    # <FISCAL-YEAR-END>, see _fds_period_end) can't be placed on the timeline
    # regardless of dedup -- and drop_duplicates below treats NaT == NaT, so
    # leaving these in would silently collapse two DIFFERENT real fiscal
    # periods' data into one bogus survivor (keeping only the earlier-filed
    # one, itself still useless with a NaT period end) instead of dropping both.
    missing_period = df["fds_period_end"].isna()
    if missing_period.any():
        log.warning("fds CIK %s: dropping %d exhibit(s) with an unparseable "
                    "period end (<PERIOD-END>/<FISCAL-YEAR-END>)",
                    cik, missing_period.sum())
        df = df[~missing_period]
    if df.empty:
        return pd.DataFrame()
    # as-first-reported: whichever filing disclosed a given fiscal period EARLIEST
    # wins, whether that's the period's own original filing or a later filing's
    # bundled comparative exhibit reporting it first for some other reason. Must
    # happen BEFORE ytd_to_discrete, which needs exactly one row per period.
    df = (df.sort_values("fundamentals_available_date")
            .drop_duplicates(subset="fds_period_end", keep="first")
            .sort_values("fds_period_end")
            .reset_index(drop=True))

    # 3/6/9-MOS exhibits report CUMULATIVE year-to-date figures, same convention
    # as a 10-Q's own statements -- turn them into discrete ~3-month figures.
    # Ratios above were computed per-exhibit on the RAW YTD values
    # (extract_and_compute); recompute on the now-discrete flows, never the
    # reverse order (see docs/US_QUARTERLY_BACKFILL_PLAN.md).
    df = df.rename(columns={"fds_period_end": "end"})
    df = companyfacts.ytd_to_discrete(df, flow_cols=["net_income", "net_revenue", "cost_of_revenue"])
    ratios = df.apply(lambda r: compute_ratios(r.to_dict(), unit_scale=1), axis=1, result_type="expand")
    df[ratios.columns] = ratios
    return df.rename(columns={"end": "fds_period_end"})


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
            # parse_fds returns a LIST (a filing can bundle multiple EX-27
            # exhibits, see its own docstring) -- an earlier version of this
            # treated it as a dict/None, so `tags is not None` was True even
            # for an empty list (has_ex27 always True), and `.get("ARTICLE")`
            # raised AttributeError on the first filing that genuinely had one.
            exhibits = parse_fds(text) if text else []
            rows.append({"year": year, "has_ex27": bool(exhibits),
                         "article": exhibits[0].get("ARTICLE") if exhibits else None})
    return pd.DataFrame(rows)
