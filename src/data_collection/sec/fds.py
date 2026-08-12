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
    # Parenthesized negative -- real bug, confirmed on TCX's actual
    # 1998-09-30 10-Q (2026-08-01): <OTHER-SE>(2,424,212) (standard accounting
    # notation for a real negative -- the company's genuine financial
    # distress), silently unparseable here while the analogous case is
    # already handled in selected_financial_data.py's _parse_value. Returning
    # NaN for a real negative, rather than the negative itself, is bad enough
    # on its own (lost data), but compounds badly in extract_line_items's
    # equity = nan_to_num(COMMON) + nan_to_num(OTHER-SE): a missing COMPONENT
    # masquerades as a genuine value of 0 instead of propagating NaN, turning
    # a real -$2.4M equity into a false exact $0.00.
    s = (s or "").replace(",", "").strip()
    if not s or s.startswith("<"):
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        val = float(s)
    except ValueError:
        return np.nan
    return -val if neg else val


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


def _rescale_dollar_fields_and_ratios(df: pd.DataFrame, idx, factor) -> None:
    """Shared tail of both multiplier-fix passes below: rescale _DOLLAR_FIELDS
    by `factor` for rows `idx`, then recompute ratios on the now-corrected
    values. Mutates `df` in place."""
    for col in _DOLLAR_FIELDS:
        if col in df.columns:
            df.loc[idx, col] = df.loc[idx, col] * factor
    corrected = df.loc[idx]
    ratios = corrected.apply(lambda r: compute_ratios(r.to_dict(), unit_scale=1), axis=1, result_type="expand")
    df.loc[idx, ratios.columns] = ratios


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

    The "one company, one scale" premise above is NOT always true, though --
    confirmed via cross-vendor validation (tests/data_collection/validate_us_vs_vendor.py,
    check_tier_seams) on GAP (CIK 39911) and GIS (CIK 40704): only 1-2 of ~20
    exhibits ever declare <MULTIPLIER> explicitly, and blindly force-applying
    that single borrowed value to every other exhibit produced rows 1000x/1e6x
    off from their OWN neighbors on both sides -- some undeclared exhibits'
    raw tag values were evidently already reported near full-dollar scale,
    others genuinely needed the borrowed factor, interleaved within the same
    CIK's history. A borrowed rescale is now only accepted when it lands the
    exhibit's total_assets closer, in log-scale, to this CIK's own explicit-
    tier median than leaving it unscaled would -- otherwise the exhibit is
    left unscaled and unmarked explicit, same conservative fallback as a CIK
    with no sibling to borrow from at all (below). Uses the CIK-wide median
    rather than the nearest single sibling in time -- simpler, and one
    mis-scaled explicit exhibit skews a median far less than it would a
    nearest-neighbor pick.

    `fds_multiplier_resolved` marks every row this function (or a real
    explicit tag) has confidently settled -- real bug, confirmed 2026-08-12:
    infer_multiplier_from_trusted_tiers downstream used to treat
    `fds_multiplier == 1.0` as its own proxy for "still unresolved", but a
    row can already sit at this CIK's own confirmed canonical scale without
    ever being individually rescaled here (its raw value simply happened to
    already match canonical, e.g. a company whose explicit filings genuinely
    declare no scaling, canonical == 1.0, the same value every untouched
    exhibit starts at by default) -- indistinguishable, under a bare value
    check, from a row that was never touched at all. Any row landing on
    canonical -- whether by an explicit tag, by already matching it, or by
    this function's own accept-and-rescale below -- is marked resolved so it
    can't be silently re-anchored to an unrelated cross-tier reference by
    that second pass; a row that DOESN'T match canonical (missing but
    rejected as implausible, see GAP/GIS above) correctly stays unresolved
    and eligible for that second chance.

    reference (the plausibility anchor) requires total_assets on the
    explicit rows, but canonical (the scale itself) must not -- real bug,
    confirmed 2026-08-12: requiring total_assets.notna() on BOTH shrank the
    "explicit" set down to whichever minority of genuinely-explicit rows
    happen to also have a clean TOTAL-ASSETS tag, so a malformed tag on most
    of them silently derived canonical from an unrepresentative remainder.
    """
    df = df.copy()
    df["fds_multiplier_resolved"] = df["fds_multiplier_explicit"]
    if "total_assets" not in df.columns:
        return df  # nothing to judge plausibility against -- don't guess
    explicit = df[df["fds_multiplier_explicit"]]
    if explicit.empty:
        return df
    canonical = explicit["fds_multiplier"].mode().iloc[0]
    df["fds_multiplier_resolved"] |= (df["fds_multiplier"] == canonical)
    reference_rows = explicit[explicit["total_assets"].notna()]
    if reference_rows.empty:
        return df
    reference = reference_rows["total_assets"].abs().median()
    missing = ~df["fds_multiplier_explicit"] & (df["fds_multiplier"] != canonical)
    if not missing.any():
        return df
    sub = df.index[missing]
    factor = canonical / df.loc[sub, "fds_multiplier"]
    scaled_assets = (df.loc[sub, "total_assets"] * factor).abs()
    current_assets = df.loc[sub, "total_assets"].abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled_dist = np.log10(scaled_assets / reference).abs()
        current_dist = np.log10(current_assets / reference).abs()
    accept = scaled_dist.index[scaled_dist < current_dist]  # NaN comparisons -> False, correctly excluded
    if len(accept) == 0:
        return df
    df.loc[accept, "fds_multiplier"] = canonical
    df.loc[accept, "fds_multiplier_resolved"] = True
    _rescale_dollar_fields_and_ratios(df, accept, factor.loc[accept])
    return df


# EX-27's own valid multiplier conventions -- the exact set seen in this
# tier's real filings (WMT explicit at 1,000,000, SWK explicit at 1,000, see
# extract_line_items's docstring; 1.0 covers a filing that genuinely needs no
# scaling). Not an arbitrary guess space -- restricting to values the format
# actually uses keeps this from accepting a coincidentally-close but wrong
# power of 10.
_VALID_MULTIPLIERS = (1.0, 1_000.0, 1_000_000.0)

# A trusted reference this large is itself implausible for the small/mid-cap
# universe that actually needs this cross-tier path (real EX-27 filers with
# NO explicit multiplier anywhere -- if a company were XBRL-era-huge, it
# would almost certainly have an ex27 sibling WITH an explicit tag, resolved
# by _fill_missing_multipliers already). A per-row acceptance test alone
# isn't enough of a guard: real bug, confirmed on AUSI (Aura Systems, CIK
# 826253, 2026-08-06), whose trusted item6 reference itself reads $56
# TRILLION total_assets (a still-unresolved item6 bug, unrelated to this
# function) -- an ex27
# candidate scaled by 1,000,000 landed "within 3x" of that reference purely
# because BOTH sides were wrong by roughly the same magnitude, not because
# either was actually correct. Rejecting the reference itself before it's
# ever used as an anchor is a cheap, independent second guard against
# exactly this coincidental-agreement failure mode.
_TRUSTED_REFERENCE_CEILING = 2e11  # $200B; see docstring for why this bound


def infer_multiplier_from_trusted_tiers(df: pd.DataFrame, trusted: pd.DataFrame,
                                         max_gap_days: int = 730) -> pd.DataFrame:
    """Second chance for ex27 rows _fill_missing_multipliers couldn't resolve at
    all -- no explicit <MULTIPLIER> ANYWHERE in this CIK's own ex27 history, so
    there is no same-tier sibling to borrow from in the first place. Confirmed
    real (2026-08-06, via tests/data_collection/validate_us_vs_vendor.py's
    tier-seam check) on AEO, ATNI, AUSI, FHI, and 13 others: every single
    collected ex27 exhibit stays at the untouched default multiplier=1.0
    forever under _fill_missing_multipliers alone, wrong for every one of
    them -- confirmed on AEO by cross-referencing its own item6 tier's already-
    correct FY1998 net_sales ($405,713K annual): AEO's raw ex27 Q3 1997
    TOTAL-REVENUES tag (104902) is genuinely thousands-scale ($104,902,000,
    ~26% of the annual figure, a plausible quarter), silently understated
    1000x by the unresolved multiplier=1.0 default.

    `trusted` is this CIK's OTHER tiers (item6/tenq/xbrl) combined -- each
    individually more reliable than an ex27 exhibit with zero same-tier
    signal to work with. For each unresolved row, finds the temporally
    nearest trusted total_assets (within `max_gap_days`, via the same
    pd.merge_asof(direction='nearest', tolerance=...) pattern already used
    for this exact problem shape in companyfacts.extract_line_items -- an
    earlier version of this reinvented it as a per-row Python loop) and
    tries every valid EX-27 multiplier, keeping whichever lands closest
    (log-scale) to that reference -- same "does scaling actually help"
    acceptance test as _fill_missing_multipliers above, just against a
    cross-tier reference instead of a same-tier canonical (there IS no
    same-tier canonical for these rows). Requires the match land within 3x,
    not just be the least-bad candidate -- a real guard against a stale/
    unrelated trusted row (including a still-imperfect item6 row -- see that
    tier's own known residual error rate) misleading this into a confident
    wrong answer: an ex27 candidate would need to coincidentally land within
    3x of a WRONG reference to be accepted, a narrow coincidence for the
    whole-decade (10x/1000x) errors this is built to catch.

    Never touches an already-resolved row -- tracked via
    `fds_multiplier_resolved` (see _fill_missing_multipliers), NOT a check
    of whether `fds_multiplier == 1.0`: real bug, confirmed 2026-08-12, a row
    _fill_missing_multipliers already correctly settled onto a canonical
    scale of 1.0 (a company whose explicit filings genuinely declare no
    scaling) is indistinguishable, under the old value-based check, from a
    row that was simply never touched -- silently re-anchoring an
    already-correct row to an unrelated cross-tier reference instead of
    leaving it alone.
    """
    if "total_assets" not in df.columns or trusted.empty or "total_assets" not in trusted.columns:
        return df
    resolved = df["fds_multiplier_resolved"] if "fds_multiplier_resolved" in df.columns \
        else df["fds_multiplier_explicit"]
    unresolved = ~resolved & df["total_assets"].notna()
    if not unresolved.any():
        return df
    ref = trusted[["end", "total_assets"]].dropna()
    ref = ref[(ref["total_assets"] != 0) & (ref["total_assets"].abs() < _TRUSTED_REFERENCE_CEILING)]
    if ref.empty:
        return df
    ref = ref.sort_values("end")

    df = df.copy()
    left = (df.loc[unresolved, ["end", "total_assets"]]
              .rename(columns={"total_assets": "raw_assets"})
              .reset_index().rename(columns={"index": "row_id"})
              .sort_values("end"))
    matched = pd.merge_asof(left, ref, on="end", direction="nearest",
                             tolerance=pd.Timedelta(days=max_gap_days))
    matched = matched[matched["total_assets"].notna() & matched["raw_assets"].notna()
                       & (matched["raw_assets"] != 0)]

    accepted = []
    for row in matched.itertuples(index=False):
        raw, target = row.raw_assets, abs(row.total_assets)
        with np.errstate(divide="ignore", invalid="ignore"):
            best = min(_VALID_MULTIPLIERS, key=lambda f: abs(np.log10(abs(raw * f) / target)))
            best_dist = abs(np.log10(abs(raw * best) / target))
        if not (best_dist < np.log10(3)):
            continue
        df.at[row.row_id, "fds_multiplier"] = best
        accepted.append(row.row_id)

    if not accepted:
        return df
    df.loc[accepted, "fds_multiplier_resolved"] = True
    _rescale_dollar_fields_and_ratios(df, accepted, df.loc[accepted, "fds_multiplier"])
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
