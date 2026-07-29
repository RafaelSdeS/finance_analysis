"""sec/item6.py — Item 6 "Selected Financial Data" chaining (Phase 7, plan §3.4).

Closes the 2001-2006 gap between the EX-27 tier (usably 1995-2000) and the
XBRL tier (2007+). Until its 2021 elimination, the SEC required a standardized
5-year summary table in every 10-K. Chaining two filings per company spans any
window; consecutive filings' tables overlap by ~4 years, giving free
cross-validation of every extracted figure against 2+ independent filings.

Table location is the hard part, not parsing (verified 2026-07-28: Intel's
2010 10-K produced 342 pandas.read_html tables; the heuristic below picked
the right one on the first try, scored well above two false positives).
Value extraction is positional (Nth numeric token in a row <-> Nth year in
the header), not column-index-based -- real filings interleave "$"
placeholder cells and NaN spacer columns inconsistently, but the COUNT of
meaningful tokens per year-block is consistent within one table.
"""

import io
import logging
import re

import pandas as pd

from . import http
from ..yf_collectors import compute_ratios

log = logging.getLogger("sec")

_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
# Curly right-single-quote (U+2019) shows up in "Stockholders' equity" etc.
_APOSTROPHE = re.compile("’")

# Item 6 tables print dollar figures under a "(in millions)"/"(in thousands)"
# caption that lives in the filing's surrounding text, not inside the parsed
# table itself -- extract_years/find_item6_table never saw it, so every
# gap-tier dollar figure was stored at face value (e.g. Intel's real
# $35,127,000,000 1994-2009 net revenue stored as bare "35127"). Confirmed
# against adjacent tiers for the SAME company (2026-07-28): INTC item6
# net_revenue median 3.42e4 vs INTC xbrl 1.36e10; IBM 9.14e4 vs IBM ex27
# 7.39e10 -- a 10^5-10^6 magnitude cliff sandwiched between two correct
# tiers. Per-share figures (EPS, dividends/share) are NEVER expressed in the
# table's caption units even when every other row is -- confirmed on Intel's
# real table above: "0.79"/"0.77" EPS sit next to "35127"/"4369" net
# revenue/income under the same "(In Millions...)" caption.
_UNITS_RE = re.compile(r"in\s+(thousands|millions|billions)\b", re.I)
_UNIT_MULTIPLIER = {"thousands": 1e3, "millions": 1e6, "billions": 1e9}
_PER_SHARE_ITEMS = {"eps_basic", "eps_diluted", "dividends_per_share"}


def detect_unit_multiplier(text: str) -> float:
    """Scale factor implied by the filing's units caption ("(in millions)" etc.),
    1.0 if none found. A filing can mention units more than once (other tables,
    MD&A prose) -- takes the most frequent mention, not just the first, since
    Item 6's own caption is virtually always the dominant one in a 10-K."""
    hits = [m.group(1).lower() for m in _UNITS_RE.finditer(text)]
    if not hits:
        return 1.0
    mode = max(set(hits), key=hits.count)
    return _UNIT_MULTIPLIER[mode]

# Filings in this date-filed window can plausibly carry the target 2001-2006
# gap years within their 5-year (or fragmented 3-year, see find_item6_table)
# lookback. Bounded the same way fds.py's EX27_ERA_END is -- fetching a
# company's ENTIRE filing history just to check a handful of years was
# already a confirmed real bug there; same principle applies here.
GAP_ERA_START = "2001-01-01"
GAP_ERA_END = "2008-12-31"

# Last-line-of-defense bound on any single extracted fiscal_year, generous
# margin either side of a filing's plausible 5-year lookback within
# GAP_ERA. Found necessary the hard way (2026-07-29): a mis-selected table
# producing even ONE implausible fiscal_year (e.g. AMG's 2054, from an
# embedded-digit false match) doesn't just corrupt that one row --
# fundamentals.py's non-calendar-FYE correction derives a company-wide
# year_offset from whichever row looks "impossible" and applies it to EVERY
# row for that CIK, so one bad year silently shifted AMG's otherwise-correct
# 2002-2006 rows by -49 years too. The _year_header_row row-selection fix
# above addresses the specific mechanism found; this bound is a cheap,
# independent backstop against any other yet-unseen mis-parse doing the same.
_FISCAL_YEAR_MIN = 1990
_FISCAL_YEAR_MAX = 2010

ROW_ALIASES = {
    "net_revenue": ["net revenue", "net sales", "total revenue", "total revenues"],
    "net_income": ["net income", "net earnings"],
    "total_assets": ["total assets"],
    "total_debt": ["long-term debt", "long term debt"],
    "equity": ["stockholders' equity", "shareowners' equity", "shareholders' equity", "total equity"],
    "dividends_per_share": ["declared"],
    "eps_basic": ["basic"],
    "eps_diluted": ["diluted"],
}


def _row_text(series: pd.Series) -> str:
    return " ".join(str(x) for x in series.tolist())


def _normalize_label(s: str) -> str:
    return _APOSTROPHE.sub("'", str(s)).strip().lower()


def _year_header_row(df: pd.DataFrame) -> list[str]:
    """Years found in whichever of the first 3 rows has the most distinct
    year-like tokens, left to right (dedup preserving first occurrence -- a
    year can span 2 merged header cells). Rows containing "$" are skipped --
    real Item 6 tables always list years bare in one dedicated row, with "$"
    appearing only on data rows below.

    Scanning row-by-row (not flattening df.head(3) together) matters:
    confirmed on two real false-positive filings (2026-07-29) where combining
    rows let unrelated data-row figures get counted alongside a genuine year
    header, tipping a wrong table over the >=3-years threshold in
    find_item6_table -- AAPL's 2004 10-K misread its Selected Quarterly
    Financial Data table as Item 6 because three quarters' net sales
    ($2,014M / $1,909M / $2,006M) are themselves 4-digit year-shaped numbers;
    AMG's misread a stock-comp footnote table where large figures like
    "119069" and "22054.0" contain embedded year-shaped substrings ("1906",
    "2054") that the old unanchored _YEAR_RE also matched.
    """
    best: list[str] = []
    for _, row in df.head(3).iterrows():
        cells = [str(c) for c in row.tolist()]
        if any("$" in c for c in cells):
            continue
        years: list[str] = []
        for c in cells:
            m = _YEAR_RE.search(c)
            if m and m.group() not in years:
                years.append(m.group())
        if len(years) > len(best):
            best = years
    return best


def find_item6_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Best-scoring candidate among ALL of a filing's pandas.read_html tables.
    Score = (year_count>=3) AND >=1 of the core keywords in the first column;
    ties broken by keyword-hit count, then row count (favors the real,
    fuller table over small false-positive fragments).

    Threshold is 3 years, not the full 5 Item 6 legally requires: confirmed
    on Intel's real 2007-filed 10-K, whose 5-year table doesn't survive as
    ONE pandas.read_html table -- it fragments into multiple candidates that
    each carry only 3 of the 5 years (an HTML table-boundary artifact, not a
    data problem; values reconcile exactly to Intel's known 2004-2006
    figures). This picks up genuine MD&A-style 3-year comparisons too, not
    strictly the SEC's "Item 6" caption -- accepted, since accurate history
    matters more here than which item officially captioned it, and the
    Phase 7 chaining strategy recovers full coverage across MORE filings
    rather than needing every single filing's table to be complete.
    """
    best, best_score = None, (0, 0, 0)
    for df in tables:
        if df.shape[1] < 3:
            continue
        years = _year_header_row(df)
        if len(years) < 3:
            continue
        firstcol = _row_text(df.iloc[:, 0]).upper()
        score = sum(k.upper() in firstcol for k in
                    ("TOTAL ASSETS", "NET INCOME", "NET REVENUE", "NET SALES", "REVENUE"))
        if score < 1:
            continue
        key = (score, len(years), df.shape[0])
        if key > best_score:
            best, best_score = df, key
    return best


def _find_year_columns(df: pd.DataFrame) -> list[str]:
    """Years found in the header row, in left-to-right column order -- see
    _year_header_row for why this must be a single row, not flattened."""
    return _year_header_row(df)


def _parse_value(s) -> float | None:
    s = str(s).strip()
    if not s or s in ("$", "nan", "NaN"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "").replace("$", "").strip()
    if not s:
        return None
    try:
        val = float(s)
    except ValueError:
        return None
    return -val if neg else val


def _row_values(row: pd.Series, n_years: int) -> list[float | None]:
    """Up to n_years numeric tokens from a row, left to right -- positional,
    not column-index-based, since $ signs and NaN spacers interleave
    inconsistently across real filings."""
    vals = []
    for cell in row:
        if pd.isna(cell):
            continue
        v = _parse_value(cell)
        if v is not None:
            vals.append(v)
    vals = vals[:n_years]
    return vals + [None] * (n_years - len(vals))


def extract_years(table: pd.DataFrame, unit_multiplier: float = 1.0) -> dict[str, dict[str, float]]:
    """The located Item 6 table -> {year: {line_item: value}}.

    Two-pass alias matching, EXACT first then substring, and never overwrites
    an already-assigned item -- real bug, found against Intel's actual table:
    the short label "Diluted" (real diluted EPS) exactly matches alias
    "diluted", but "Weighted average diluted common shares outstanding" ALSO
    contains "diluted" as a substring and, processed after, silently
    overwrote the correct EPS values with share-count figures. Exact-match
    rows are resolved first so the short, precise label wins before any
    longer label's substring collision is even considered.

    `unit_multiplier` (see detect_unit_multiplier) rescales every dollar-figure
    row to full currency units -- every ROW_ALIASES item except _PER_SHARE_ITEMS,
    which the caption never applies to (see that set's docstring).
    """
    years = _find_year_columns(table)
    if not years:
        return {}
    out = {y: {} for y in years}
    rows = [(_normalize_label(row.iloc[0]), row) for _, row in table.iterrows()]

    def assign(label_test):
        for item, aliases in ROW_ALIASES.items():
            if any(item in out[y] for y in years):
                continue  # already assigned by a higher-priority pass
            scale = 1.0 if item in _PER_SHARE_ITEMS else unit_multiplier
            for label, row in rows:
                if label and label_test(label, aliases):
                    vals = _row_values(row.iloc[1:], len(years))
                    for y, v in zip(years, vals):
                        if v is not None:
                            out[y][item] = v * scale
                    break

    assign(lambda label, aliases: label in aliases)                       # exact match
    assign(lambda label, aliases: any(a in label for a in aliases))       # substring fallback
    return {y: v for y, v in out.items() if v}


def build_cik_history(cik: int, filings: pd.DataFrame) -> pd.DataFrame:
    """Chain every 10-K in the gap era for one CIK, extracting each filing's
    Item 6 (or Item-6-like MD&A comparison) table and keeping the EARLIEST
    filing per fiscal year -- as-first-reported, same rule as the EX-27 and
    XBRL tiers (plan §3.3). Overlapping years across consecutive filings are
    NOT deduplicated away silently -- cross-validate them before trusting a
    company's chained history (verified manually on Intel: 2005/2006 net
    revenue, net income, and both EPS figures agreed exactly across two
    independent filings 3 years apart).
    """
    cik_filings = filings[(filings["cik"] == cik) & (filings["form_type"].str.startswith("10-K"))
                           & (filings["date_filed"] >= GAP_ERA_START) & (filings["date_filed"] <= GAP_ERA_END)]
    rows = []
    for row in cik_filings.itertuples():
        resp = http.get(f"https://www.sec.gov/Archives/{row.filename}")
        if resp is None:
            continue
        try:
            tabs = pd.read_html(io.StringIO(resp.text))
        except Exception:
            # pd.read_html raises more than ValueError on malformed real-world
            # HTML -- confirmed on YUM's real filing history (2026-07-28): one
            # filing's table structure crashed pandas' internal TextParser
            # with an IndexError, not a ValueError. Uncaught, this took down
            # the CIK's ENTIRE fundamentals build (item6 has no per-CIK try/
            # except of its own in fundamentals.py), discarding YUM's
            # perfectly good xbrl-tier data (611 us-gaap concepts) along with
            # it. The whole point of this loop is "skip a filing that doesn't
            # parse cleanly, try the next one" -- any parse failure qualifies.
            continue
        table = find_item6_table(tabs)
        if table is None:
            continue
        unit_multiplier = detect_unit_multiplier(resp.text)
        for year_str, items in extract_years(table, unit_multiplier).items():
            year = int(year_str)
            if not (_FISCAL_YEAR_MIN <= year <= _FISCAL_YEAR_MAX):
                continue
            ratios = compute_ratios(items, unit_scale=1)
            rows.append({**items, **ratios, "fiscal_year": year,
                         "fundamentals_available_date": row.date_filed,
                         "item6_form": row.form_type, "item6_filename": row.filename})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["cik"] = cik
    return (df.sort_values("fundamentals_available_date")
              .drop_duplicates(subset="fiscal_year", keep="first")
              .sort_values("fiscal_year")
              .reset_index(drop=True))
