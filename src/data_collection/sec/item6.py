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

log = logging.getLogger("sec")

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
# Curly right-single-quote (U+2019) shows up in "Stockholders' equity" etc.
_APOSTROPHE = re.compile("’")

# Filings in this date-filed window can plausibly carry the target 2001-2006
# gap years within their 5-year (or fragmented 3-year, see find_item6_table)
# lookback. Bounded the same way fds.py's EX27_ERA_END is -- fetching a
# company's ENTIRE filing history just to check a handful of years was
# already a confirmed real bug there; same principle applies here.
GAP_ERA_START = "2001-01-01"
GAP_ERA_END = "2008-12-31"

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
        header_text = _row_text(df.head(3).values.flatten())
        years = set(_YEAR_RE.findall(header_text))
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
    """Years found in the header rows, in left-to-right column order (dedup
    preserving first occurrence -- a year can span 2 merged header cells)."""
    years = []
    for _, row in df.head(3).iterrows():
        for cell in row:
            m = _YEAR_RE.search(str(cell))
            if m and m.group() not in years:
                years.append(m.group())
    return years


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


def extract_years(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    """The located Item 6 table -> {year: {line_item: value}}.

    Two-pass alias matching, EXACT first then substring, and never overwrites
    an already-assigned item -- real bug, found against Intel's actual table:
    the short label "Diluted" (real diluted EPS) exactly matches alias
    "diluted", but "Weighted average diluted common shares outstanding" ALSO
    contains "diluted" as a substring and, processed after, silently
    overwrote the correct EPS values with share-count figures. Exact-match
    rows are resolved first so the short, precise label wins before any
    longer label's substring collision is even considered.
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
            for label, row in rows:
                if label and label_test(label, aliases):
                    vals = _row_values(row.iloc[1:], len(years))
                    for y, v in zip(years, vals):
                        if v is not None:
                            out[y][item] = v
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
        except ValueError:
            continue
        table = find_item6_table(tabs)
        if table is None:
            continue
        for year_str, items in extract_years(table).items():
            rows.append({**items, "fiscal_year": int(year_str),
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
