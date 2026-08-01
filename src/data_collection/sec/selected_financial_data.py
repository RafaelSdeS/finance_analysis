"""sec/selected_financial_data.py — the "Item 6" gap tier (Phase 7, plan §3.4).

What "Item 6" is: every 10-K annual report is split into numbered Items per
SEC Regulation S-K. Item 6 was captioned "Selected Financial Data" -- a
standardized 5-year summary table (net revenue, net income, EPS, total
assets, ...) that Reg S-K Item 301 required in every 10-K until the SEC
eliminated it in 2021 as redundant with the MD&A section. This module scrapes
that table out of old filings; "item6"/"Item 6" still shows up throughout
this codebase (column names, tier labels, docs) as the standard short name
for exactly this data source -- this file just has the more legible one.

Why it exists: it closes the 2001-2006 gap between the EX-27 tier (usably
1995-2000) and the XBRL tier (2007+), the two other fundamentals sources this
pipeline has. Chaining two filings per company spans any window; consecutive
filings' tables overlap by ~4 years, giving free cross-validation of every
extracted figure against 2+ independent filings.

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
# table itself -- extract_years/find_selected_financial_data_table never saw it, so every
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

# A bare "(3)"-style cell referencing a table footnote -- present in some
# year-columns of a row but not others (confirmed on ORCL's real 2006 10-K:
# "Total assets" carries a "(3)" marker cell for 2006 and 2005 only, not
# 2004-2002). _parse_value can't tell it apart from a genuine small negative
# dollar figure by shape alone. See _row_values for why this only strips
# marker-shaped tokens when the row's raw token count exceeds n_years.
_FOOTNOTE_RE = re.compile(r"^\(\d{1,2}\)$")


def detect_unit_multiplier(text: str, prefer_first: bool = False) -> float:
    """Scale factor implied by the filing's units caption ("(in millions)" etc.),
    1.0 if none found. A filing can mention units more than once (other tables,
    MD&A prose) -- takes the most frequent mention, not just the first, since
    Item 6's own caption is virtually always the dominant one in a 10-K.

    `prefer_first=True` (used when `text` is a single winning table's own
    cells, not the whole filing) instead takes the FIRST mention outright --
    real bug, confirmed on AAPL's actual 2005 10-K (2026-07-30): its Item 6
    table states its governing caption once up front, "(In millions, except
    share and per share amounts)", then separately captions its
    shares-outstanding sub-row "(in thousands)" -- a 1-vs-1 tie under the old
    mode-based `max(set(hits), key=hits.count)` selection, whose outcome
    depends on `set()`'s hash-seed-dependent iteration order (not even
    deterministic across runs). Confirmed live: AAPL's FY2001-2005 net_revenue
    stored 1000x too small (e.g. FY2001 $5,363,000 instead of the real
    $5,363,000,000 -- Home Depot's FY1994-1999 show the identical exact-1000x
    pattern). The table's own governing caption is always stated in its
    header, before any per-row exception caption further down -- first
    mention is the correct, deterministic tiebreak for a single table's text;
    the whole-filing fallback below keeps the frequency heuristic, where
    disambiguating Item 6's caption among many unrelated mentions elsewhere
    in the document is a genuinely fuzzier problem.
    """
    hits = [m.group(1).lower() for m in _UNITS_RE.finditer(text)]
    if not hits:
        return 1.0
    if prefer_first:
        return _UNIT_MULTIPLIER[hits[0]]
    mode = max(set(hits), key=hits.count)
    return _UNIT_MULTIPLIER[mode]

# Filings in this date-filed window can plausibly carry the target 2001-2006
# gap years within their 5-year (or fragmented 3-year, see find_selected_financial_data_table)
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
    find_selected_financial_data_table -- AAPL's 2004 10-K misread its Selected Quarterly
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


def find_selected_financial_data_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
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

    Ranks by YEAR COUNT first, keyword score second -- real bug, confirmed
    on ZION's actual 2005 10-K (2026-07-30): Item 6 there is incorporated by
    reference to an exhibit (no real table in the parsed document at all), so
    the old (score, years, rows) ordering let a business SEGMENT's condensed
    income statement -- a 3-year MD&A fragment that happens to spell out
    "Total assets"/"Net income (loss)"/"Total revenue" verbatim (score 3) --
    beat the actual company-wide 5-year table, which labels its equivalent
    row just "Assets" under an "AT YEAR-END" header (score 2, since "Assets"
    alone doesn't match "TOTAL ASSETS"). A genuine Item 6 table's defining
    trait is covering more of the requested history, not how many keywords
    its labels happen to spell out -- year count now decides first.
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
        key = (len(years), score, df.shape[0])
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
    inconsistently across real filings.

    Real bug, confirmed on ORCL's actual 2006 10-K (2026-07-30): a footnote
    reference marker like "(3)" sits in its own cell, parses as a valid
    negative number under _parse_value, and inflates this row's token count
    beyond n_years -- which doesn't just produce one wrong value, it shifts
    EVERY later year's real figure one position early (ORCL's "Total assets"
    read 2006 correctly, then read 2005's marker cell as the 2005 value,
    corrupting 2005 through 2002 too; the impossible -3,000,000 total_assets
    was only the visible symptom). A genuine row's token count always equals
    n_years, so marker-shaped tokens are only ever stripped when the raw
    count is in EXCESS of n_years -- a real small negative dollar figure in
    an already-aligned row must never be discarded.

    Colspan-duplicated cells are collapsed BEFORE the footnote-marker check --
    real bug, confirmed on BOOM's (Dynamic Materials) actual 2005 10-K
    (2026-07-30): its "Total assets" row's HTML round-trips through
    pandas.read_html with every year's value duplicated into two adjacent
    columns (a colspan-to-columns rendering artifact specific to that row),
    stacked with one genuine footnote-marker cell. Stripping only the marker
    left 10 tokens for n_years=5 -- not an exact match -- so the check above
    silently gave up and fell through to the first 5 RAW (still-duplicated)
    tokens, corrupting every year one position off. Collapsing adjacent
    equal-value pairs first (only once the row is already known to have too
    many tokens, same conservative trigger as the footnote check) recovers
    the true per-year values before the footnote marker is even considered.
    """
    # A parenthesized negative split across two adjacent HTML cells -- real
    # bug, confirmed on AAPL's actual 2005 10-K (2026-07-30): the "Net income
    # (loss)" row's FY2001 column literally renders as two separate cells,
    # "(25" and ")", not one "(25)" cell. _parse_value only recognizes a
    # negative when BOTH parens are in the SAME string, so "(25" alone (no
    # trailing ")") fell through as a positive 25 -- Apple's real $25M
    # FY2001 net LOSS was stored as a $25M profit. Merge any cell that opens
    # a paren without closing it into its very next non-blank cell first.
    raw_cells = [str(c).strip() for c in row if not pd.isna(c)]
    merged_cells: list[str] = []
    i = 0
    while i < len(raw_cells):
        cell = raw_cells[i]
        if (cell.startswith("(") and not cell.endswith(")")
                and i + 1 < len(raw_cells) and raw_cells[i + 1].startswith(")")):
            merged_cells.append(cell + raw_cells[i + 1])
            i += 2
        else:
            merged_cells.append(cell)
            i += 1

    tokens: list[tuple[str, float]] = []
    for cell in merged_cells:
        v = _parse_value(cell)
        if v is not None:
            tokens.append((cell, v))
    if len(tokens) > n_years:
        deduped: list[tuple[str, float]] = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i][1] == tokens[i + 1][1]:
                deduped.append(tokens[i])
                i += 2
            else:
                deduped.append(tokens[i])
                i += 1
        if len(deduped) < len(tokens):
            tokens = deduped
    if len(tokens) > n_years:
        kept = [(s, v) for s, v in tokens if not _FOOTNOTE_RE.match(s)]
        if len(kept) == n_years:
            tokens = kept
    vals = [v for _, v in tokens][:n_years]
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
        table = find_selected_financial_data_table(tabs)
        if table is None:
            continue
        # Prefer the WINNING table's own units caption over the whole
        # document's -- real bug, confirmed on ZION's actual 2005 10-K
        # (2026-07-30): the winning table's own caption said "(Amounts in
        # millions)", but scanning the whole filing picked "thousands" instead
        # (the dominant caption of the much larger main financial statements
        # elsewhere in the same combined submission), silently rescaling this
        # table's figures 1000x too small. Falls back to the whole document
        # only when the winning table doesn't state its own units at all (a
        # caption living in a preceding paragraph, outside the parsed table).
        #
        # A share-count row's OWN local caption (e.g. "...shares outstanding,
        # in thousands") must not be read as the table's governing caption --
        # real bug, confirmed on TXN's actual 2006 10-K (2026-07-30): that
        # table states no dollar-figure caption in its own cells at all (its
        # real "(in millions)" caption lives in a preceding paragraph outside
        # the parsed table), so its ONLY units mention is the shares row's
        # local "in thousands" -- which the code below then wrongly applied
        # to net_revenue/net_income too, understating both 1000x. Excluding
        # any row whose label mentions "shares" from caption detection (never
        # from value extraction -- that's extract_years' own per_share_items
        # exemption, untouched here) empties table_text's unit mentions for a
        # table like this, correctly falling through to the whole-document
        # scan instead.
        caption_rows = table[~table.apply(lambda r: "shares" in _row_text(r).lower(), axis=1)]
        table_text = " ".join(str(c) for c in caption_rows.to_numpy().flatten())
        unit_multiplier = (detect_unit_multiplier(table_text, prefer_first=True) if _UNITS_RE.search(table_text)
                            else detect_unit_multiplier(resp.text))
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
    # item6 is annual by construction (Item 6 "Selected Financial Data" is a
    # 10-K-only disclosure, no quarterly equivalent) -- constant, never derived.
    # See docs/US_QUARTERLY_BACKFILL_PLAN.md for the period_months/flows_*
    # convention shared across all 4 fundamentals tiers.
    df["period_months"] = pd.array([12] * len(df), dtype="Int8")
    df["flows_derived"] = pd.Series(0, index=df.index, dtype="int8")
    df["flows_defined"] = pd.Series(1, index=df.index, dtype="int8")
    return (df.sort_values("fundamentals_available_date")
              .drop_duplicates(subset="fiscal_year", keep="first")
              .sort_values("fiscal_year")
              .reset_index(drop=True))
