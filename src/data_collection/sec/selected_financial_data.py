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

from . import cover_page, http
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
# "in millions" is not the only real caption phrasing -- real bug, confirmed
# on XOM's actual 2002 10-K (2026-08-06): its table's own caption literally
# reads "(millions of dollars, except per share amounts)" -- no "in" at all,
# so the old "in\s+(...)"-only pattern found nothing anywhere (table, Item 6
# heading, AND whole document), silently falling through to a wrong
# whole-document guess despite the real caption sitting right in the table's
# own cells the whole time (XOM's real ~$144.5B 1999 total assets stored as
# $144.5M). Matches either "in millions" or "millions of dollars" -- the two
# phrasings actually seen across real filings so far.
_UNITS_RE = re.compile(r"in\s+(thousands|millions|billions)\b|\b(thousands|millions|billions)\s+of\s+dollars\b", re.I)
_UNIT_MULTIPLIER = {"thousands": 1e3, "millions": 1e6, "billions": 1e9}
_PER_SHARE_ITEMS = {"eps_basic", "eps_diluted", "dividends_per_share"}

# A bare "(3)"-style cell referencing a table footnote -- present in some
# year-columns of a row but not others (confirmed on ORCL's real 2006 10-K:
# "Total assets" carries a "(3)" marker cell for 2006 and 2005 only, not
# 2004-2002). _parse_value can't tell it apart from a genuine small negative
# dollar figure by shape alone. See _row_values for why this only strips
# marker-shaped tokens when the row's raw token count exceeds n_years.
_FOOTNOTE_RE = re.compile(r"^\(\d{1,2}\)$")

# A pure alignment artifact ($ sign, blank, pandas' NaN-as-string) -- never a
# real year-column slot, safe to drop outright.
def _is_spacer(cell: str) -> bool:
    s = cell.strip()
    if not s or s in ("$", "nan", "NaN"):
        return True
    return not s.strip("()").replace(",", "").replace("$", "").strip()


# A genuine "not reported this year" marker -- dash/N-A-shaped -- as opposed
# to arbitrary non-numeric TEXT that isn't a real data slot at all. Narrow
# allowlist, not "anything that isn't a spacer": real bug, confirmed on AME's
# actual 2003 10-K (2026-08-06), found immediately after first trying the
# broader "any non-spacer cell is a real slot" rule -- its "Net sales" row
# label repeats a SECOND time as the row's very next cell (an HTML colspan
# artifact), and under the broad rule that duplicate label text got kept as
# a placeholder slot, shifting every real year's value one position off
# (2003 -> None, 2002 -> the real 2003 value, ... 1999's real value dropped
# off the end entirely). A dash/N-A cell is unambiguous; free-text isn't --
# so only the former counts as a real slot. See _row_values.
#
# Dash run is unbounded (`-+`/`–+`/`—+`), not capped at 3 -- real bug,
# confirmed 2026-08-12: a 4+-dash convention (some filers use "----") fell
# through the old `{1,3}` cap, parsed as neither a number nor a placeholder,
# and got silently dropped -- the exact under-count/position-shift bug this
# regex exists to prevent, just at a dash count nobody had tested yet.
_PLACEHOLDER_RE = re.compile(r"^(-+|–+|—+|n\.?/?a\.?|nm|n\.m\.)$", re.I)


def _is_placeholder(cell: str) -> bool:
    return bool(_PLACEHOLDER_RE.match(cell.strip().strip("()")))


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
    hits = [(m.group(1) or m.group(2)).lower() for m in _UNITS_RE.finditer(text)]
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


def _is_caption_only_row(row: pd.Series, years: tuple[str, ...] | list[str] = ()) -> bool:
    """A genuine table-WIDE caption row carries no real FINANCIAL data of its
    own -- either the caption occupies the label cell itself with every other
    cell blank (ZION's real shape -- which, in a compact table, can double as
    the year-header row itself, e.g. column 2 = "2004", hence the year-shape
    exemption below), or the label is empty and the caption text merely
    repeats across otherwise non-numeric cells (AME's real shape). A row that
    has ACTUAL dollar/share data alongside a units-sounding label (TXN's
    share count, FHI's "MANAGED AND ADMINISTERED ASSETS...(in millions)" row)
    is reporting a real line item with its own LOCAL caption, not the
    table's governing one -- see build_cik_history's caption detection for
    the real filings this distinguishes.

    A bare number is exempted from disqualifying "real data" only when it
    equals one of `years` -- THIS TABLE'S OWN real detected year columns (see
    _find_year_columns), not any 1900-2099-shaped integer. Real bug,
    confirmed 2026-08-12: a blanket 1900-2099 range exemption misclassified a
    genuine "shares outstanding" row as a table-wide caption row whenever its
    own share-count values happened to be integers in that range (a small/
    stable-share-count company) -- reintroducing the exact TXN-style bug this
    module's docstring says was already fixed once, just via a different
    coincidence. Anchoring to the table's own actual years still exempts the
    genuine merged caption+year-header case (ZION's shape) since that row's
    numbers ARE those years, while a coincidentally year-shaped but otherwise
    unrelated data value no longer passes. `years` defaults to empty (no
    exemption at all) for callers with no year-header concept of their own
    (see tenq.py's reuse of this same function for its differently-shaped
    tables)."""
    year_ints = {int(y) for y in years}
    for c in row.iloc[1:]:
        v = _parse_value(str(c))
        if v is not None and not (v.is_integer() and int(v) in year_ints):
            return False
    return True


def _normalize_label(s: str) -> str:
    # Whitespace-collapsed -- real bug, confirmed on AAPL's actual
    # 2004-02-10 10-Q (2026-08-01, tenq.py Phase 3): "Cost of  sales" renders
    # with an embedded double space (an HTML-entity artifact), which failed
    # every alias match against the clean single-space "cost of sales" alias
    # -- a real income-statement line item silently unmapped. Purely
    # normalizing (never removes distinguishing content), safe for every
    # existing caller/test.
    return re.sub(r"\s+", " ", _APOSTROPHE.sub("'", str(s))).strip().lower()


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


# A real "Item 6. Selected Financial Data" section heading -- allows arbitrary
# HTML noise both between "Item" and "6" AND between "6" and "Selected
# Financial Data" (real filings interleave <A NAME=...></A> anchors and
# &nbsp; runs in both gaps), but requires the actual caption phrase nearby,
# not just any "Item 6" mention (a table of contents entry, or "Item 601 of
# Regulation S-K", would false-match a bare \bitem\s*6\b alone).
#
# The first gap must NOT be restricted to \s* (real whitespace only) -- real
# bug, confirmed on KMB's actual 10-K (2026-08-06): its TABLE OF CONTENTS
# entry writes "Item 6." with a genuine space (matched fine), but the REAL
# body heading writes "ITEM&nbsp;6." -- the literal 6-character HTML entity
# "&nbsp;", not whitespace at all under \s. \s*-only matched the TOC line
# instead of the real heading, landing this function on a page-number blob
# with no caption anywhere near it, silently falling through to the
# whole-document scan (KMB's real ~$14.3B net sales stored as $14.3M).
_ITEM6_HEADING_RE = re.compile(r"item.{0,20}?6\b.{0,200}?selected\s+financial\s+data", re.I | re.S)


def _item6_heading_text(text: str, window: int = 2000) -> str:
    """Text between a real Item 6 heading and the next <TABLE> tag (or `window`
    chars, whichever comes first) -- the units caption paragraph almost always
    sits in exactly this gap, physically between the heading and the table
    itself. Empty string if no heading found.

    Real bug, confirmed on ATR's actual 2002 10-K (2026-08-06): "ITEM 6.
    SELECTED FINANCIAL DATA ... In millions of dollars" sits right before the
    table, outside the table's own cells (build_cik_history's table_text check
    correctly falls through), but the OLD whole-document fallback
    (detect_unit_multiplier(resp.text)) picked "thousands" instead -- this
    combined submission bundles several OTHER financial statements captioned
    in thousands, outnumbering Item 6's own single "millions" mention 9-to-1
    under the mode-based whole-document scan. A much narrower target than the
    whole filing: anchored on the one heading Reg S-K actually requires,
    checked BEFORE falling all the way back to the frequency-mode scan.

    Prefers whichever match's own window states a units caption, over just
    the FIRST match -- real bug, confirmed on KMB's actual 10-K (2026-08-06):
    a 10-K's table of contents virtually always ALSO spells out "Item 6.
    Selected Financial Data" (with a page number, no caption, immediately
    after) before the real section heading appears later in the body. Taking
    only the first match landed on the TOC line every time, never the real
    heading below it. Falls back to the first match's window when NONE of
    them state a caption (the heading is still a useful, narrower anchor than
    the whole document even without one -- callers still gate on _UNITS_RE
    before trusting it).
    """
    matches = list(_ITEM6_HEADING_RE.finditer(text))
    if not matches:
        return ""
    windows = []
    for m in matches:
        tail = text[m.end():m.end() + window]
        table_start = re.search(r"<table", tail, re.I)
        windows.append(tail[:table_start.start()] if table_start else tail)
    return next((w for w in windows if _UNITS_RE.search(w)), windows[0])


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

    A genuine "-"/"N/A"-style placeholder cell (a real but unparseable data
    slot, distinct from a "$"/blank alignment spacer -- see _is_spacer) MUST
    still occupy its position as a None, not be dropped outright -- the
    over-count guards above only handle a row with too MANY tokens; nothing
    previously handled too FEW. Dropping a genuine placeholder shrinks the
    token count below n_years, and the unconditional `+ [None] * ...` padding
    at the tail then appends the missing slot(s) at the END regardless of
    where the gap actually was -- silently shifting every later year's real
    value one position early, the same corruption class as the already-fixed
    footnote-marker and colspan bugs above, just from under- instead of
    over-counting. Hypothesized from validate_us_vs_vendor.py's tier-seam
    check (GAP/AUSI net_revenue reading implausibly small across a short
    contiguous run of years, both cross-vendor and against directly adjacent
    tiers of the SAME company) -- direct EDGAR access to confirm the exact
    real row was unavailable in this environment, so this is reasoned from
    the code's own established failure pattern, not a byte-for-byte
    real-filing fixture like the cases above.
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
        if cell.startswith("(") and not cell.endswith(")"):
            if i + 1 < len(raw_cells) and raw_cells[i + 1].startswith(")"):
                merged_cells.append(cell + raw_cells[i + 1])
                i += 2
                continue
            # A colspan-duplicated open-paren cell, closing paren in a THIRD
            # separate cell -- real bug, confirmed on AAPL's actual
            # 2004-02-10 10-Q (2026-08-01, tenq.py Phase 3): "Net income
            # (loss)" renders as ['(8', '(8', ')'] (the negative value
            # duplicated by colspan exactly like every other figure in the
            # row, THEN its closing paren alone) -- not the simpler 2-cell
            # ['(25', ')'] split already handled above. Without this, "(8"
            # parsed as a positive 8, turning a real net LOSS into a profit.
            if (i + 2 < len(raw_cells) and raw_cells[i + 1] == cell
                    and raw_cells[i + 2].startswith(")")):
                merged_cells.append(cell + raw_cells[i + 2])
                i += 3
                continue
        merged_cells.append(cell)
        i += 1

    tokens: list[tuple[str, float | None]] = []
    for cell in merged_cells:
        if _is_spacer(cell):
            continue
        v = _parse_value(cell)
        if v is not None:
            tokens.append((cell, v))
        elif _is_placeholder(cell):
            tokens.append((cell, None))
        # else: arbitrary non-numeric text (e.g. a duplicated label fragment
        # bleeding in from an HTML colspan) -- not a real slot, dropped.
    if len(tokens) > n_years:
        deduped: list[tuple[str, float | None]] = []
        i = 0
        while i < len(tokens):
            if (i + 1 < len(tokens) and tokens[i][1] is not None
                    and tokens[i][1] == tokens[i + 1][1]):
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
        # A row with its OWN real numeric data (a share-count row, or any other
        # line item) must not have its LOCAL caption read as the table's
        # governing one -- real bug, confirmed on TXN's actual 2006 10-K
        # (2026-07-30): that table states no dollar-figure caption in its own
        # cells at all (its real "(in millions)" caption lives in a preceding
        # paragraph outside the parsed table), so its ONLY units mention is a
        # share-count row's local "...shares outstanding, in thousands" --
        # which the code below then wrongly applied to net_revenue/net_income
        # too, understating both 1000x.
        #
        # Generalized past "shares" specifically -- real bug, confirmed on
        # FHI's actual 2001 10-K (2026-08-06): its real "Total revenue"/"Total
        # assets" figures are (thousands)-scale, but a DIFFERENT, unrelated
        # line item -- "MANAGED AND ADMINISTERED ASSETS AT PERIOD END (in
        # millions)" (an asset-manager-specific metric, not the company's own
        # balance sheet) -- states its OWN "(in millions)" caption inline with
        # its label, and (having no "shares" keyword) sailed straight past the
        # old shares-only filter, understating the real dollar fields 1000x.
        # The actual distinguishing trait of a genuine table-WIDE caption row
        # isn't its label text at all: it's that the row carries NO real
        # numeric data of its own (confirmed across every real caption row
        # seen so far -- ZION's fixture below, where the caption occupies the
        # label cell itself with every other cell blank; AME's, where the
        # label is empty and the caption text merely repeats across otherwise
        # non-numeric cells). A row that has REAL data alongside a units
        # phrase -- TXN's share count, FHI's AUM figures -- is reporting an
        # actual line item with its own LOCAL caption, not the table's
        # governing one, regardless of what that phrase says. See
        # _is_caption_only_row.
        #
        # Before falling all the way back to the whole document, try the text
        # right after the real "Item 6" heading (see _item6_heading_text) --
        # real bug, confirmed on ATR's actual 2002 10-K (2026-08-06): its
        # caption ("In millions of dollars") lives in exactly that gap, but
        # the whole-document mode-based scan below picked "thousands" instead,
        # outnumbered 9-to-1 by OTHER tables' captions in the same combined
        # submission -- a wrong pick, not an absent one, so table_text's own
        # empty check can't catch it. Narrowing to the heading-anchored window
        # first resolves it correctly without touching the whole-document
        # fallback still needed when no Item 6 heading is found at all.
        table_years = _find_year_columns(table)
        caption_rows = table[table.apply(lambda r: _is_caption_only_row(r, table_years), axis=1)]
        table_text = " ".join(str(c) for c in caption_rows.to_numpy().flatten())
        heading_text = _item6_heading_text(resp.text)
        if _UNITS_RE.search(table_text):
            unit_multiplier = detect_unit_multiplier(table_text, prefer_first=True)
        elif _UNITS_RE.search(heading_text):
            unit_multiplier = detect_unit_multiplier(heading_text, prefer_first=True)
        else:
            unit_multiplier = detect_unit_multiplier(resp.text)
        # One cover-page parse per filing. Item 6 tables span up to 5 fiscal
        # years at once, but the cover page's "as of" date only genuinely
        # describes the CURRENT (most recent) one -- attaching it to every
        # year in the table would misattribute a recent share count to years
        # up to 4 back. Older years get their own correct value once (if)
        # THEIR OWN dedicated filing is chained in -- the whole point of
        # chaining Item 6 tables across a company's history (module docstring).
        shares_outstanding, shares_outstanding_asof = cover_page.extract_shares_outstanding(
            resp.text, row.date_filed)
        years_items = extract_years(table, unit_multiplier)
        current_year = max((int(y) for y in years_items), default=None)
        for year_str, items in years_items.items():
            year = int(year_str)
            if not (_FISCAL_YEAR_MIN <= year <= _FISCAL_YEAR_MAX):
                continue
            ratios = compute_ratios(items, unit_scale=1)
            row_dict = {**items, **ratios, "fiscal_year": year,
                        "fundamentals_available_date": row.date_filed,
                        "item6_form": row.form_type, "item6_filename": row.filename}
            if year == current_year:
                row_dict["shares_outstanding"] = shares_outstanding
                row_dict["shares_outstanding_asof"] = shares_outstanding_asof
            rows.append(row_dict)
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
