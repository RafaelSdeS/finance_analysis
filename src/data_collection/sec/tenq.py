"""sec/tenq.py — 10-Q inline-HTML statement parser (2001-2006 quarterly tier).

Phase 3 of docs/US_QUARTERLY_BACKFILL_PLAN.md: fills the 2001-2006 window with
REAL quarterly resolution (Q1-Q3; Q4 is derived cross-tier against item6's
annual total in fundamentals.py, since 10-Qs only ever cover Q1-Q3).

Why a new file, not an extension of selected_financial_data.py: that module's
entire table locator/extractor is built around a SINGLE year-header row (one
token per YEAR). A 10-Q's statement-of-operations uses a TWO-row period
header whose spans mark off "Three/Six/Nine Months Ended" blocks, each paired
with its own date row below it, and each period appears TWICE per statement
(current-quarter column + prior-year comparative column) -- an incompatible
table shape, not a small variation on Item 6's.

Verified against AAPL's real 2004-02-10 (fiscal Q1'04) and 2004-08-05 (fiscal
Q3'04) 10-Qs (2026-08-01): Reg S-X captions ("CONDENSED CONSOLIDATED
STATEMENTS OF OPERATIONS") are real, but pandas.read_html doesn't preserve
them AS rows inside the parsed table (they're surrounding HTML text, outside
the <table> boundary) -- so table location here scores on having a genuine
2-row period header instead, same shape as
selected_financial_data.find_selected_financial_data_table's year-count-first
scoring, just swapped to block-count-first. Q3'04's real income statement
(39 tables in that filing) reconciled exactly: Net sales 2014/1545 (3mo cur/
prior) and 5929/4492 (9mo cur/prior); Cost of sales 1455/1117.

Scope, deliberately smaller than a full statement set: income statement only
(net_revenue, net_income, cost_of_revenue). These are printed as a DISCRETE
~3-month figure directly (confirmed above) -- no YTD differencing needed at
all, unlike EX-27's cumulative tags (fds.py). Cash-flow statement figures
(cashflow_ops, capex) ARE cumulative-YTD-only in a 10-Q, would need
companyfacts.ytd_to_discrete plus a genuinely different per-column period-
length reconciliation against the income-statement rows -- real added
complexity and a second table-location problem not yet verified against live
data. NOT attempted this pass; flagged so it isn't mistaken for coverage that
exists. Balance-sheet (instant) items are also not parsed -- a balance
sheet's header is two bare DATES with no period-length dimension at all (a
third distinct table shape), and item6's existing annual total_assets/equity
rows already cover the same 2001-2006 window at annual granularity.
"""

import io
import re

import pandas as pd

from . import cover_page, http
from .selected_financial_data import (
    _is_caption_only_row, _normalize_label, _row_text, _row_values, _UNITS_RE, detect_unit_multiplier,
)
from ..ratios import compute_ratios

# Filings in this window can plausibly carry real 10-Q HTML statements --
# bounded the same way fds.py's EX27_ERA_END and selected_financial_data's
# GAP_ERA are (fetching a company's entire filing history to check a handful
# of years was already a confirmed real bug in both of those).
TENQ_ERA_START = "2001-01-01"
TENQ_ERA_END = "2007-12-31"

_LENGTH_WORDS = {"THREE": 3, "SIX": 6, "NINE": 9, "TWELVE": 12}
_LENGTH_RE = re.compile(r"\b(THREE|SIX|NINE|TWELVE)\s+MONTHS\s+ENDED\b")
_YEAR_ENDED_RE = re.compile(r"\bYEAR\s+ENDED\b")
_DATE_LIKE_RE = re.compile(r"[A-Za-z]{3,}\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}")

# Short, distinctive substrings feeding the same two-pass (exact, then
# substring) matching as selected_financial_data's ROW_ALIASES. NOT reusing
# that module's ROW_ALIASES: its "eps_basic": ["basic"] / "dividends_per_
# share": ["declared"] entries are tuned to Item 6's terse 5-year-summary
# labels and would mismatch badly against a full statement of operations.
ROW_ALIASES = {
    "net_revenue": ["net sales", "net revenue", "total revenue", "total net sales", "total revenues"],
    "net_income": ["net income", "net earnings", "net loss"],
    "cost_of_revenue": ["cost of sales", "cost of revenue", "cost of goods sold"],
}

_INCOME_KEYWORDS = ("NET SALES", "NET REVENUE", "TOTAL REVENUE", "COST OF SALES", "COST OF REVENUE")


def _row_period_lengths(row: pd.Series) -> list[int | None]:
    out = []
    for c in row.tolist():
        s = str(c).upper()
        m = _LENGTH_RE.search(s)
        if m:
            out.append(_LENGTH_WORDS[m.group(1)])
        elif _YEAR_ENDED_RE.search(s):
            out.append(12)
        else:
            out.append(None)
    return out


def _row_dates(row: pd.Series) -> list[pd.Timestamp | None]:
    out = []
    for c in row.tolist():
        s = str(c).strip()
        if not _DATE_LIKE_RE.search(s):
            out.append(None)
            continue
        ts = pd.to_datetime(s, errors="coerce")
        out.append(ts if pd.notna(ts) else None)
    return out


def parse_period_header(table: pd.DataFrame) -> list[tuple[int, pd.Timestamp]]:
    """Scan the table's first few rows for a period-LENGTH row ("Three/Six/
    Nine Months Ended" / "Year Ended") followed by a DATES row, and return one
    (months, period_end) pair per distinct period-block, left to right,
    deduping adjacent colspan-duplicated columns (verified real shape on
    AAPL's 2004 10-Q: a single header cell's text is repeated across every
    physical column its colspan covers, same artifact
    selected_financial_data._row_values already collapses on data rows).

    Column boundaries are deliberately NOT tracked -- extract_statement maps
    values to blocks POSITIONALLY (Nth extracted value -> Nth block), the same
    strategy selected_financial_data.extract_years already uses for years,
    because a data row's own colspan/spacer layout doesn't reliably line up
    with the header row's (real filings interleave "$" placeholder columns
    inconsistently, per _row_values's docstring).
    """
    length_row = dates_row = None
    for _, row in table.head(5).iterrows():
        lengths = _row_period_lengths(row)
        if length_row is None and sum(v is not None for v in lengths) >= 2:
            length_row = lengths
            continue
        if length_row is not None:
            dates = _row_dates(row)
            if sum(v is not None for v in dates) >= 2:
                dates_row = dates
                break
    if length_row is None or dates_row is None:
        return []

    blocks: list[tuple[int, pd.Timestamp]] = []
    for months, date in zip(length_row, dates_row):
        if months is None or date is None:
            continue
        pair = (months, date)
        if not blocks or blocks[-1] != pair:
            blocks.append(pair)
    return blocks


def find_statement_table(tables: list[pd.DataFrame], keywords: tuple[str, ...]) -> pd.DataFrame | None:
    """Best-scoring candidate for a period-blocked financial statement among a
    filing's pandas.read_html tables. Same scoring shape as
    selected_financial_data.find_selected_financial_data_table (structure
    signal first, keyword score second, row count last) -- swapped from bare
    YEAR count to PERIOD-BLOCK count, since a 10-Q has no single "Item 6"-style
    caption inside the parsed table to search for at all (verified: Reg S-X
    captions live in surrounding HTML text, not the table pandas.read_html
    returns)."""
    best, best_score = None, (0, 0, 0)
    for df in tables:
        if df.shape[1] < 3:
            continue
        blocks = parse_period_header(df)
        if len(blocks) < 2:
            continue
        # Reject a common-size (percentage-of-revenue) MD&A table -- real bug,
        # confirmed on NSYS's real 2003-08-14 10-Q (2026-08-01): a "Results of
        # Operations as a Percentage of Net Sales" table has the identical
        # period-block header shape and matches "NET SALES"/"COST OF..." in
        # its first column just as well as the real dollar statement, so it
        # won the block-count/keyword-score tie and got picked -- extracting
        # net_revenue=100 (its Net Sales row IS, by definition, ~100%) instead
        # of the real dollar figure. The tell: its "%"-placeholder cells
        # outnumber "$"-placeholder cells table-wide, the inverse of a real
        # statement (which may have ONE legitimate "gross margin %" sub-row,
        # but is otherwise "$"-dominant -- confirmed on AAPL's real statement,
        # 4 "$" cells vs 2 "%" cells).
        symbols = [str(c).strip() for c in df.to_numpy().flatten()]
        if symbols.count("%") > symbols.count("$"):
            continue
        # Whitespace-normalize before keyword matching -- real bug, confirmed
        # on AAPL's actual 2004-02-10 10-Q (2026-08-01): its real 43-row
        # income statement renders "Cost of  sales" (embedded double space,
        # an HTML-entity artifact), which silently failed a plain "COST OF
        # SALES" substring check while a smaller 7-row MD&A summary table
        # elsewhere in the SAME filing (clean single-space text) matched both
        # keywords and won the tie on keyword score alone -- the real
        # statement lost to a decoy fragment missing net_income entirely.
        firstcol = re.sub(r"\s+", " ", _row_text(df.iloc[:, 0])).upper()
        score = sum(k in firstcol for k in keywords)
        if score < 1:
            continue
        key = (len(blocks), score, df.shape[0])
        if key > best_score:
            best, best_score = df, key
    return best


def _table_unit_multiplier(table: pd.DataFrame, doc_text: str) -> float:
    """Winning table's own caption first, whole document only when the table
    states none of its own -- same rule and reasoning as
    selected_financial_data.detect_unit_multiplier's docstring (a table's own
    caption can differ from the filing's dominant one).

    Caption-row detection reuses selected_financial_data._is_caption_only_row
    (no year-header concept here, so its exemption stays off by default) --
    NOT a blanket "shares" keyword filter. Real bug, confirmed 2026-08-12:
    that module already replaced the same blanket filter for the identical
    reason (see its build_cik_history docstring) -- it let a non-shares LOCAL
    caption on an unrelated real data row slip through as if it were the
    table's governing caption, AND wrongly excluded legitimate rows that
    merely happened to mention the word "shares" without being a caption at
    all. The real distinguishing trait is whether the row carries any real
    numeric data of its own, not its label text."""
    caption_rows = table[table.apply(_is_caption_only_row, axis=1)]
    table_text = " ".join(str(c) for c in caption_rows.to_numpy().flatten())
    if _UNITS_RE.search(table_text):
        return detect_unit_multiplier(table_text, prefer_first=True)
    return detect_unit_multiplier(doc_text)


def extract_statement(table: pd.DataFrame, unit_multiplier: float = 1.0) -> dict[tuple[pd.Timestamp, int], dict]:
    """The located table -> {(period_end, months): {line_item: value}}, one
    entry per period-block found by parse_period_header. Two-pass alias
    matching (exact label, then substring), never overwrites an
    already-assigned item -- identical rule to
    selected_financial_data.extract_years, same reasoning (a short precise
    label must win before a longer label's substring collision is considered)."""
    blocks = parse_period_header(table)
    if not blocks:
        return {}
    n = len(blocks)
    out = {(end, months): {} for months, end in blocks}
    rows = [(_normalize_label(row.iloc[0]), row) for _, row in table.iterrows()]

    def assign(label_test):
        for item, aliases in ROW_ALIASES.items():
            if any(item in v for v in out.values()):
                continue  # already assigned by a higher-priority pass
            for label, row in rows:
                if label and label_test(label, aliases):
                    vals = _row_values(row.iloc[1:], n)
                    for (months, end), v in zip(blocks, vals):
                        if v is not None:
                            out[(end, months)][item] = v * unit_multiplier
                    break

    assign(lambda label, aliases: label in aliases)
    assign(lambda label, aliases: any(a in label for a in aliases))
    return {k: v for k, v in out.items() if v}


def _current_quarter_items(three_month_blocks: dict[tuple, dict]) -> tuple[pd.Timestamp, dict] | None:
    """Among the discrete 3-month blocks a table's items were extracted for
    (current quarter + prior-year comparative, per Reg S-X convention), the
    CURRENT one -- selecting by MAX end date is robust to column order
    without assuming Reg S-X always lists the current period first."""
    if not three_month_blocks:
        return None
    key = max(three_month_blocks, key=lambda k: k[0])
    return key[0], three_month_blocks[key]


def build_cik_history(cik: int, filings: pd.DataFrame) -> pd.DataFrame:
    """Chain every 10-Q in the tenq era for one CIK, extracting each filing's
    income statement's DISCRETE 3-month block (the current quarter's own
    directly-printed figures, never the redundant YTD block alongside it) and
    keeping the EARLIEST filing per period end -- as-first-reported, same
    rule as every other tier (plan §3.3).
    """
    cik_filings = filings[(filings["cik"] == cik) & (filings["form_type"].str.startswith("10-Q"))
                           & (filings["date_filed"] >= TENQ_ERA_START) & (filings["date_filed"] <= TENQ_ERA_END)]
    rows = []
    for row in cik_filings.itertuples():
        resp = http.get(f"https://www.sec.gov/Archives/{row.filename}")
        if resp is None:
            continue
        try:
            tabs = pd.read_html(io.StringIO(resp.text))
        except Exception:
            # Same "skip a filing that doesn't parse cleanly" rule as
            # selected_financial_data.build_cik_history (pd.read_html raises
            # more than ValueError on malformed real-world HTML) -- also
            # covers the plain-.txt pre-HTML era filings (no <table> at all).
            continue
        if not tabs:
            continue

        table = find_statement_table(tabs, _INCOME_KEYWORDS)
        if table is None:
            continue
        um = _table_unit_multiplier(table, resp.text)
        three_mo = {k: v for k, v in extract_statement(table, um).items() if k[1] == 3}
        cur = _current_quarter_items(three_mo)
        if cur is None:
            continue
        end, items = cur
        if "net_revenue" not in items:
            continue
        shares_outstanding, shares_outstanding_asof = cover_page.extract_shares_outstanding(
            resp.text, row.date_filed)
        rows.append({**items, "end": end, "period_months": 3,
                     "flows_derived": 0, "flows_defined": 1,
                     "fundamentals_available_date": row.date_filed,
                     "tenq_form": row.form_type, "tenq_filename": row.filename,
                     "shares_outstanding": shares_outstanding,
                     "shares_outstanding_asof": shares_outstanding_asof})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["cik"] = cik
    # as-first-reported: whichever filing disclosed a given quarter EARLIEST wins.
    df = (df.sort_values("fundamentals_available_date")
            .drop_duplicates(subset="end", keep="first")
            .sort_values("end")
            .reset_index(drop=True))
    ratios = df.apply(lambda r: compute_ratios(r.to_dict(), unit_scale=1), axis=1, result_type="expand")
    df[ratios.columns] = ratios
    return df
