"""sec/cover_page.py — shares-outstanding cover-page parser (pre-2009 tiers).

Every 10-K/10-Q's cover page has always been required (Reg S-K/Exchange Act
rules that pre-date EDGAR, not an XBRL-era convention) to state the
registrant's shares outstanding as of a recent date. None of the three
pre-2009 fundamentals tiers extracted it before this (fds.py's own header
comment already flagged EX-27 Article 5 as never carrying a shares-count tag
at all; item6/tenq never attempted it either) -- only companyfacts.py's XBRL
tier (2009+, via CommonStockSharesOutstanding/EntityCommonStockSharesOutstanding)
did, which is why market_cap/pl/earnings_yield_vs_selic read 0% populated for
every year before 2009 (cross-vendor validation finding, 2026-08-12).

This closes that gap using text ALREADY fetched by all three tiers (the full
submission .txt, not just the exhibit) -- zero new HTTP calls, parsing only.

Verified against real live filings (2026-08-12), at least 3 distinct
cover-page templates in the wild:
  AAPL 10-K, filed 1994-12-13 (accession 0000320193-94-000016):
    "119,891,418 shares of Common Stock Issued and Outstanding as of ..."
  XOM 10-K, filed 2002-03-27 (0000930661-02-000889):
    "Common Stock, without par value (6,792,598,170 shares\n   outstanding
     at February 28, 2002)" -- embedded in the securities-registered table,
     not a standalone sentence.
  XOM 10-Q, filed 2003-05-14 (0000034088-03-000063):
    a tabular "Class / Outstanding as of March 31, 2003" header over
    "Common stock, without par value    6,679,390,610".

A 4th template ("As of [date], there were N shares ... outstanding.") is
also matched -- common, well-known SEC boilerplate, but NOT individually
confirmed against a live fetch this session the way the 3 above were.

Real caveat, not fixed here: EDGAR electronic filing was phased in 1993-1996
(mandatory by 1996) -- there is no free source, structured or text, for
shares outstanding before that. This closes the gap to roughly 1994/1996-2008,
not to 1962 (the price panel's own, unrelated floor).

Deliberately not attempted: multi-class registrants (e.g. dual-class Class A/
Class B common stock) -- the tabular template can list more than one class
row, and this returns only the FIRST class matched, not a sum across classes
(summing would be wrong whenever only one class is publicly traded/priced).
Narrower coverage for those names, not a wrong number.
"""

import re

import pandas as pd

# Cover-page material sits in the first few KB of a filing's full submission
# text, always before Item 1's body -- verified against all 3 filings above
# (real matches land within the first ~5,000 chars of each). 40,000 leaves
# comfortable margin for an unusually padded cover page while staying well
# short of Item 1, where "shares...outstanding" language reappears in
# unrelated contexts (beneficial-ownership disclosures, stock plans, ...).
_COVER_WINDOW = 40_000

_NUM = r"([\d,]{4,})"  # >=4 digits w/ commas -- a real share count is never a bare 1-3 digit number
_DATE = r"([A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4})"

# Tried in order; first match in the cover-page window wins. Each pattern
# captures exactly 2 groups -- (shares, date) except the two date-first ones
# (indices in _DATE_FIRST), which are (date, shares) since those templates
# state the date before the number.
_PATTERNS = [
    # AAPL-style, verified live: "119,891,418 shares of Common Stock Issued
    # and Outstanding as of March 1, 1995"
    re.compile(_NUM + r"\s+shares\s+of\b[^.\n]{0,80}?\bOutstanding\s+as\s+of\s+" + _DATE, re.I),
    # XOM 10-K-style, verified live: "(6,792,598,170 shares\n   outstanding
    # at February 28, 2002)"
    re.compile(_NUM + r"\s+shares\s+outstanding\s+(?:at|as\s+of)\s+" + _DATE, re.I),
    # Common SEC cover-page boilerplate ("As of [date], there were N shares
    # of common stock outstanding.") -- NOT individually verified against a
    # live fetch this session (unlike the other 3), included because it's a
    # widely-known standard phrasing distinct from the shapes above (date
    # BEFORE the share count, no repeated trailing date).
    re.compile(r"as\s+of\s+" + _DATE + r",?\s+there\s+were\s+" + _NUM
               + r"\s+shares\b[^.\n]{0,120}?\boutstanding\b", re.I),
    # XOM 10-Q-style tabular header, verified live: "Outstanding as of
    # March 31, 2003" then, within the table body, "Common stock, without
    # par value  6,679,390,610"
    re.compile(r"Outstanding\s+as\s+of\s+" + _DATE + r"[\s\S]{0,400}?Common\s+[Ss]tock[^\n]{0,60}?" + _NUM, re.I),
]
_DATE_FIRST = {2, 3}  # indices into _PATTERNS whose groups are (date, shares) not (shares, date)

_MIN_SHARES, _MAX_SHARES = 1_000, 1e12
_MAX_ASOF_LAG_DAYS = 270  # a cover-page "as of" date is always shortly BEFORE its own filing


def _to_shares(s: str) -> float:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return float("nan")


def extract_shares_outstanding(text: str | None, filing_date) -> tuple[float, pd.Timestamp]:
    """Best-effort single shares-outstanding figure off a 10-K/10-Q cover
    page, plus the date it was stated as of. (nan, NaT) if nothing in the
    cover-page window matches a known template or passes the sanity checks
    below -- never guessed, same convention as every other tier in this
    pipeline (loaders.load_dividends, fds.py's multiplier resolution, ...).

    `filing_date` anchors plausibility: a cover-page "as of" date is always
    on or shortly before the filing that states it, never a stray date from
    unrelated boilerplate elsewhere in the document -- rejects any match
    whose date falls outside [-5, +270] days of `filing_date`.
    """
    if not text:
        return float("nan"), pd.NaT
    window = text[:_COVER_WINDOW]
    filing_date = pd.Timestamp(filing_date)
    for i, pat in enumerate(_PATTERNS):
        for m in pat.finditer(window):
            if i in _DATE_FIRST:
                date_str, num_str = m.group(1), m.group(2)
            else:
                num_str, date_str = m.group(1), m.group(2)
            shares = _to_shares(num_str)
            if not (_MIN_SHARES <= shares <= _MAX_SHARES):
                continue
            asof = pd.to_datetime(date_str, errors="coerce")
            if pd.isna(asof):
                continue
            lag = (filing_date - asof).days
            if not (-5 <= lag <= _MAX_ASOF_LAG_DAYS):
                continue
            return shares, asof
    return float("nan"), pd.NaT
