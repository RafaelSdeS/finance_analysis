"""sec/universe.py — point-in-time US public-company roster from EDGAR full-index.

Every quarter since 1994Q1, SEC publishes master.idx: every filing that
quarter, pipe-delimited (cik|company_name|form_type|date_filed|filename) --
format confirmed stable across the full 1994-2026 span (2026-07-28). A CIK
filing a 10-K/10-Q variant that quarter is "actively reporting" -- this is
what makes the roster survivorship-bias-free: Enron/Lehman/WorldCom/Twitter
all show up here because the index is built from what was ACTUALLY FILED at
the time, not from any current-day company list (which silently drops them
-- see sec/crosswalk.py's tier-1 SEC company_tickers.json for exactly that
failure mode).

Two-tier output:
  - filings cache: one row per qualifying (cik, form_type, date_filed) filing,
    cached per-quarter (config.US_SEC_DIR/full_index/{year}q{q}.parquet) --
    a rerun only re-fetches the current in-progress quarter; closed quarters
    are treated as immutable (ponytail: a rare late-indexed filing landing in
    an already-closed quarter would be missed until a full rebuild -- accepted,
    full-index snapshots are near-universally stable once a quarter closes).
  - roster: one row per (cik, year), the actual survivorship-bias-free
    universe -- every CIK that filed a 10-K/10-Q that calendar year.
"""

import logging
import re
from datetime import date

import pandas as pd

from .. import config
from . import http

log = logging.getLogger("sec")

FULL_INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx"
# 1993 pre-dates EDGAR's mandatory e-filing rollout and is effectively empty
# (verified 2026-07-28: 1993Q1 full-index is 14 lines, header only).
FIRST_YEAR, FIRST_QTR = 1994, 1
CACHE_DIR = config.US_SEC_DIR / "full_index"
FILINGS_PATH = config.US_SEC_DIR / "edgar_10k10q_filings.parquet"
ROSTER_PATH = config.US_SEC_DIR / "us_universe_roster.parquet"

# Matches 10-K/10-Q and every historical variant (10-K405, 10-KSB, .../A amendments)
# but NOT "NT 10-K"/"NT 10-Q" (late-filing notifications -- not an actual filing).
_QUALIFYING_FORM = re.compile(r"^10-K|^10-Q")


def _quarters_through_now() -> list[tuple[int, int]]:
    today = date.today()
    current_q = (today.month - 1) // 3 + 1
    out = []
    for year in range(FIRST_YEAR, today.year + 1):
        max_q = current_q if year == today.year else 4
        out.extend((year, q) for q in range(1, max_q + 1))
    return out


def parse_master_idx(text: str) -> pd.DataFrame:
    """Pure parse of one master.idx's text -> qualifying-forms DataFrame. No network;
    the one function worth unit-testing without hitting EDGAR (see test_sec_universe.py)."""
    lines = text.splitlines()
    rule = next((i for i, l in enumerate(lines) if l.startswith("---")), None)
    if rule is None:
        return pd.DataFrame(columns=["cik", "company_name", "form_type", "date_filed", "filename"])
    rows = [l.split("|") for l in lines[rule + 1:] if l.count("|") == 4]
    df = pd.DataFrame(rows, columns=["cik", "company_name", "form_type", "date_filed", "filename"])
    if df.empty:
        return df
    df = df[df["form_type"].str.match(_QUALIFYING_FORM)].copy()
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")
    df["date_filed"] = pd.to_datetime(df["date_filed"], errors="coerce")
    return df.dropna(subset=["cik", "date_filed"]).reset_index(drop=True)


def fetch_quarter(year: int, q: int) -> pd.DataFrame | None:
    """Cached per quarter; only the current in-progress quarter is re-fetched."""
    cache = CACHE_DIR / f"{year}q{q}.parquet"
    is_current = (year, q) == max(_quarters_through_now())
    if cache.exists() and not is_current:
        return pd.read_parquet(cache)
    resp = http.get(FULL_INDEX_URL.format(year=year, q=q))
    if resp is None:
        return None
    df = parse_master_idx(resp.text)
    df["quarter"] = f"{year}Q{q}"
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def build_filings(progress: bool = True) -> pd.DataFrame:
    """Fetch/cache every quarter 1994Q1->now; return the concatenated 10-K/10-Q filings table.
    Long-running on a cold cache (~130 quarters, ~1-2GB of full-index text) -- run via
    `python -m src.data_collection.sec.universe` for visible per-quarter progress."""
    frames = []
    quarters = _quarters_through_now()
    for i, (year, q) in enumerate(quarters):
        df = fetch_quarter(year, q)
        if df is not None and len(df):
            frames.append(df)
        if progress:
            log.info("sec universe: %d/%d quarters done (%dQ%d) — %d qualifying filings so far",
                      i + 1, len(quarters), year, q, sum(len(f) for f in frames))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    config.US_SEC_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(FILINGS_PATH, index=False)
    return result


def build_roster(filings: pd.DataFrame | None = None) -> pd.DataFrame:
    """(cik, year) roster: every CIK that filed a 10-K/10-Q that calendar year, plus
    the latest company_name seen that year and a filing count."""
    if filings is None:
        filings = pd.read_parquet(FILINGS_PATH) if FILINGS_PATH.exists() else build_filings()
    f = filings.copy()
    f["year"] = f["date_filed"].dt.year
    f = f.sort_values("date_filed")
    roster = (f.groupby(["cik", "year"])
                .agg(company_name=("company_name", "last"), n_filings=("form_type", "size"))
                .reset_index())
    config.US_SEC_DIR.mkdir(parents=True, exist_ok=True)
    roster.to_parquet(ROSTER_PATH, index=False)
    return roster


def compute_coverage(roster: pd.DataFrame, crosswalk: pd.DataFrame, price_dir=None) -> pd.DataFrame:
    """Per-year: roster size (all CIKs filing a 10-K/10-Q that year) vs. how many
    resolve to a ticker (crosswalk) AND have a price file on disk. This is the
    measured-not-just-acknowledged survivorship-bias number from the plan's §4.2.

    IMPORTANT: with only a tier-1 crosswalk (current listings, see crosswalk.py),
    "not priced" conflates two different things -- genuinely no price data
    collected yet, OR a dead company whose ticker this crosswalk tier can't
    recover at all (needs tiers 2-4). Coverage here is a LOWER BOUND, not the
    final number; don't read it as more precise than that.
    """
    price_dir = price_dir or config.US_PRICES_DIR
    have_tickers = {p.stem for p in price_dir.glob("*.parquet")} if price_dir.exists() else set()
    cik_to_ticker = dict(zip(crosswalk["cik"], crosswalk["ticker"]))

    rows = []
    for year, grp in roster.groupby("year"):
        ciks = set(grp["cik"])
        priced = sum(1 for c in ciks if cik_to_ticker.get(c) in have_tickers)
        rows.append({"year": int(year), "roster_ciks": len(ciks), "priced_ciks": priced,
                     "coverage": priced / len(ciks) if ciks else float("nan")})
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(message)s")
    filings = build_filings()
    roster = build_roster(filings)
    log.info("done: %d filings, %d (cik,year) roster rows, %d distinct CIKs",
              len(filings), len(roster), roster["cik"].nunique())
