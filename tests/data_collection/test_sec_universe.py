"""
test_sec_universe.py
=====================
Self-check for sec/universe.py's pure parsing/aggregation logic (no network):
  - parse_master_idx: pipe-delimited EDGAR full-index -> qualifying-forms only.
    Must keep 10-K/10-Q and every historical variant (10-K405, .../A amendments)
    but exclude "NT 10-K" (late-filing notice, not an actual filing) and every
    non-10-K/Q form (SC 13G, S-8, etc. -- the vast majority of any real quarter).
  - compute_coverage: per-year roster-vs-priced measurement, including the
    "not priced" case (tier-1 crosswalk can't resolve a dead company's ticker).

Usage: python tests/data_collection/test_sec_universe.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_collection.sec import universe

FAKE_MASTER_IDX = """Description:           Master Index of EDGAR Dissemination Feed
Last Data Received:    March 31, 1994
Comments:              webmaster@sec.gov
Anonymous FTP:         ftp://ftp.sec.gov/edgar/


CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
100240|TURNER BROADCASTING SYSTEM INC|10-K|1994-03-31|edgar/data/100240/a.txt
100240|TURNER BROADCASTING SYSTEM INC|10-K405|1994-03-15|edgar/data/100240/b.txt
100240|TURNER BROADCASTING SYSTEM INC|10-K/A|1994-04-01|edgar/data/100240/c.txt
200001|SOME BANK INC|10-Q|1994-05-01|edgar/data/200001/d.txt
200001|SOME BANK INC|NT 10-K|1994-03-30|edgar/data/200001/e.txt
300001|SOME HOLDER|SC 13G/A|1994-02-07|edgar/data/300001/f.txt
400001|SOME COMPANY|S-8|1994-02-11|edgar/data/400001/g.txt
"""


def test_parse_master_idx():
    df = universe.parse_master_idx(FAKE_MASTER_IDX)
    assert len(df) == 4, f"expected 4 qualifying filings, got {len(df)}"
    assert set(df["form_type"]) == {"10-K", "10-K405", "10-K/A", "10-Q"}
    assert 300001 not in set(df["cik"])  # SC 13G/A excluded
    assert 400001 not in set(df["cik"])  # S-8 excluded
    nt = df[(df["cik"] == 200001)]
    assert len(nt) == 1 and nt.iloc[0]["form_type"] == "10-Q"  # NT 10-K excluded, real 10-Q kept
    print("OK: parse_master_idx keeps 10-K/Q variants, excludes NT-notices and unrelated forms")


def test_parse_master_idx_empty_input():
    assert universe.parse_master_idx("no rule line here").empty
    print("OK: parse_master_idx handles malformed input without raising")


def test_compute_coverage():
    roster = pd.DataFrame({
        "cik": [1, 2, 3, 4],
        "year": [2000, 2000, 2000, 2001],
        "company_name": ["A", "B", "C", "A"],
        "n_filings": [1, 1, 1, 1],
    })
    crosswalk = pd.DataFrame({"cik": [1, 2], "ticker": ["AAA", "BBB"]})  # cik 3, 4 unresolved (dead/tier-1 gap)

    with tempfile.TemporaryDirectory() as tmp:
        price_dir = Path(tmp)
        (price_dir / "AAA.parquet").touch()  # cik 1 -> priced
        # BBB has a ticker but no price file -> not priced
        cov = universe.compute_coverage(roster, crosswalk, price_dir=price_dir)

    row_2000 = cov[cov["year"] == 2000].iloc[0]
    assert row_2000["roster_ciks"] == 3
    assert row_2000["priced_ciks"] == 1  # only cik 1 (AAA) has both a ticker AND a price file
    assert abs(row_2000["coverage"] - 1 / 3) < 1e-9

    row_2001 = cov[cov["year"] == 2001].iloc[0]
    assert row_2001["roster_ciks"] == 1 and row_2001["priced_ciks"] == 0  # cik 4: no crosswalk entry at all
    print("OK: compute_coverage counts only ciks with BOTH a resolved ticker AND an on-disk price file")


if __name__ == "__main__":
    test_parse_master_idx()
    test_parse_master_idx_empty_input()
    test_compute_coverage()
