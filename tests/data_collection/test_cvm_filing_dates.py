"""
test_cvm_filing_dates.py
=========================
_fetch_year() parses one CVM ITR/DFP yearly register into (cnpj,
reference_date, received_date) rows -- the data this repo's anti-lookahead
guarantee is built on (CLAUDE.md: "real filing dates... 41,530 filings from
1,223 companies, 100% coverage"). No test previously touched any of the five
cvm/*.py sub-modules besides ratios.py.

Covers the parsing logic in isolation (http.fetch_zip/read_csv mocked, no
real network or zip needed): cnpj digit-stripping, unparseable-date rows
dropped, and the earliest-receipt-wins dedup for restated filings (a company
can appear twice for the same quarter across filing versions -- the market
saw the numbers at the first, not the restated, receipt date).

Usage:
    python tests/data_collection/test_cvm_filing_dates.py
"""

import sys
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection.cvm import filing_dates


def _row(cnpj, cvm_code, ref, recv):
    return {"CNPJ_CIA": cnpj, "CD_CVM": cvm_code, "DT_REFER": ref, "DT_RECEB": recv}


def test_fetch_year_parses_and_strips_cnpj():
    rows = [_row("11.111.111/0001-01", "1234", "2026-03-31", "2026-04-20")]
    with mock.patch.object(filing_dates.http, "fetch_zip", return_value=object()), \
         mock.patch.object(filing_dates.http, "read_csv", return_value=rows):
        out = filing_dates._fetch_year("itr", 2026)

    assert out is not None and len(out) == 1
    r = out.iloc[0]
    assert r["cnpj"] == "11111111000101", "CNPJ must be digits-only, no punctuation"
    assert r["reference_date"] == pd.Timestamp("2026-03-31")
    assert r["received_date"] == pd.Timestamp("2026-04-20")
    assert r["report_type"] == "ITR"
    print("OK: _fetch_year parses and strips cnpj")


def test_fetch_year_keeps_earliest_receipt_for_restated_filing():
    """Same (cnpj, cvm_code, quarter) filed twice (a restatement) -- the
    market saw the numbers at the FIRST receipt, so the later, restated
    receipt date must not win."""
    rows = [
        _row("11111111000101", "1234", "2026-03-31", "2026-05-15"),  # restatement, later
        _row("11111111000101", "1234", "2026-03-31", "2026-04-20"),  # original, earlier
    ]
    with mock.patch.object(filing_dates.http, "fetch_zip", return_value=object()), \
         mock.patch.object(filing_dates.http, "read_csv", return_value=rows):
        out = filing_dates._fetch_year("dfp", 2026)

    assert len(out) == 1, "must collapse to one row per (cnpj, cvm_code, quarter)"
    assert out.iloc[0]["received_date"] == pd.Timestamp("2026-04-20")
    print("OK: earliest receipt wins for a restated filing")


def test_fetch_year_drops_rows_with_unparseable_dates():
    rows = [
        _row("11111111000101", "1234", "2026-03-31", "2026-04-20"),  # valid
        _row("22222222000102", "5678", "", "2026-04-20"),            # missing quarter-end
        _row("33333333000103", "9012", "2026-03-31", ""),            # missing receipt date
    ]
    with mock.patch.object(filing_dates.http, "fetch_zip", return_value=object()), \
         mock.patch.object(filing_dates.http, "read_csv", return_value=rows):
        out = filing_dates._fetch_year("itr", 2026)

    assert len(out) == 1
    assert out.iloc[0]["cnpj"] == "11111111000101"
    print("OK: rows with unparseable dates are dropped")


def test_fetch_year_returns_none_when_year_not_published():
    with mock.patch.object(filing_dates.http, "fetch_zip", return_value=None):
        out = filing_dates._fetch_year("dfp", 1999)

    assert out is None
    print("OK: unpublished year (404) returns None")


def test_fetch_year_returns_none_on_empty_register():
    with mock.patch.object(filing_dates.http, "fetch_zip", return_value=object()), \
         mock.patch.object(filing_dates.http, "read_csv", return_value=[]):
        out = filing_dates._fetch_year("itr", 2026)

    assert out is None
    print("OK: empty register returns None")


if __name__ == "__main__":
    test_fetch_year_parses_and_strips_cnpj()
    test_fetch_year_keeps_earliest_receipt_for_restated_filing()
    test_fetch_year_drops_rows_with_unparseable_dates()
    test_fetch_year_returns_none_when_year_not_published()
    test_fetch_year_returns_none_on_empty_register()
