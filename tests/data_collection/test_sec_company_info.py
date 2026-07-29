"""
test_sec_company_info.py
=========================
Self-check for sec/company_info.py (no network; mocks http.get and the
crosswalk read).

Usage: python tests/data_collection/test_sec_company_info.py
"""

import json
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_collection.sec import company_info


def _fake_submissions(sic: str, sic_description: str) -> mock.Mock:
    return mock.Mock(text=json.dumps({"sic": sic, "sicDescription": sic_description}))


def test_collect_company_info_extracts_sic_and_description():
    cw = pd.DataFrame({"ticker": ["AAPL", "XOM"], "cik": [320193, 34088]})
    responses = {
        320193: _fake_submissions("3571", "Electronic Computers"),
        34088: _fake_submissions("2911", "Petroleum Refining"),
    }
    with mock.patch.object(company_info.crosswalk, "CROSSWALK_PATH", mock.Mock(exists=lambda: True)), \
         mock.patch("pandas.read_parquet", return_value=cw), \
         mock.patch.object(company_info.http, "get", side_effect=lambda url: responses[
             int(url.rsplit("CIK", 1)[1].split(".")[0])]), \
         mock.patch.object(company_info.config, "US_COMPANY_INFO_PATH", Path("/tmp/unused_company_info.parquet")), \
         mock.patch("pandas.DataFrame.to_parquet"):
        df = company_info.collect_company_info(["AAPL", "XOM"])

    assert len(df) == 2
    aapl = df[df["ticker"] == "AAPL"].iloc[0]
    assert aapl["sic"] == "3571" and aapl["sic_description"] == "Electronic Computers"
    xom = df[df["ticker"] == "XOM"].iloc[0]
    assert xom["sic"] == "2911" and xom["sic_description"] == "Petroleum Refining"
    print("OK: sic/sic_description correctly extracted per ticker via submissions.json")


def test_unresolvable_ticker_and_failed_fetch_are_skipped_not_fatal():
    # Real shape of failure this guards against: one bad ticker (no crosswalk entry,
    # or a transient SEC fetch failure after http.py's retries are exhausted) must not
    # crash the whole batch -- mirrors collect_fundamentals_us's per-ticker try/except.
    cw = pd.DataFrame({"ticker": ["AAPL"], "cik": [320193]})
    with mock.patch.object(company_info.crosswalk, "CROSSWALK_PATH", mock.Mock(exists=lambda: True)), \
         mock.patch("pandas.read_parquet", return_value=cw), \
         mock.patch.object(company_info.http, "get", side_effect=[None]), \
         mock.patch.object(company_info.config, "US_COMPANY_INFO_PATH", Path("/tmp/unused_company_info.parquet")), \
         mock.patch("pandas.DataFrame.to_parquet"):
        df = company_info.collect_company_info(["NOTINCROSSWALK", "AAPL"])

    assert len(df) == 0, "NOTINCROSSWALK has no CIK, AAPL's fetch failed -- both must be skipped, not crash"
    print("OK: an unresolvable ticker and a failed fetch are both skipped, batch completes")


if __name__ == "__main__":
    test_collect_company_info_extracts_sic_and_description()
    test_unresolvable_ticker_and_failed_fetch_are_skipped_not_fatal()
