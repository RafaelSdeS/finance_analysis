"""
test_sec_company_info.py
=========================
Self-check for sec/company_info.py (no network; mocks http.get). Uses real
temp files for the crosswalk and output paths rather than blanket-mocking
pandas.read_parquet, since collect_company_info now reads from two distinct
paths (crosswalk input, its own output for resume) that must behave
independently.

Usage: python tests/data_collection/test_sec_company_info.py
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_collection.sec import company_info


def _fake_submissions(sic: str, sic_description: str) -> mock.Mock:
    return mock.Mock(text=json.dumps({"sic": sic, "sicDescription": sic_description}))


def test_collect_company_info_extracts_sic_and_description():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cw_path = tmp / "crosswalk.parquet"
        pd.DataFrame({"ticker": ["AAPL", "XOM"], "cik": [320193, 34088]}).to_parquet(cw_path, index=False)
        out_path = tmp / "company_info.parquet"
        responses = {320193: _fake_submissions("3571", "Electronic Computers"),
                     34088: _fake_submissions("2911", "Petroleum Refining")}

        with mock.patch.object(company_info.crosswalk, "CROSSWALK_PATH", cw_path), \
             mock.patch.object(company_info.config, "US_COMPANY_INFO_PATH", out_path), \
             mock.patch.object(company_info.config, "US_SEC_DIR", tmp), \
             mock.patch.object(company_info.http, "get",
                                side_effect=lambda url: responses[int(url.rsplit("CIK", 1)[1].split(".")[0])]):
            df = company_info.collect_company_info(["AAPL", "XOM"])

        assert len(df) == 2
        aapl = df[df["ticker"] == "AAPL"].iloc[0]
        assert aapl["sic"] == "3571" and aapl["sic_description"] == "Electronic Computers"
        assert out_path.exists(), "must persist the output file, not just return the DataFrame"
    print("OK: sic/sic_description correctly extracted per ticker via submissions.json")


def test_unresolvable_ticker_and_failed_fetch_are_skipped_not_fatal():
    # Real shape of failure this guards against: one bad ticker (no crosswalk entry,
    # or a transient SEC fetch failure after http.py's retries are exhausted) must not
    # crash the whole batch -- mirrors collect_fundamentals_us's per-ticker try/except.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cw_path = tmp / "crosswalk.parquet"
        pd.DataFrame({"ticker": ["AAPL"], "cik": [320193]}).to_parquet(cw_path, index=False)
        out_path = tmp / "company_info.parquet"

        with mock.patch.object(company_info.crosswalk, "CROSSWALK_PATH", cw_path), \
             mock.patch.object(company_info.config, "US_COMPANY_INFO_PATH", out_path), \
             mock.patch.object(company_info.config, "US_SEC_DIR", tmp), \
             mock.patch.object(company_info.http, "get", return_value=None):
            df = company_info.collect_company_info(["NOTINCROSSWALK", "AAPL"])

        assert len(df) == 0, "NOTINCROSSWALK has no CIK, AAPL's fetch failed -- both must be skipped, not crash"
    print("OK: an unresolvable ticker and a failed fetch are both skipped, batch completes")


def test_resume_skips_already_resolved_tickers_and_checkpoints_progress():
    # Real risk this guards against: collect_company_info builds one combined output
    # file (unlike fundamentals.py's one-file-per-ticker), so a kill/crash mid-run used
    # to lose ALL in-memory progress -- confirmed by watching a real 10,432-ticker run
    # sit with zero persisted output for 20+ minutes. Confirms both halves: an existing
    # ticker on disk is skipped (not re-fetched), and a mid-run checkpoint (flush_every)
    # actually persists partial progress before the run finishes.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cw_path = tmp / "crosswalk.parquet"
        pd.DataFrame({"ticker": ["AAPL", "XOM", "MSFT", "KO"],
                      "cik": [320193, 34088, 789019, 21344]}).to_parquet(cw_path, index=False)
        out_path = tmp / "company_info.parquet"
        # AAPL already resolved on disk from a prior (interrupted) run.
        pd.DataFrame({"ticker": ["AAPL"], "cik": [320193], "sic": ["3571"],
                      "sic_description": ["Electronic Computers"]}).to_parquet(out_path, index=False)

        fetched = []

        def fake_get(url):
            cik = int(url.rsplit("CIK", 1)[1].split(".")[0])
            fetched.append(cik)
            return _fake_submissions("9999", "Test Industry")

        with mock.patch.object(company_info.crosswalk, "CROSSWALK_PATH", cw_path), \
             mock.patch.object(company_info.config, "US_COMPANY_INFO_PATH", out_path), \
             mock.patch.object(company_info.config, "US_SEC_DIR", tmp), \
             mock.patch.object(company_info.http, "get", side_effect=fake_get):
            df = company_info.collect_company_info(["AAPL", "XOM", "MSFT", "KO"], flush_every=2)

        assert 320193 not in fetched, "AAPL already on disk -- must not be re-fetched"
        assert set(fetched) == {34088, 789019, 21344}, f"the 3 new tickers must all be fetched, got {fetched}"
        assert len(df) == 4, "final result must include the resumed AAPL row plus the 3 newly fetched"
        assert set(df["ticker"]) == {"AAPL", "XOM", "MSFT", "KO"}

        on_disk_after = pd.read_parquet(out_path)
        assert len(on_disk_after) == 4, "final flush must persist all 4 rows, not just the checkpoint's subset"
    print("OK: resume skips already-resolved tickers, and progress checkpoints to disk mid-run")


if __name__ == "__main__":
    test_collect_company_info_extracts_sic_and_description()
    test_unresolvable_ticker_and_failed_fetch_are_skipped_not_fatal()
    test_resume_skips_already_resolved_tickers_and_checkpoints_progress()
