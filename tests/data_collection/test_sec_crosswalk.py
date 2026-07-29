"""
test_sec_crosswalk.py
======================
Self-check for sec/crosswalk.py's CIK override (no network; mocks http.get).

Real bug, found auditing the top-500 collection run (2026-07-28): SEC's
company_tickers.json occasionally points a ticker at a newly-created
holding-company shell CIK with zero (or near-zero) filing history, while the
real, decades-long filing history stays under the OLD CIK indefinitely.
Confirmed on two cases -- XOM (ticker -> CIK 2115436 "ExxonMobil Holdings
Corp", 0 filings; real filer is CIK 34088) and BLK (ticker -> CIK 2012383,
created 2024 as "BlackRock Funding, Inc."; real filer is CIK 1364742,
73 filings back to 2006) -- distinguished from a genuinely new company
(spinoff/IPO/merger) by the OLD entity's real filings continuing under a
demoted name, not a merger-shell placeholder that was always destined to
become the new public entity. CIK_OVERRIDES patches both at crosswalk-build
time so the fix survives a future refetch from SEC (a one-off edit of the
cached parquet would be silently lost the next time build_crosswalk_tier1()
runs). This test loops over CIK_OVERRIDES directly so a future addition to
that dict is covered automatically, without a matching test edit.

Usage: python tests/data_collection/test_sec_crosswalk.py
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection.sec import crosswalk


def test_cik_overrides_apply_at_build_time():
    fake_data = {
        str(i): {"cik_str": 999_000_000 + i, "ticker": ticker, "title": f"{ticker} Shell Corp"}
        for i, ticker in enumerate(crosswalk.CIK_OVERRIDES)
    }
    fake_data["unaffected"] = {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
    fake_resp = mock.Mock(text=json.dumps(fake_data))
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "crosswalk.parquet"
        with mock.patch.object(crosswalk.http, "get", return_value=fake_resp), \
             mock.patch.object(crosswalk, "CROSSWALK_PATH", tmp_path):
            df = crosswalk.build_crosswalk_tier1()

    for ticker, real_cik in crosswalk.CIK_OVERRIDES.items():
        got = int(df.loc[df["ticker"] == ticker, "cik"].iloc[0])
        assert got == real_cik, f"{ticker} must be overridden to the real filer CIK {real_cik}, got {got}"
    aapl_cik = int(df.loc[df["ticker"] == "AAPL", "cik"].iloc[0])
    assert aapl_cik == 320193, "tickers without an override must keep SEC's original CIK unchanged"
    print(f"OK: all {len(crosswalk.CIK_OVERRIDES)} CIK_OVERRIDES entries apply at build time, survive a refetch")


def test_build_crosswalk_tier1_raises_clear_error_on_fetch_failure():
    # Real bug: http.get() returns None after exhausting retries (see
    # sec/http.py); build_crosswalk_tier1 used to do resp.text unchecked,
    # raising an obscure AttributeError deep inside json.loads instead of a
    # clear error identifying what failed -- this is a hard prerequisite for
    # every ticker in a batch run, so it should fail loudly, not obscurely.
    with mock.patch.object(crosswalk.http, "get", return_value=None):
        try:
            crosswalk.build_crosswalk_tier1()
            assert False, "must raise when the crosswalk fetch fails, not silently produce nothing"
        except RuntimeError as e:
            assert "crosswalk" in str(e).lower()
    print("OK: build_crosswalk_tier1 raises a clear RuntimeError when the SEC fetch fails")


if __name__ == "__main__":
    test_cik_overrides_apply_at_build_time()
    test_build_crosswalk_tier1_raises_clear_error_on_fetch_failure()
