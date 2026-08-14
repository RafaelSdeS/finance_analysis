"""
test_sec_http.py
=================
Self-check for sec/http.py's retry logic (no network; mocks requests.get).

Real bug, found retrying fundamentals collection at scale (2026-07-28):
raise_for_status() ran AFTER the retry loop had already exited on a
successful (but bad-status) response, so a transient 5xx (confirmed on
FLEX/SNPS: real "503 Service Unavailable" from SEC) raised uncaught on the
FIRST attempt -- no retry at all, unlike a connection error. Nothing
downstream catches this per-filing either, so it crashed the whole CIK's
fundamentals build. Fixed by moving raise_for_status() inside the retry
loop's try block.

Usage: python tests/data_collection/test_sec_http.py
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests

from src.data_collection.sec import http


def _resp(status_code, text="ok"):
    r = requests.Response()
    r.status_code = status_code
    r._content = text.encode()
    return r


def test_transient_5xx_is_retried_not_raised():
    with mock.patch.object(http, "_throttle"), mock.patch.object(http.time, "sleep"), \
         mock.patch("requests.get", side_effect=[_resp(503), _resp(200)]):
        resp = http.get("https://www.sec.gov/whatever")
    assert resp is not None and resp.status_code == 200, (
        "a transient 503 must be retried, not raised uncaught on the first attempt")
    print("OK: a transient 5xx status is retried, not raised immediately")


def test_persistent_5xx_gives_up_after_retries():
    with mock.patch.object(http, "_throttle"), mock.patch.object(http.time, "sleep"), \
         mock.patch("requests.get", side_effect=[_resp(503), _resp(503), _resp(503)]):
        resp = http.get("https://www.sec.gov/whatever")
    assert resp is None, "must give up and return None after exhausting retries, not raise"
    print("OK: a persistent 5xx exhausts retries and returns None rather than raising")


def test_404_returns_none_without_retry():
    with mock.patch.object(http, "_throttle"), \
         mock.patch("requests.get", side_effect=[_resp(404)]) as m:
        resp = http.get("https://www.sec.gov/whatever")
    assert resp is None
    assert m.call_count == 1, "a 404 is an expected outcome (missing quarter/CIK), not worth retrying"
    print("OK: 404 returns None immediately, without retrying")


def test_retry_backs_off_before_each_attempt():
    # Real pothole: retries fired ~0.12s apart (just the per-request throttle
    # floor, no actual backoff), so a 429/503 usually burned every attempt
    # inside a fraction of a second with no real recovery window. Must sleep
    # BACKOFF_BASE * 2**attempt before each retry (not before the final,
    # already-failed attempt).
    with mock.patch.object(http, "_throttle"), mock.patch.object(http.time, "sleep") as fake_sleep, \
         mock.patch("requests.get", side_effect=[_resp(503), _resp(503), _resp(503)]):
        http.get("https://www.sec.gov/whatever")
    assert fake_sleep.call_args_list == [mock.call(http.BACKOFF_BASE * 2**0), mock.call(http.BACKOFF_BASE * 2**1)], (
        f"expected 2 backoff sleeps (after attempts 0 and 1, none after the final attempt), "
        f"got {fake_sleep.call_args_list}")
    print("OK: get() backs off exponentially before each retry, not just the per-request throttle")


def test_archive_url_cached_on_disk_second_call_skips_network_and_throttle():
    with tempfile.TemporaryDirectory() as tmp, \
         mock.patch.object(http, "ARCHIVE_CACHE_DIR", Path(tmp)), \
         mock.patch.object(http, "_throttle") as mock_throttle, \
         mock.patch("requests.get", side_effect=[_resp(200, "the filing body")]) as mock_get:
        url = "https://www.sec.gov/Archives/edgar/data/1234/0001.txt"
        first = http.get(url)
        assert first.text == "the filing body"
        assert mock_get.call_count == 1 and mock_throttle.call_count == 1

        second = http.get(url)  # must be served from disk, no network, no throttle
        assert second.text == "the filing body", f"cache hit must return identical content, got {second.text!r}"
        assert mock_get.call_count == 1, "a cache hit must not issue a second HTTP request"
        assert mock_throttle.call_count == 1, "a cache hit must skip _throttle() entirely -- that's the actual speedup"
    print("OK: an Archives URL is cached on disk; a second get() skips both the request and the throttle")


def test_non_archive_url_never_cached():
    # data.sec.gov/api/xbrl/companyfacts is mutable (restatements land there) and
    # must never be served stale -- confirm a non-/Archives/ URL always re-fetches.
    with tempfile.TemporaryDirectory() as tmp, \
         mock.patch.object(http, "ARCHIVE_CACHE_DIR", Path(tmp)), \
         mock.patch.object(http, "_throttle"), \
         mock.patch("requests.get", side_effect=[_resp(200, "v1"), _resp(200, "v2")]) as mock_get:
        url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
        first = http.get(url)
        second = http.get(url)
        assert first.text == "v1" and second.text == "v2", \
            "a non-Archives URL must re-fetch every call, never serve a cached value"
        assert mock_get.call_count == 2
    print("OK: a non-Archives URL (e.g. companyfacts) is never cached, always re-fetched live")


def test_404_on_archive_url_is_not_cached():
    # A 404 could be a genuine permanent miss OR the exhausted-retries branch of a
    # transient network error -- caching it would risk permanently mislabeling a
    # temporary SEC-side hiccup as "this document doesn't exist."
    with tempfile.TemporaryDirectory() as tmp, \
         mock.patch.object(http, "ARCHIVE_CACHE_DIR", Path(tmp)), \
         mock.patch.object(http, "_throttle"), \
         mock.patch("requests.get", side_effect=[_resp(404), _resp(200, "found on retry")]) as mock_get:
        url = "https://www.sec.gov/Archives/edgar/data/9999/missing.txt"
        first = http.get(url)
        assert first is None
        second = http.get(url)
        assert second is not None and second.text == "found on retry", \
            "a 404 must not be cached -- the next call must hit the network again"
        assert mock_get.call_count == 2
    print("OK: a 404 on an Archives URL is not cached, so a later real fetch isn't blocked by it")


if __name__ == "__main__":
    test_transient_5xx_is_retried_not_raised()
    test_persistent_5xx_gives_up_after_retries()
    test_404_returns_none_without_retry()
    test_retry_backs_off_before_each_attempt()
    test_archive_url_cached_on_disk_second_call_skips_network_and_throttle()
    test_non_archive_url_never_cached()
    test_404_on_archive_url_is_not_cached()
