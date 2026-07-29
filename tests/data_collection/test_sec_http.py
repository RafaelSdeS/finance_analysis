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


if __name__ == "__main__":
    test_transient_5xx_is_retried_not_raised()
    test_persistent_5xx_gives_up_after_retries()
    test_404_returns_none_without_retry()
    test_retry_backs_off_before_each_attempt()
