"""sec/http.py — shared SEC EDGAR download plumbing (full-index, XBRL, bulk data).

SEC requires no API key but asks for a descriptive User-Agent identifying the
requester, and enforces a 10 req/s cap per IP (verified 2026-07-28: 429 on
excess, per SEC's published fair-access policy). One throttled retry-with-
backoff GET; every sec/ module (full-index text, companyfacts JSON, bulk
zips later) goes through this one function, mirroring cvm/http.py's role for
CVM open-data.
"""

import logging
import time

import requests

from .. import config

log = logging.getLogger("sec")

TIMEOUT = (15, 120)  # (connect, read)
RETRIES = 2
MIN_INTERVAL = 0.12  # SEC's 10 req/s cap -> floor 0.1s; 0.12 leaves margin

_last_request = 0.0


def _throttle():
    global _last_request
    wait = MIN_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def get(url: str) -> requests.Response | None:
    """One throttled GET with retry. Returns None on 404 (a missing quarter/CIK
    is an expected, non-error outcome across ~130 quarters / thousands of CIKs).

    Real bug, found retrying fundamentals collection at scale (2026-07-28):
    raise_for_status() used to run AFTER the retry loop, so a transient 5xx
    (confirmed on FLEX/SNPS: real "503 Service Unavailable" from SEC) raised
    uncaught on the FIRST attempt, with no retry at all -- unlike a connection
    error, which does get retried. Since nothing downstream catches this per-
    filing either, it crashed the CIK's entire fundamentals build. Moved
    inside the try block so a bad status code gets the same retry-with-
    backoff treatment as a connection failure (HTTPError is itself a
    RequestException subclass).
    """
    for attempt in range(RETRIES + 1):
        _throttle()
        try:
            resp = requests.get(url, headers={"User-Agent": config.SEC_USER_AGENT}, timeout=TIMEOUT)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == RETRIES:
                log.warning("%s: network error after %d attempts: %s", url, RETRIES + 1, e)
                return None
            log.warning("%s: %s — retrying (%d/%d)", url, type(e).__name__, attempt + 1, RETRIES)
    return None
