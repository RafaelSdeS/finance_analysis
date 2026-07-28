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
    is an expected, non-error outcome across ~130 quarters / thousands of CIKs)."""
    resp = None
    for attempt in range(RETRIES + 1):
        _throttle()
        try:
            resp = requests.get(url, headers={"User-Agent": config.SEC_USER_AGENT}, timeout=TIMEOUT)
            break
        except requests.RequestException as e:
            if attempt == RETRIES:
                log.warning("%s: network error after %d attempts: %s", url, RETRIES + 1, e)
                return None
            log.warning("%s: %s — retrying (%d/%d)", url, type(e).__name__, attempt + 1, RETRIES)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp
