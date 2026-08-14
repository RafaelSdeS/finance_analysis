"""sec/http.py — shared SEC EDGAR download plumbing (full-index, XBRL, bulk data).

SEC requires no API key but asks for a descriptive User-Agent identifying the
requester, and enforces a 10 req/s cap per IP (verified 2026-07-28: 429 on
excess, per SEC's published fair-access policy). One throttled retry-with-
backoff GET; every sec/ module (full-index text, companyfacts JSON, bulk
zips later) goes through this one function, mirroring cvm/http.py's role for
CVM open-data.
"""

import hashlib
import logging
import threading
import time
from pathlib import Path

import requests

from .. import config

log = logging.getLogger("sec")

TIMEOUT = (15, 120)  # (connect, read)
RETRIES = 2
MIN_INTERVAL = 0.12  # SEC's 10 req/s cap -> floor 0.1s; 0.12 leaves margin
BACKOFF_BASE = 2  # seconds; doubles each retry (2s, 4s, ...)

_last_request = 0.0
_lock = threading.Lock()  # callers now run from a ThreadPoolExecutor (fundamentals.py) --
# check-then-set on _last_request isn't atomic, so concurrent threads could both see
# a stale gap and fire together, bursting past the 10 req/s cap this exists to enforce.

# On-disk cache for immutable Archive documents (2026-08-13): a full fundamentals
# rebuild is routine (every derivation fix must reach already-collected companies
# too, per the module-level bug log elsewhere in sec/) and re-downloads identical,
# never-changing documents every time -- a 1997 EX-27 exhibit cannot change. Scoped
# to /Archives/ ONLY: a filed document is permanently immutable, unlike
# data.sec.gov/api/xbrl/companyfacts (restatements land there), which must stay
# live and is never cached. No expiry needed -- this isn't a size/staleness
# tradeoff, Archives content genuinely never changes once filed.
ARCHIVE_CACHE_DIR = config.US_SEC_DIR / "archive_cache"


def _is_archive_url(url: str) -> bool:
    return url.startswith("https://www.sec.gov/Archives/")


def _cache_path(url: str) -> Path:
    return ARCHIVE_CACHE_DIR / f"{hashlib.sha256(url.encode()).hexdigest()}.txt"


class _CachedResponse:
    """Minimal requests.Response stand-in for a cache hit -- every sec/ caller
    only ever reads .text (confirmed: no caller touches .json()/.content/
    .status_code outside this module), so that's all a cache hit needs to supply."""
    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text


def _throttle():
    global _last_request
    with _lock:
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

    Archive URLs are cached on-disk on a successful (200) response, keyed by a
    hash of the URL -- see ARCHIVE_CACHE_DIR above. Deliberately NOT caching a
    404: unlike a genuine "this document doesn't exist" being permanent, this
    function's None return is also reachable after exhausting retries on a
    transient network error, and conflating the two would risk caching a
    temporary SEC-side hiccup as a permanent miss. A cache hit skips
    _throttle() entirely -- that's the actual speedup, not just skipping the
    download.
    """
    cacheable = _is_archive_url(url)
    if cacheable:
        cache_path = _cache_path(url)
        if cache_path.exists():
            return _CachedResponse(cache_path.read_text(encoding="utf-8"))

    for attempt in range(RETRIES + 1):
        _throttle()
        try:
            resp = requests.get(url, headers={"User-Agent": config.SEC_USER_AGENT}, timeout=TIMEOUT)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            if cacheable:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(resp.text, encoding="utf-8")
            return resp
        except requests.RequestException as e:
            if attempt == RETRIES:
                log.warning("%s: network error after %d attempts: %s", url, RETRIES + 1, e)
                return None
            log.warning("%s: %s — retrying (%d/%d)", url, type(e).__name__, attempt + 1, RETRIES)
            # Backoff before the retry, distinct from _throttle's per-request floor:
            # without this, all RETRIES+1 attempts fire ~0.12s apart, so a 429 or a
            # transient 503 usually burns every attempt inside a fraction of a
            # second and gives up -- no real recovery window at all.
            time.sleep(BACKOFF_BASE * 2 ** attempt)
    return None
