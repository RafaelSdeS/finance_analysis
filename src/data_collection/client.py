"""
client.py — resilient HTTP helpers shared by all collectors.

Retries with exponential backoff on 429/5xx and network errors; fails fast on
4xx client errors. Plain functions, no class — there's one client config.
"""

import logging
import time

import httpx

from . import config

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def make_client(base_url: str, api_key: str | None = None) -> httpx.Client:
    headers = {"X-API-Key": api_key} if api_key else {}
    return httpx.Client(
        base_url=base_url,
        headers=headers,
        timeout=config.HTTP_TIMEOUT,
        follow_redirects=True,
    )


def get_json(client: httpx.Client, path: str, params: dict | None = None) -> dict:
    """GET with backoff retry. Returns parsed JSON or raises after max retries."""
    last_err = None
    for attempt in range(config.MAX_RETRIES):
        try:
            r = client.get(path, params=params or {})
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            last_err = e
            wait = min(config.BACKOFF_BASE * 2 ** attempt, config.BACKOFF_MAX)
            log.warning("%s: network error (%s), retry in %ds", path, e, wait)
            time.sleep(wait)
            continue

        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                # BCB intermittently returns an empty 200 body — transient, retry
                last_err = "empty/non-JSON 200 body"
                wait = min(config.BACKOFF_BASE * 2 ** attempt, config.BACKOFF_MAX)
                log.warning("%s: %s, retry in %ds", path, last_err, wait)
                time.sleep(wait)
                continue

        if r.status_code in RETRYABLE_STATUS:
            wait = min(config.BACKOFF_BASE * 2 ** attempt, config.BACKOFF_MAX)
            log.warning("%s: HTTP %d, retry in %ds", path, r.status_code, wait)
            time.sleep(wait)
            continue

        # 4xx (other than 429): client error, don't retry
        r.raise_for_status()

    raise RuntimeError(f"max retries exceeded for {path}: {last_err}")
