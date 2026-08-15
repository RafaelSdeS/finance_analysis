"""
test_client_fail_fast.py
=========================
client.py's retry loops used to sleep+log "retry in Xs" before raising even
when no further attempt was going to be made -- wasted latency on every
transient BolsAI/BCB error, contradicting both the log message and the
module's own "retries with exponential backoff" docstring. Verifies get_json
only sleeps when another attempt genuinely remains.

Both cases patch config.MAX_RETRIES locally on purpose: this tests the retry
LOOP's behaviour at any setting, independent of the shipped default (raised
1 -> 3 on 2026-08-15, since at 1 the backoff branch was unreachable and a
single transient BolsAI error permanently skip-listed a real ticker).

Usage:
    python tests/data_collection/test_client_fail_fast.py
"""

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection import client, config


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class _FakeClient:
    """Always returns a retryable 500 -- exercises the retry-exhaustion path."""

    def __init__(self):
        self.calls = 0

    def get(self, path, params=None):
        self.calls += 1
        return _FakeResponse(500)


def test_no_sleep_when_out_of_retries():
    fake = _FakeClient()
    with mock.patch.object(config, "MAX_RETRIES", 1), \
         mock.patch.object(client.time, "sleep") as mock_sleep:
        try:
            client.get_json(fake, "/x")
            assert False, "must raise once the single attempt fails"
        except RuntimeError:
            pass
    assert fake.calls == 1
    mock_sleep.assert_not_called()
    print("OK: MAX_RETRIES=1 fails immediately, no wasted sleep")


def test_sleeps_and_retries_when_another_attempt_remains():
    fake = _FakeClient()
    with mock.patch.object(config, "MAX_RETRIES", 2), \
         mock.patch.object(client.time, "sleep") as mock_sleep:
        try:
            client.get_json(fake, "/x")
            assert False, "must raise once both attempts fail"
        except RuntimeError:
            pass
    assert fake.calls == 2, f"expected 2 attempts, got {fake.calls}"
    mock_sleep.assert_called_once()
    print("OK: MAX_RETRIES=2 still sleeps once between the two attempts")


if __name__ == "__main__":
    test_no_sleep_when_out_of_retries()
    test_sleeps_and_retries_when_another_attempt_remains()
