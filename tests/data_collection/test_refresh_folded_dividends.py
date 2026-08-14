"""
test_refresh_folded_dividends.py
==================================
Self-check for refresh.py's _refresh_prices_and_dividends orchestration
(no network; mocks collect_prices_yf/collect_dividends_yf at the module
level -- this tests WHICH calls happen with WHAT args, not the underlying
collectors themselves, which have their own self-checks).

Before 2026-08-13, refresh.py ran collect_dividends_yf THEN collect_prices_yf
as two always-separate passes. Now dividends detection+write is folded into
the price fetch (collect_prices_yf's collect_dividends=True) -- this file
locks in the branching that preserves every --only combination's old
behavior while landing the folded fast path for the common case.

Usage: python tests/data_collection/test_refresh_folded_dividends.py
"""

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection import refresh


def test_default_stages_two_pass_when_something_changed():
    calls = []

    def fake_collect_prices_yf(tickers, mode, workers=1, full_refetch=None, collect_dividends=False, **kw):
        calls.append({"tickers": tuple(tickers), "full_refetch": full_refetch, "collect_dividends": collect_dividends})
        return {"AAPL"} if full_refetch == set() else set()

    with mock.patch.object(refresh, "collect_prices_yf", side_effect=fake_collect_prices_yf), \
         mock.patch.object(refresh, "collect_dividends_yf") as mock_divs:
        refresh._refresh_prices_and_dividends(["AAPL", "MSFT"], "test", {"prices", "dividends"},
                                                full=False, workers=1)

    mock_divs.assert_not_called()
    assert len(calls) == 2, f"expected pass 1 (tail probe) + pass 2 (full re-fetch for changed), got {calls}"
    assert calls[0]["full_refetch"] == set() and calls[0]["collect_dividends"] is True, \
        f"pass 1 must be tail-only-for-everyone with dividends folded in, got {calls[0]}"
    assert calls[1]["tickers"] == ("AAPL",) and calls[1]["full_refetch"] == {"AAPL"}, \
        f"pass 2 must re-fetch ONLY the changed subset in full, got {calls[1]}"
    print("OK: default stages run pass 1 (tail, folded dividends) + pass 2 (full re-fetch) when something changed")


def test_default_stages_single_pass_when_nothing_changed():
    calls = []

    def fake_collect_prices_yf(tickers, mode, workers=1, full_refetch=None, collect_dividends=False, **kw):
        calls.append(full_refetch)
        return set()  # nothing changed

    with mock.patch.object(refresh, "collect_prices_yf", side_effect=fake_collect_prices_yf), \
         mock.patch.object(refresh, "collect_dividends_yf") as mock_divs:
        refresh._refresh_prices_and_dividends(["AAPL", "MSFT"], "test", {"prices", "dividends"},
                                                full=False, workers=1)

    mock_divs.assert_not_called()
    assert len(calls) == 1, f"an empty changed set must skip pass 2 entirely (no second request), got {len(calls)} call(s)"
    print("OK: no second pass fires when nothing changed (the common, optimized case)")


def test_only_prices_folds_no_dividends():
    with mock.patch.object(refresh, "collect_prices_yf", return_value=set()) as mock_prices, \
         mock.patch.object(refresh, "collect_dividends_yf") as mock_divs:
        refresh._refresh_prices_and_dividends(["AAPL"], "test", {"prices"}, full=False, workers=1)

    mock_divs.assert_not_called()
    assert mock_prices.call_args.kwargs["collect_dividends"] is False, \
        "--only prices (no dividends stage) must not fold dividends extraction in"
    print("OK: --only prices runs a plain tail-only price pass, no dividends folded in")


def test_only_dividends_falls_back_to_standalone_collector():
    with mock.patch.object(refresh, "collect_prices_yf") as mock_prices, \
         mock.patch.object(refresh, "collect_dividends_yf") as mock_divs:
        refresh._refresh_prices_and_dividends(["AAPL"], "test", {"dividends"}, full=False, workers=1)

    mock_prices.assert_not_called()
    mock_divs.assert_called_once()
    print("OK: --only dividends (no prices stage) falls back to the standalone collect_dividends_yf")


def test_full_flag_skips_two_pass_split():
    with mock.patch.object(refresh, "collect_prices_yf", return_value=set()) as mock_prices, \
         mock.patch.object(refresh, "collect_dividends_yf") as mock_divs:
        refresh._refresh_prices_and_dividends(["AAPL"], "test", {"prices", "dividends"}, full=True, workers=1)

    mock_divs.assert_not_called()
    assert mock_prices.call_count == 1, f"--full must be ONE full-span pass, not the two-pass detect/re-fetch split, got {mock_prices.call_count}"
    assert mock_prices.call_args.kwargs.get("full_refetch") is None, \
        "--full must pass full_refetch=None (full-span for every ticker), not a tail-only probe"
    assert mock_prices.call_args.kwargs["collect_dividends"] is True
    print("OK: --full does one full-span pass with dividends folded in, skipping the two-pass split")


if __name__ == "__main__":
    test_default_stages_two_pass_when_something_changed()
    test_default_stages_single_pass_when_nothing_changed()
    test_only_prices_folds_no_dividends()
    test_only_dividends_falls_back_to_standalone_collector()
    test_full_flag_skips_two_pass_split()
