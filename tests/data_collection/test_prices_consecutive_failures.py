"""
test_prices_consecutive_failures.py
=====================================
Self-check for collect_prices_yf's consecutive-failure guard (no network;
mocks every internal that touches yfinance/disk).

Real incident, found live (2026-07-29): a laptop suspend/resume left the
prices collection process with a stale yfinance session. Every subsequent
ticker failed, each one individually logged as "no yfinance coverage" --
manually verified afterward that a fresh process fetched all of them
instantly and correctly. Nothing in the code would have caught this on its
own; it would have silently mislabeled the rest of a 10,432-ticker run.
MAX_CONSECUTIVE_FAILURES aborts loudly instead once a failure streak gets
implausibly long for a genuine coverage gap.

Usage: python tests/data_collection/test_prices_consecutive_failures.py
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_collection import yf_collectors as yfc


def _run(tickers, fetch_results, price_dir, skip_existing=False):
    """fetch_results: dict ticker -> return value for _fetch_and_shape_prices
    (None = no coverage/failure, a DataFrame = success)."""
    with mock.patch.object(yfc, "checkpoint") as mock_cp, \
         mock.patch.object(yfc, "_prices_fetch_start", return_value="2020-01-01"), \
         mock.patch.object(yfc, "_bolsai_junction_date", return_value=None), \
         mock.patch.object(yfc, "_reconcile_yfinance_junction", side_effect=lambda t, p, df, j: df), \
         mock.patch.object(yfc, "_fetch_and_shape_prices", side_effect=lambda t, *a, **k: fetch_results[t]), \
         mock.patch.object(yfc, "_merge_save", side_effect=lambda df, *a, **k: df), \
         mock.patch.object(yfc, "sleep"):
        mock_cp.load.return_value = {}
        yfc.collect_prices_yf(tickers, mode="test", price_dir=price_dir, suffix="", floor="1900-01-01",
                              skip_existing=skip_existing)


def _fake_price_df():
    return pd.DataFrame({"trade_date": pd.to_datetime(["2026-01-01"]), "close": [1.0]})


def test_long_failure_streak_aborts_loudly():
    tickers = [f"T{i}" for i in range(yfc.MAX_CONSECUTIVE_FAILURES + 10)]
    fetch_results = {t: None for t in tickers}  # every ticker "fails"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _run(tickers, fetch_results, Path(tmp))
            assert False, "must raise once the consecutive-failure streak exceeds the threshold"
        except RuntimeError as e:
            assert "consecutive" in str(e).lower()
            assert str(yfc.MAX_CONSECUTIVE_FAILURES) in str(e)
    print("OK: a long consecutive-failure streak aborts loudly instead of running to completion")


def test_occasional_failures_interspersed_with_successes_do_not_abort():
    # Real coverage gaps DO cluster (OTC/foreign/shell tickers grouped in the crosswalk's
    # roughly market-cap-sorted order) -- must not false-positive on a normal run just
    # because SOME tickers along the way have no coverage.
    n = yfc.MAX_CONSECUTIVE_FAILURES * 3
    tickers = [f"T{i}" for i in range(n)]
    fetch_results = {t: (None if i % 5 == 0 else _fake_price_df()) for i, t in enumerate(tickers)}
    with tempfile.TemporaryDirectory() as tmp:
        _run(tickers, fetch_results, Path(tmp))  # must not raise
    print("OK: failures interspersed with real successes (streak never gets long) don't abort")


def test_success_resets_the_streak_just_under_threshold():
    # Exactly (threshold - 1) failures, then one success, then (threshold - 1) more failures --
    # must NOT abort, since the streak resets and never actually reaches the threshold.
    half = yfc.MAX_CONSECUTIVE_FAILURES - 1
    tickers = [f"A{i}" for i in range(half)] + ["GOOD"] + [f"B{i}" for i in range(half)]
    fetch_results = {t: None for t in tickers}
    fetch_results["GOOD"] = _fake_price_df()
    with tempfile.TemporaryDirectory() as tmp:
        _run(tickers, fetch_results, Path(tmp))  # must not raise
    print("OK: a single success resets the streak, so two near-threshold runs don't combine")


def test_resume_mode_tolerates_a_streak_past_the_normal_threshold():
    # Real bug, found live (2026-07-30): skip_existing=True permanently skips every
    # ticker that already succeeded, so each successive resume pass draws its
    # "still to fetch" pool from an increasingly concentrated remainder of exactly
    # the tickers that failed last time -- a streak well past the normal (non-resume)
    # threshold is EXPECTED here, not a sign of a stale connection (individually
    # verified live: a fresh session fetched AAPL/MSFT instantly right after a 40-long
    # streak of genuinely-uncoverable tickers). A streak strictly between the two
    # thresholds must NOT abort in resume mode, though it would in normal mode.
    n = yfc.MAX_CONSECUTIVE_FAILURES + 20
    assert n < yfc.MAX_CONSECUTIVE_FAILURES_RESUME, "test assumes the resume threshold is much looser"
    tickers = [f"T{i}" for i in range(n)]
    fetch_results = {t: None for t in tickers}
    with tempfile.TemporaryDirectory() as tmp:
        _run(tickers, fetch_results, Path(tmp), skip_existing=True)  # must NOT raise
    print("OK: resume mode tolerates a failure streak that would abort a normal run")


def test_resume_mode_still_aborts_past_its_own_much_higher_threshold():
    tickers = [f"T{i}" for i in range(yfc.MAX_CONSECUTIVE_FAILURES_RESUME + 5)]
    fetch_results = {t: None for t in tickers}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _run(tickers, fetch_results, Path(tmp), skip_existing=True)
            assert False, "even resume mode must eventually abort on a truly catastrophic streak"
        except RuntimeError as e:
            assert str(yfc.MAX_CONSECUTIVE_FAILURES_RESUME) in str(e)
    print("OK: resume mode still has a ceiling, just a much higher one")


if __name__ == "__main__":
    test_long_failure_streak_aborts_loudly()
    test_occasional_failures_interspersed_with_successes_do_not_abort()
    test_success_resets_the_streak_just_under_threshold()
    test_resume_mode_tolerates_a_streak_past_the_normal_threshold()
    test_resume_mode_still_aborts_past_its_own_much_higher_threshold()
