#!/usr/bin/env python3
"""
Split repair: repair_unadjusted_splits() rescales adj_* history where a
corporate event (split/inplit) was left unadjusted. Mirrors
src/build_dataset/repair.py.

Only exercised previously via the leak-detection check in
test_final_dataset.py, which runs against the real production dataset and
the real corporate_events.parquet -- a good regression guard for the 53
already-known historical events, but it can't catch a bug in the repair
logic itself (wrong rescale window, wrong direction, a brand-new event)
since it only re-verifies events already baked into current data.

Run from project root: python tests/build_dataset/test_repair.py
or: pytest tests/build_dataset/test_repair.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_dataset import repair


def _prices(ticker, dates, adj_close):
    return pd.DataFrame({
        "ticker": ticker,
        "trade_date": pd.to_datetime(dates),
        "adj_open": adj_close, "adj_high": adj_close,
        "adj_low": adj_close, "adj_close": adj_close,
    })


def _events_file(tmp_path, rows):
    path = tmp_path / "corporate_events.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_repair_rescales_unadjusted_split(tmp_path, monkeypatch) -> None:
    """A 2:1 split left unadjusted: pre-event adj_close is 2x too high relative
    to the post-event scale (a fake ~-69% daily return, ln(0.5), right at the
    event). repair_unadjusted_splits must divide every pre-event row by the
    recorded factor so the series becomes continuous, and leave post-event
    rows untouched."""
    monkeypatch.setattr(repair, "CORPORATE_EVENTS_PATH", _events_file(
        tmp_path, [{"ticker": "TEST3", "date": pd.Timestamp("2026-03-01"), "factor": 2.0}]
    ))

    dates = pd.date_range("2026-02-24", periods=10, freq="D")
    adj_close = [200.0] * 5 + [100.0] * 5  # unadjusted jump right at the split
    prices = _prices("TEST3", dates, adj_close)

    result = repair.repair_unadjusted_splits(prices.copy())

    assert np.allclose(result.loc[:4, "adj_close"], 100.0), "pre-event rows must be rescaled 200 -> 100"
    assert np.allclose(result.loc[5:, "adj_close"], 100.0), "post-event rows must be untouched"
    assert np.allclose(result.loc[:4, "adj_open"], 100.0), "every ADJ_PRICE_COLS column must be rescaled together"


def test_repair_matches_inverse_factor_direction(tmp_path, monkeypatch) -> None:
    """The audit log's factor direction is inconsistent (documented in
    repair.py's own docstring): a recorded factor of 0.5 must repair the same
    2x-style jump as a recorded factor of 2.0 would, since the matching logic
    checks both factor and 1/factor."""
    monkeypatch.setattr(repair, "CORPORATE_EVENTS_PATH", _events_file(
        tmp_path, [{"ticker": "TEST3", "date": pd.Timestamp("2026-03-01"), "factor": 0.5}]
    ))

    dates = pd.date_range("2026-02-24", periods=10, freq="D")
    adj_close = [200.0] * 5 + [100.0] * 5
    prices = _prices("TEST3", dates, adj_close)

    result = repair.repair_unadjusted_splits(prices.copy())

    assert np.allclose(result.loc[:4, "adj_close"], 100.0)


def test_repair_ignores_jump_outside_event_window(tmp_path, monkeypatch) -> None:
    """A jump matching the factor but years away from the recorded event date
    (outside EVENT_WINDOW_DAYS) is left alone -- presumably an unrelated
    market move, not this split."""
    monkeypatch.setattr(repair, "CORPORATE_EVENTS_PATH", _events_file(
        tmp_path, [{"ticker": "TEST3", "date": pd.Timestamp("2020-01-01"), "factor": 2.0}]
    ))

    dates = pd.date_range("2026-02-24", periods=10, freq="D")
    adj_close = [200.0] * 5 + [100.0] * 5
    prices = _prices("TEST3", dates, adj_close)

    result = repair.repair_unadjusted_splits(prices.copy())

    assert np.allclose(result["adj_close"], adj_close)


def test_repair_ignores_event_below_detectable_jump_threshold(tmp_path, monkeypatch) -> None:
    """A recorded event whose |ln(1/factor)| is below MIN_DETECTABLE_JUMP is
    filtered out before matching even starts (can't be told apart from a
    normal market move) -- a real 2x jump in the prices must still be left
    alone, since the recorded event doesn't describe a jump that size."""
    monkeypatch.setattr(repair, "CORPORATE_EVENTS_PATH", _events_file(
        tmp_path, [{"ticker": "TEST3", "date": pd.Timestamp("2026-03-01"), "factor": 1.05}]
    ))  # |ln(1/1.05)| ~= 0.049, well under MIN_DETECTABLE_JUMP (0.3)

    dates = pd.date_range("2026-02-24", periods=10, freq="D")
    adj_close = [200.0] * 5 + [100.0] * 5  # a genuine 2x jump is present regardless
    prices = _prices("TEST3", dates, adj_close)

    result = repair.repair_unadjusted_splits(prices.copy())

    assert np.allclose(result["adj_close"], adj_close)


def test_repair_skips_missing_corporate_events_file(tmp_path, monkeypatch) -> None:
    """No corporate_events.parquet on disk (e.g. a --mode update run that never
    collects it) must be a no-op, not a crash."""
    monkeypatch.setattr(repair, "CORPORATE_EVENTS_PATH", tmp_path / "does_not_exist.parquet")

    prices = _prices("TEST3", pd.date_range("2026-01-01", periods=3), [100.0, 100.0, 100.0])
    result = repair.repair_unadjusted_splits(prices.copy())

    assert np.allclose(result["adj_close"], [100.0, 100.0, 100.0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
