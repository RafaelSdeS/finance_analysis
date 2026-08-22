"""
test_validate_price_jump.py
============================
Regression guard for validate.py's MAX_PLAUSIBLE_DAILY_MOVE jump-warning check
(828ea41, "feat: add implausible daily price-move validation check"). Market-
agnostic: validate_prices() is the exact function test_br_data_quality.py and
test_us_data_quality.py both call over real on-disk data. Those two sweeps only
assert the warning RATE stays under a measured real-data ceiling -- if the
warning logic itself went dead (a mistyped threshold, an inverted comparison),
both would keep passing forever, since "0 warnings" reads as clean either way.
This synthesizes a known-implausible jump directly, so the warning path is
proven to actually fire, independent of whatever happens to be on disk --
covers the gap for both BR and US, which share this one validator.

Run from project root:
    python tests/data_collection/test_validate_price_jump.py
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection import validate  # noqa: E402


def _price_df(adj_close: list[float]) -> pd.DataFrame:
    """Raw OHLC held flat (never implausible); adj_high/adj_low bracket
    adj_close per-row so only the day-over-day RATIO check can fire -- an
    OHLC-bracket violation would otherwise mask the jump check as a hard
    error instead of the warning this test targets."""
    n = len(adj_close)
    close = pd.Series(adj_close, dtype=float)
    return pd.DataFrame({
        "ticker": ["TEST3"] * n,
        "trade_date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open": [10.0] * n, "high": [10.5] * n, "low": [9.5] * n, "close": [10.0] * n,
        "volume": [1000] * n, "volume_adjusted": [1000] * n,
        "traded_amount": [10_000.0] * n, "num_trades": [100] * n,
        "adj_open": close, "adj_high": close * 1.01, "adj_low": close * 0.99, "adj_close": close,
    })


def test_implausible_jump_triggers_warning():
    df = _price_df([10.0, 10.1, 1010.0, 1015.0, 1020.0])  # ~100x jump on day 3
    r = validate.validate_prices(df)
    assert r.passed, "an implausible jump is a WARN, not a hard error"
    assert any(w.startswith(f"1 day(s) with adj_close moving >{validate.MAX_PLAUSIBLE_DAILY_MOVE}x")
               for w in r.warnings), r.warnings


def test_normal_series_has_no_jump_warning():
    df = _price_df([10.0, 10.1, 10.2, 10.3, 10.4])
    r = validate.validate_prices(df)
    assert r.passed
    assert not any("adj_close moving >" in w for w in r.warnings), r.warnings


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
