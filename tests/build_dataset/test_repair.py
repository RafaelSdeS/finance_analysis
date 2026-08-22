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


def _prices(ticker, dates, adj_close, volume=None):
    df = pd.DataFrame({
        "ticker": ticker,
        "trade_date": pd.to_datetime(dates),
        "adj_open": adj_close, "adj_high": adj_close,
        "adj_low": adj_close, "adj_close": adj_close,
    })
    if volume is not None:
        df["volume"] = volume
        df["volume_adjusted"] = volume
    return df


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


def test_repair_rescales_volume_opposite_direction_from_price(tmp_path, monkeypatch) -> None:
    """A 1:4 split (factor=4) must divide pre-event price by 4 AND multiply
    pre-event volume/volume_adjusted by 4 -- same shares-outstanding logic as
    a real split (more shares, same dollar activity), and the same
    dollar-volume-invariant convention continuity.py's merger-ratio scaling
    uses. Scaling volume the same direction as price (dividing both) would
    silently double the discontinuity in turnover_ratio/volume_ratio_20d
    instead of removing it."""
    monkeypatch.setattr(repair, "CORPORATE_EVENTS_PATH", _events_file(
        tmp_path, [{"ticker": "TEST3", "date": pd.Timestamp("2026-03-01"), "factor": 4.0}]
    ))

    dates = pd.date_range("2026-02-20", periods=10, freq="D")
    adj_close = [400.0] * 5 + [100.0] * 5  # unadjusted 1:4 split
    volume = [10_000] * 5 + [40_000] * 5   # post-split volume already at new-share scale
    prices = _prices("TEST3", dates, adj_close, volume=volume)

    result = repair.repair_unadjusted_splits(prices.copy())

    assert np.allclose(result.loc[:4, "adj_close"], 100.0)
    assert np.allclose(result.loc[:4, "volume"], 40_000), (
        "pre-event volume must be MULTIPLIED by factor (more new-share-equivalent "
        "shares traded), matching the price division, so volume*price is invariant "
        "across the splice"
    )
    assert np.allclose(result.loc[:4, "volume_adjusted"], 40_000)
    assert np.allclose(result.loc[5:, "volume"], 40_000), "post-event volume untouched"


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


def _price_series(ticker, dates, close, adj_close):
    return pd.DataFrame({
        "ticker": ticker,
        "trade_date": pd.to_datetime(dates),
        "open": close, "high": close, "low": close, "close": close,
        "adj_open": adj_close, "adj_high": adj_close, "adj_low": adj_close, "adj_close": adj_close,
    })


def test_repair_rescales_raw_ohlc_for_flagged_tickers(tmp_path, monkeypatch) -> None:
    """RAW_OHLC_ALSO_UNADJUSTED tickers (currently just TIMS3) need their plain
    open/high/low/close rescaled the same way as adj_*, because the vendor's
    "unadjusted" close is itself on the wrong scale for that ticker -- not
    the normal case, where raw close deliberately keeps showing the real
    historical jump. An unflagged ticker with the exact same shape must have
    its raw OHLC left alone."""
    monkeypatch.setattr(repair, "CORPORATE_EVENTS_PATH", _events_file(
        tmp_path, [{"ticker": "TIMS3", "date": pd.Timestamp("2026-03-01"), "factor": 100.0},
                   {"ticker": "TEST3", "date": pd.Timestamp("2026-03-01"), "factor": 100.0}]
    ))

    dates = pd.date_range("2026-02-15", periods=10, freq="D")
    close = [200.0] * 5 + [2.0] * 5  # unadjusted 100:1 jump, both raw and adj_*
    prices = pd.concat([
        _price_series("TIMS3", dates, close, close),
        _price_series("TEST3", dates, close, close),
    ], ignore_index=True)

    result = repair.repair_unadjusted_splits(prices.copy())

    tims3 = result[result["ticker"] == "TIMS3"].reset_index(drop=True)
    assert np.allclose(tims3.loc[:4, "close"], 2.0), "TIMS3's raw close must be rescaled like adj_close"
    assert np.allclose(tims3.loc[:4, "open"], 2.0), "every RAW_PRICE_COLS column must be rescaled together"
    assert np.allclose(tims3.loc[5:, "close"], 2.0), "post-event rows untouched"

    test3 = result[result["ticker"] == "TEST3"].reset_index(drop=True)
    assert np.allclose(test3.loc[:4, "adj_close"], 2.0), "adj_close is still repaired for an unflagged ticker"
    assert np.allclose(test3.loc[:4, "close"], 200.0), "but its raw close must be left alone"


def test_glitch_repair_snaps_back_isolated_single_day_pulse() -> None:
    """PETR4/ITUB4/SBSP3/... shape (docs/DATA_LAYER_FOLLOWUP_FINDINGS.md): a
    single day's adj_close is corrupted -- ratio (adj_close/close) jumps hard
    on day D and reverts back to the pre-glitch level by day D+1, while close
    barely moves. Must repair only that one row, holding the prior day's
    ratio, and leave every other row untouched."""
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    close = [10.0] * 6
    adj_close = [5.0, 5.0, 1.5, 5.0, 5.0, 5.0]  # steady ratio 0.5 except the glitch day
    prices = _price_series("TEST3", dates, close, adj_close)

    result = repair.repair_isolated_adj_close_glitches(prices.copy())

    assert np.allclose(result["adj_close"], [5.0] * 6), result["adj_close"].tolist()
    assert np.allclose(result["adj_open"], [5.0] * 6)


def test_glitch_repair_ignores_ratio_pulse_caused_by_bad_close() -> None:
    """AFLT3's real shape (docs/DATA_LAYER_FOLLOWUP_FINDINGS.md): a single
    day's RAW close is the one that's wrong (not adj_close) -- close dips
    and bounces back while adj_close is already correctly, permanently
    adjusted (e.g. right after repair_unadjusted_splits fixed a real split
    that raw close's own error happened to land next to). The ratio still
    pulses-and-reverts on that day, but adj_close is the side that's right
    here -- "fixing" it would reintroduce a discontinuity that was already
    removed. Must be left alone."""
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    close = [18.0, 18.0, 0.018, 18.0, 18.0, 18.0]     # bad raw close for one day
    adj_close = [6.35] * 6                             # already correctly adjusted, stable
    prices = _price_series("TEST3", dates, close, adj_close)

    result = repair.repair_isolated_adj_close_glitches(prices.copy())

    assert np.allclose(result["adj_close"], adj_close), "a bad raw close must not corrupt a correct adj_close"


def test_glitch_repair_leaves_real_split_alone() -> None:
    """A real split changes the ratio PERMANENTLY -- day D+1 does NOT revert
    to the pre-event level, so this must never be treated as a glitch."""
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    close = [10.0] * 6
    adj_close = [5.0, 5.0, 2.5, 2.5, 2.5, 2.5]  # ratio steps down and stays
    prices = _price_series("TEST3", dates, close, adj_close)

    result = repair.repair_isolated_adj_close_glitches(prices.copy())

    assert np.allclose(result["adj_close"], adj_close), "a genuine permanent shift must be left alone"


def test_glitch_repair_leaves_real_price_moves_alone() -> None:
    """A real, volatile price move (even a sharp round trip) keeps ratio flat
    because close and adj_close move together -- must never false-positive."""
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    close = [10.0, 10.0, 3.0, 10.0, 10.0, 10.0]  # crash and recovery
    adj_close = [5.0, 5.0, 1.5, 5.0, 5.0, 5.0]   # same ratio (0.5) throughout
    prices = _price_series("TEST3", dates, close, adj_close)

    result = repair.repair_isolated_adj_close_glitches(prices.copy())

    assert np.allclose(result["adj_close"], adj_close), "a real price move with flat ratio must be left alone"


def test_glitch_repair_handles_short_series() -> None:
    """Fewer than 3 rows: nothing to compare against, must not crash."""
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    prices = _price_series("TEST3", dates, [10.0, 10.0], [5.0, 5.0])

    result = repair.repair_isolated_adj_close_glitches(prices.copy())

    assert np.allclose(result["adj_close"], [5.0, 5.0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
