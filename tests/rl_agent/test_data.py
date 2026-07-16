"""
Test: data.py's GlobalAssetIndex, slot calendar, CDI unit validation, and
PricePanel's price_relative / window_tensor math (docs/EIIE_AGENT_PLAN.md
Phase 2). Synthetic data only -- no file I/O, runs anywhere.

Run from project root:
    python tests/rl_agent/test_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.rl_agent.data import (  # noqa: E402
    CASH_GIDX,
    GlobalAssetIndex,
    PricePanel,
    _build_slot_calendar,
    validate_cdi_daily_percent,
)
from test_utils import print_check, print_header, print_section_end  # noqa: E402


def test_global_asset_index(passed, failed):
    membership = pd.DataFrame({
        "period_id": [0, 0, 0],
        "ticker": ["CCC", "AAA", "BBB"],
        "start": pd.to_datetime(["2020-01-01"] * 3),
        "end": pd.to_datetime(["2020-06-01"] * 3),
    })
    idx = GlobalAssetIndex.from_membership(membership)

    ok = idx.tickers == ("AAA", "BBB", "CCC")
    print_check("GlobalAssetIndex: alphabetical order, independent of file row order", ok, str(idx.tickers))
    passed, failed = passed + ok, failed + (not ok)

    ok = idx.ticker_to_gidx == {"AAA": 1, "BBB": 2, "CCC": 3}
    print_check("GlobalAssetIndex: gidx starts at 1 (0 reserved for cash)", ok, str(idx.ticker_to_gidx))
    passed, failed = passed + ok, failed + (not ok)

    ok = idx.n_global == 4 and CASH_GIDX == 0
    print_check("GlobalAssetIndex: n_global = N_union + 1 (cash)", ok, f"n_global={idx.n_global}")
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_cdi_validation(passed, failed):
    # Real anchor values verified against BCB series 12 (docs/EIIE_AGENT_PLAN.md):
    # 2000-01-03 cdi=0.068318 (~18.7% p.a.), 2026-07-14 cdi=0.052531 (~14% p.a.)
    good = np.array([0.068318, 0.052531, 0.045513])
    try:
        validate_cdi_daily_percent(good)
        ok = True
    except AssertionError:
        ok = False
    print_check("validate_cdi_daily_percent: accepts real known-good daily-percent values", ok)
    passed, failed = passed + ok, failed + (not ok)

    # annualized-percent value dropped into the daily field (off by ~200x)
    bad_annualized_in_daily = np.array([14.0])
    try:
        validate_cdi_daily_percent(bad_annualized_in_daily)
        ok = False
    except AssertionError:
        ok = True
    print_check("validate_cdi_daily_percent: rejects an annualized-% value used as daily", ok)
    passed, failed = passed + ok, failed + (not ok)

    # fraction instead of percent (off by 100x, too small)
    bad_fraction = np.array([0.00068318])
    try:
        validate_cdi_daily_percent(bad_fraction)
        ok = False
    except AssertionError:
        ok = True
    print_check("validate_cdi_daily_percent: rejects a fraction used instead of percent", ok)
    passed, failed = passed + ok, failed + (not ok)

    try:
        validate_cdi_daily_percent(np.array([-0.01]))
        ok = False
    except AssertionError:
        ok = True
    print_check("validate_cdi_daily_percent: rejects negative CDI", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_slot_calendar(passed, failed):
    # AAA always qualifies; BBB qualifies only period 0; CCC enters period 1
    # (a universe rotation at the quarter boundary) -- with 2 pre-history
    # calendar days before the first period start.
    membership = pd.DataFrame({
        "period_id": [0, 0, 1, 1],
        "ticker": ["AAA", "BBB", "AAA", "CCC"],
        "start": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-08", "2020-01-08"]),
        "end": pd.to_datetime(["2020-01-08", "2020-01-08", "2020-06-01", "2020-06-01"]),
    })
    asset_index = GlobalAssetIndex.from_membership(membership)  # AAA=1, BBB=2, CCC=3
    calendar = pd.bdate_range("2019-12-30", "2020-01-14")  # first 2 days precede period 0
    slot_gidx, valid = _build_slot_calendar(calendar, membership, asset_index, n_slots=2)

    pre_history = calendar < pd.Timestamp("2020-01-01")
    ok = not valid[pre_history].any()
    print_check("slot calendar: pre-history days (before first period) are all-invalid", ok)
    passed, failed = passed + ok, failed + (not ok)

    day_in_p0 = np.where(calendar == pd.Timestamp("2020-01-02"))[0][0]
    ok = set(slot_gidx[day_in_p0][valid[day_in_p0]].tolist()) == {1, 2}
    print_check("slot calendar: period 0 active members are {AAA, BBB}",
                ok, f"got {slot_gidx[day_in_p0]}, valid={valid[day_in_p0]}")
    passed, failed = passed + ok, failed + (not ok)

    day_in_p1 = np.where(calendar == pd.Timestamp("2020-01-09"))[0][0]
    ok = set(slot_gidx[day_in_p1][valid[day_in_p1]].tolist()) == {1, 3}
    print_check("slot calendar: period 1 rotation -- BBB drops out, CCC enters -- members are {AAA, CCC}",
                ok, f"got {slot_gidx[day_in_p1]}, valid={valid[day_in_p1]}")
    passed, failed = passed + ok, failed + (not ok)

    ok = bool(np.all(slot_gidx[day_in_p0][:2] == np.sort(slot_gidx[day_in_p0][:2])))
    print_check("slot calendar: slots filled sorted ascending by permanent global index", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def _toy_panel(n_slots=2, window=3):
    """Minimal 6-day, 2-asset (+cash) PricePanel for price_relative/window_tensor checks."""
    tickers = ("AAA", "BBB")
    asset_index = GlobalAssetIndex(tickers=tickers, ticker_to_gidx={"AAA": 1, "BBB": 2})
    dates = pd.bdate_range("2020-01-01", periods=6)
    close = np.array([
        [1.0, 10.0, 20.0],
        [1.0, 11.0, 22.0],
        [1.0, 12.0, 24.0],
        [1.0, 13.0, 26.0],
        [1.0, 14.0, 28.0],
        [1.0, 15.0, 30.0],
    ])
    high = close.copy()
    low = close - 1.0
    cdi_factor = np.full(6, 1.0004)
    slot_gidx = np.array([[1, 2]] * 5 + [[1, 2]])
    valid = np.array([[True, True]] * 5 + [[True, False]])  # last day: slot 1 (BBB) masked out
    return PricePanel(
        asset_index=asset_index, dates=dates, close=close, high=high, low=low,
        cdi_factor=cdi_factor, slot_gidx=slot_gidx, valid=valid,
        window=window, start_idx=0, end_idx=5,
    )


def test_price_relative(passed, failed):
    panel = _toy_panel()
    y = panel.price_relative(2)

    ok = np.isclose(y[CASH_GIDX], 1.0004)
    print_check("price_relative: cash index = CDI factor", ok, f"got {y[CASH_GIDX]}")
    passed, failed = passed + ok, failed + (not ok)

    ok = np.isclose(y[1], 12.0 / 11.0) and np.isclose(y[2], 24.0 / 22.0)
    print_check("price_relative: asset ratios = v_t / v_{t-1}", ok, f"got {y[1]:.6f}, {y[2]:.6f}")
    passed, failed = passed + ok, failed + (not ok)

    try:
        panel.price_relative(0)
        ok = False
    except ValueError:
        ok = True
    print_check("price_relative: t=0 raises (no t-1 available)", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_window_tensor(passed, failed):
    panel = _toy_panel(window=3)

    X = panel.window_tensor(2, features=("close",))
    expected_aaa = np.array([10.0, 11.0, 12.0]) / 12.0
    expected_bbb = np.array([20.0, 22.0, 24.0]) / 24.0
    ok = X.shape == (1, 2, 3)
    print_check("window_tensor: shape = (features, n_slots, window)", ok, str(X.shape))
    passed, failed = passed + ok, failed + (not ok)

    ok = np.allclose(X[0, 0], expected_aaa) and np.allclose(X[0, 1], expected_bbb)
    print_check("window_tensor: each slot normalized by its own price at t (last column = 1)",
                ok, f"got {X[0]}")
    passed, failed = passed + ok, failed + (not ok)

    ok = np.isclose(X[0, 0, -1], 1.0) and np.isclose(X[0, 1, -1], 1.0)
    print_check("window_tensor: last column is exactly 1 by construction (eq. 18)", ok)
    passed, failed = passed + ok, failed + (not ok)

    # last day: slot 1 (BBB) is masked -- must be flat-filled to 1.0, not garbage
    X_masked = panel.window_tensor(4, features=("low",))  # index 4 is fine (>= window-1); check slot masking separately at t=5
    ok = X_masked.shape == (1, 2, 3)
    print_check("window_tensor: valid day still produces correct shape", ok)
    passed, failed = passed + ok, failed + (not ok)

    X_last = panel.window_tensor(5, features=("close",))
    ok = np.allclose(X_last[0, 1], 1.0)
    print_check("window_tensor: masked/invalid slot is flat-filled to all-ones (never NaN/garbage)",
                ok, f"got {X_last[0, 1]}")
    passed, failed = passed + ok, failed + (not ok)

    try:
        panel.window_tensor(1, features=("close",))  # window=3 needs t >= 2
        ok = False
    except ValueError:
        ok = True
    print_check("window_tensor: t < window-1 raises (insufficient lookback)", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def main():
    print_header("test_data")
    passed = failed = 0

    passed, failed = test_global_asset_index(passed, failed)
    passed, failed = test_cdi_validation(passed, failed)
    passed, failed = test_slot_calendar(passed, failed)
    passed, failed = test_price_relative(passed, failed)
    passed, failed = test_window_tensor(passed, failed)

    print_section_end(passed, failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
