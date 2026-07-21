"""
Test: conviction_model/data.py's window_tensor() normalization/padding and
resample_branch_frame()'s gap-fill. Synthetic frames only -- no dependency on
data/raw or data/processed, stays in the `fast` test group.

Run from project root:
    python tests/conviction_model/test_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.conviction_model.data import (  # noqa: E402
    DAILY_FEATURES, DAILY_WINDOW, MONTHLY_FEATURES, MONTHLY_WINDOW, QUARTERLY_FEATURES,
    QUARTERLY_WINDOW, WEEKLY_FEATURES, WEEKLY_WINDOW, branch_windows_from_frames,
    resample_branch_frame, window_tensor,
)
from test_utils import print_check, print_header, print_section_end  # noqa: E402

TOL = 1e-9


def test_self_normalization_divides_by_anchor_value(passed, failed):
    calendar = pd.bdate_range("2020-01-01", periods=5)
    frame = pd.DataFrame({"adj_close": [10.0, 11.0, 9.0, 12.0, 12.0]}, index=calendar)
    out = window_tensor(frame, calendar[-1], window=5, features=("adj_close",))
    expected = frame["adj_close"].to_numpy() / 12.0
    ok = np.allclose(out[0], expected, atol=TOL)
    print_check("window_tensor: 'self'-normalized channel divides by the anchor (as_of) value",
                ok, f"got {out[0]}, expected {expected}")
    return passed + ok, failed + (not ok)


def test_padding_when_insufficient_history(passed, failed):
    calendar = pd.bdate_range("2020-01-01", periods=4)
    frame = pd.DataFrame({"adj_close": [8.0, 9.0, 10.0, 10.0],
                           "return_1m": [0.01, 0.02, 0.03, 0.04]}, index=calendar)
    out = window_tensor(frame, calendar[-1], window=10, features=("adj_close", "return_1m"))
    price_pad_ok = (bool(np.all(out[0, :6] == 1.0))
                     and np.allclose(out[0, 6:], [0.8, 0.9, 1.0, 1.0], atol=TOL))
    tech_pad_ok = bool(np.all(out[1, :6] == 0.0)) and np.allclose(out[1, 6:], [0.01, 0.02, 0.03, 0.04], atol=TOL)
    ok = price_pad_ok and tech_pad_ok
    print_check("window_tensor: left-pads short history -- 1.0 for 'self' channels, 0.0 for others",
                ok, f"price row={out[0]}, technical row={out[1]}")
    return passed + ok, failed + (not ok)


def test_log1p_squash_is_sign_preserving(passed, failed):
    calendar = pd.bdate_range("2020-01-01", periods=3)
    frame = pd.DataFrame({"pl_zhist_5y": [-50.0, 0.0, 50.0]}, index=calendar)
    out = window_tensor(frame, calendar[-1], window=3, features=("pl_zhist_5y",))
    expected = np.sign([-50.0, 0.0, 50.0]) * np.log1p(np.abs([-50.0, 0.0, 50.0]))
    ok = np.allclose(out[0], expected, atol=TOL)
    print_check("window_tensor: 'log1p' channel is a sign-preserving squash, not a raw log",
                ok, f"got {out[0]}, expected {expected}")
    return passed + ok, failed + (not ok)


def test_internal_nan_neutralized_not_left_as_nan(passed, failed):
    calendar = pd.bdate_range("2020-01-01", periods=4)
    frame = pd.DataFrame({"adj_close": [10.0, np.nan, 10.0, 10.0],
                           "return_1m": [0.01, np.nan, 0.03, 0.04]}, index=calendar)
    out = window_tensor(frame, calendar[-1], window=4, features=("adj_close", "return_1m"))
    ok = bool(np.isfinite(out).all()) and out[0, 1] == 1.0 and out[1, 1] == 0.0
    print_check("window_tensor: an in-window NaN (not just warm-up padding) is neutralized per channel",
                ok, f"price row={out[0]}, technical row={out[1]}")
    return passed + ok, failed + (not ok)


def test_resample_ffills_a_no_trading_period(passed, failed):
    # two trading days, then a 10-day gap (no rows at all in between) --
    # a plain resample('W').last() would leave that week's row NaN.
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-20"])
    frame = pd.DataFrame({"adj_close": [10.0, 11.0, 15.0]}, index=dates)
    out = resample_branch_frame(frame, "W")
    ok = bool(out["adj_close"].notna().all())
    print_check("resample_branch_frame: ffills a period with zero underlying trading rows",
                ok, f"resampled={out['adj_close'].tolist()}")
    return passed + ok, failed + (not ok)


def test_window_tensor_pads_fully_when_as_of_predates_all_rows(passed, failed):
    # a resampled weekly/monthly frame's first row can sit AFTER as_of (e.g.
    # a ticker's first month hasn't closed yet) -- zero rows at/before as_of,
    # not "wrong ticker": should fully pad, not raise (regression test for
    # the padded[-n_have:] slice bug at n_have=0, and the too-eager
    # hist.empty raise it was hiding behind).
    calendar = pd.bdate_range("2020-02-01", periods=3)
    frame = pd.DataFrame({"adj_close": [10.0, 11.0, 12.0],
                           "return_1m": [0.01, 0.02, 0.03]}, index=calendar)
    as_of = pd.Timestamp("2020-01-15")  # before the frame's first row
    out = window_tensor(frame, as_of, window=5, features=("adj_close", "return_1m"))
    ok = bool(np.all(out[0] == 1.0)) and bool(np.all(out[1] == 0.0))
    print_check("window_tensor: as_of before all rows fully left-pads instead of raising",
                ok, f"price row={out[0]}, technical row={out[1]}")
    return passed + ok, failed + (not ok)


def test_window_tensor_raises_on_completely_empty_frame(passed, failed):
    frame = pd.DataFrame({"adj_close": pd.Series(dtype=float)})
    try:
        window_tensor(frame, pd.Timestamp("2020-01-01"), window=5, features=("adj_close",))
        ok = False
    except ValueError:
        ok = True
    print_check("window_tensor: a genuinely empty frame (no rows at all) still raises", ok)
    return passed + ok, failed + (not ok)


def test_branch_windows_from_frames_shapes(passed, failed):
    rng = np.random.default_rng(0)
    daily_cols = list(dict.fromkeys(DAILY_FEATURES + WEEKLY_FEATURES + MONTHLY_FEATURES))
    calendar = pd.bdate_range("2010-01-01", periods=400)
    daily_frame = pd.DataFrame(rng.normal(size=(400, len(daily_cols))), index=calendar, columns=daily_cols)
    quarters = pd.bdate_range("2010-01-01", periods=20, freq="QE")
    quarterly_frame = pd.DataFrame(rng.normal(size=(20, len(QUARTERLY_FEATURES))),
                                    index=quarters, columns=list(QUARTERLY_FEATURES))

    windows = branch_windows_from_frames(daily_frame, quarterly_frame, calendar[-1])
    shapes_ok = (windows["daily"].shape == (len(DAILY_FEATURES), DAILY_WINDOW)
                 and windows["weekly"].shape == (len(WEEKLY_FEATURES), WEEKLY_WINDOW)
                 and windows["monthly"].shape == (len(MONTHLY_FEATURES), MONTHLY_WINDOW)
                 and windows["fundamentals"].shape == (len(QUARTERLY_FEATURES), QUARTERLY_WINDOW))
    finite_ok = all(np.isfinite(v).all() for v in windows.values())
    ok = shapes_ok and finite_ok
    print_check("branch_windows_from_frames: all 4 branches present with the right [features, window] shapes",
                ok, f"shapes={{k: v.shape for k, v in windows.items()}}" if not shapes_ok else "ok")
    return passed + ok, failed + (not ok)


def main() -> int:
    print_header("conviction_model/data.py")
    passed = failed = 0
    for test_fn in [
        test_self_normalization_divides_by_anchor_value,
        test_padding_when_insufficient_history,
        test_log1p_squash_is_sign_preserving,
        test_internal_nan_neutralized_not_left_as_nan,
        test_resample_ffills_a_no_trading_period,
        test_window_tensor_pads_fully_when_as_of_predates_all_rows,
        test_window_tensor_raises_on_completely_empty_frame,
        test_branch_windows_from_frames_shapes,
    ]:
        passed, failed = test_fn(passed, failed)
    print_section_end(passed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
