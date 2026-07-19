"""
Test: h_series/spine.py's decision grid, expanding folds, and forward
relative-return target construction. Deterministic synthetic price data.

Run from project root:
    python tests/h_series/test_spine.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.h_series.spine import (  # noqa: E402
    active_universe_by_date, build_forward_targets, hac_lag_for_horizon,
    iter_expanding_folds, k_trading_days_later, monthly_decision_dates,
)
from test_utils import print_check, print_header, print_section_end  # noqa: E402

TOL = 1e-6


def test_hac_lag_for_horizon(passed, failed):
    ok = (hac_lag_for_horizon(21) == 0 and hac_lag_for_horizon(63) == 2 and hac_lag_for_horizon(1) == 0)
    print_check("hac_lag_for_horizon: k=21 -> lag 0, k=63 -> lag 2 (round(k/21)-1)",
                ok, f"21->{hac_lag_for_horizon(21)}, 63->{hac_lag_for_horizon(63)}")
    return passed + ok, failed + (not ok)


def test_iter_expanding_folds_boundaries(passed, failed):
    folds = iter_expanding_folds(window_end="2020-06-30", initial_train_end="2018-12-31", step_months=12)
    ok = (folds[0].train_end == pd.Timestamp("2018-12-31")
          and folds[0].oos_end == pd.Timestamp("2019-12-31")
          and folds[-1].oos_end == pd.Timestamp("2020-06-30")  # clipped to window_end, not overshooting
          and all(f.train_end < f.oos_end for f in folds))
    print_check("iter_expanding_folds: steps annually, last fold clipped to window_end", ok,
                str([(f.fold_id, str(f.train_end.date()), str(f.oos_end.date())) for f in folds]))
    return passed + ok, failed + (not ok)


def test_monthly_decision_dates_last_trading_day(passed, failed):
    dates = pd.bdate_range("2020-01-01", "2020-03-31")  # business days, no holiday calendar
    out = monthly_decision_dates(dates)
    ok = len(out) == 3 and out[-1] == dates[dates.month == 3].max()
    print_check("monthly_decision_dates: one date per month, the last trading day of each",
                ok, str([str(d.date()) for d in out]))
    return passed + ok, failed + (not ok)


def test_k_trading_days_later(passed, failed):
    calendar = pd.bdate_range("2020-01-01", periods=10)
    dates = pd.DatetimeIndex([calendar[2]])
    out = k_trading_days_later(calendar, dates, k=3)
    ok = out[0] == calendar[5]
    print_check("k_trading_days_later: offsets by trading-day POSITION, not calendar days",
                ok, str(out[0]))
    return passed + ok, failed + (not ok)


def test_k_trading_days_later_past_end_is_nat(passed, failed):
    calendar = pd.bdate_range("2020-01-01", periods=5)
    dates = pd.DatetimeIndex([calendar[4]])
    out = k_trading_days_later(calendar, dates, k=3)
    ok = pd.isna(out[0])
    print_check("k_trading_days_later: past the end of the calendar returns NaT, not a fabricated date",
                ok, str(out[0]))
    return passed + ok, failed + (not ok)


def _membership():
    return pd.DataFrame({
        "period_id": [1, 1, 2, 2],
        "ticker": ["AAA", "BBB", "AAA", "CCC"],
        "start": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"]),
        "end": pd.to_datetime(["2020-02-01", "2020-02-01", "2020-03-01", "2020-03-01"]),
    })


def test_active_universe_by_date_half_open(passed, failed):
    universe = active_universe_by_date(_membership(), pd.DatetimeIndex(["2020-01-15", "2020-02-15"]))
    jan = set(universe.loc[universe["decision_date"] == "2020-01-15", "ticker"])
    feb = set(universe.loc[universe["decision_date"] == "2020-02-15", "ticker"])
    ok = jan == {"AAA", "BBB"} and feb == {"AAA", "CCC"}
    print_check("active_universe_by_date: half-open [start,end) period membership, rotates correctly",
                ok, f"jan={jan}, feb={feb}")
    return passed + ok, failed + (not ok)


def test_build_forward_targets_known_returns(passed, failed):
    # AAA doubles, BBB flat, bench +10% over the k-day window -- hand-computable relative returns.
    calendar = pd.bdate_range("2020-01-01", periods=10)
    prices_wide = pd.DataFrame({
        "AAA": np.linspace(10, 10, 10),
        "BBB": np.linspace(10, 10, 10),
    }, index=calendar)
    prices_wide.loc[calendar[5], "AAA"] = 20.0  # AAA: +100% from decision_date to decision_date+k
    bench = pd.Series(10.0, index=calendar)
    bench.loc[calendar[5]] = 11.0  # bench: +10% over the same window

    decision_dates = pd.DatetimeIndex([calendar[2]])
    universe = pd.DataFrame({"decision_date": decision_dates.tolist() * 2, "ticker": ["AAA", "BBB"]})
    out = build_forward_targets(prices_wide, bench, decision_dates, k=3, universe=universe)

    aaa_ret = float(out.loc[out["ticker"] == "AAA", "fwd_rel_return"].iloc[0])
    bbb_ret = float(out.loc[out["ticker"] == "BBB", "fwd_rel_return"].iloc[0])
    ok = abs(aaa_ret - (1.0 - 0.10)) < TOL and abs(bbb_ret - (0.0 - 0.10)) < TOL
    print_check("build_forward_targets: relative return = own return minus bench return, hand-verified",
                ok, f"AAA={aaa_ret:.4f} (expect 0.90), BBB={bbb_ret:.4f} (expect -0.10)")
    return passed + ok, failed + (not ok)


def test_build_forward_targets_rank_bounded(passed, failed):
    calendar = pd.bdate_range("2020-01-01", periods=10)
    rng = np.random.default_rng(1)
    tickers = [f"T{i}" for i in range(8)]
    prices_wide = pd.DataFrame(
        {t: 10.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 10)) for t in tickers}, index=calendar,
    )
    bench = pd.Series(10.0 * np.cumprod(1.0 + rng.normal(0, 0.005, 10)), index=calendar)
    decision_dates = pd.DatetimeIndex([calendar[1]])
    universe = pd.DataFrame({"decision_date": decision_dates.tolist() * len(tickers), "ticker": tickers})
    out = build_forward_targets(prices_wide, bench, decision_dates, k=3, universe=universe)
    ok = bool((out["target_rank"].dropna().abs() <= 0.5 + TOL).all()) and out["n_universe_members"].iloc[0] == 8
    print_check("build_forward_targets: target_rank bounded to [-0.5,0.5], n_universe_members correct",
                ok, str(out[["ticker", "fwd_rel_return", "target_rank"]].to_dict("records")))
    return passed + ok, failed + (not ok)


def main() -> int:
    print_header("h_series/spine.py")
    passed = failed = 0
    for test_fn in [
        test_hac_lag_for_horizon,
        test_iter_expanding_folds_boundaries,
        test_monthly_decision_dates_last_trading_day,
        test_k_trading_days_later,
        test_k_trading_days_later_past_end_is_nat,
        test_active_universe_by_date_half_open,
        test_build_forward_targets_known_returns,
        test_build_forward_targets_rank_bounded,
    ]:
        passed, failed = test_fn(passed, failed)
    print_section_end(passed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
