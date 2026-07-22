"""
Test: conviction_model/labels.py's label construction (Phase 0).
Deterministic synthetic price/CDI data -- no dependency on data/raw or
data/processed, so this stays in the `fast` test group.

Run from project root:
    python tests/conviction_model/test_labels.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.conviction_model.labels import (  # noqa: E402
    DRAWDOWN_HORIZON, HORIZONS, VOL_LOOKBACK_DAYS, build_conviction_labels, build_cdi_cumulative_index,
    compute_drawdown_severity, compute_risk_adjusted_excess_returns, trailing_volatility,
)
from src.h_series.spine import build_forward_targets  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402

TOL = 1e-6


def test_trailing_volatility_exact(passed, failed):
    # log-price walks 0,a,0,a,0 -> the 4 diffs are exactly [a,-a,a,-a] ->
    # closed-form sample std (ddof=1) = a * sqrt(4/3), hand-derivable.
    a = 0.01
    log_price = np.array([0.0, a, 0.0, a, 0.0])
    prices = pd.DataFrame({"X": np.exp(log_price)}, index=pd.bdate_range("2020-01-01", periods=5))
    vol = trailing_volatility(prices, window=4)
    expected = a * np.sqrt(4.0 / 3.0)
    ok = abs(float(vol["X"].iloc[4]) - expected) < TOL
    print_check("trailing_volatility: closed-form sample std on a known alternating return path",
                ok, f"got {vol['X'].iloc[4]:.6f}, expected {expected:.6f}")
    return passed + ok, failed + (not ok)


def test_trailing_volatility_warmup_is_nan(passed, failed):
    prices = pd.DataFrame({"X": np.linspace(10, 11, 5)}, index=pd.bdate_range("2020-01-01", periods=5))
    vol = trailing_volatility(prices, window=4)
    ok = bool(vol["X"].iloc[:4].isna().all()) and bool(pd.notna(vol["X"].iloc[4]))
    print_check("trailing_volatility: NaN for the first `window` rows (warm-up), defined after",
                ok, str(vol["X"].tolist()))
    return passed + ok, failed + (not ok)


def test_trailing_volatility_masks_zero_price(passed, failed):
    # A single vendor-rounding-artifact zero (adj_close underflows its 2dp
    # floor -- see features.py::adj_close_precision_degraded) must not raise
    # a divide-by-zero warning or poison the whole window with -inf/NaN.
    import warnings

    vals = np.linspace(10, 11, 10)
    vals[4] = 0.0
    prices = pd.DataFrame({"X": vals}, index=pd.bdate_range("2020-01-01", periods=10))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            vol = trailing_volatility(prices, window=4)
            no_warning = True
        except RuntimeWarning:
            no_warning = False

    last = vol["X"].iloc[-1] if no_warning else None
    ok = no_warning and pd.notna(last) and np.isfinite(last)
    print_check("trailing_volatility: a zero price is masked to NaN, not log(0)=-inf",
                ok, "raised RuntimeWarning" if not no_warning else f"last={last}")
    return passed + ok, failed + (not ok)


def test_drawdown_severity_masks_interior_zero_price(passed, failed):
    # A single vendor-rounding-artifact zero MID-WINDOW (adj_close underflows its 2dp
    # floor -- see features.py::adj_close_precision_degraded) must not raise a
    # divide-by-zero warning or poison the whole path to +inf via log(0)=-inf leaking
    # into excess_path/running_max. Regression test for compute_drawdown_severity
    # missing the same mask trailing_volatility already has (this file's own sibling
    # function -- the exact caveat this bug was caught by, 2026-07-22).
    import warnings

    n_post = DRAWDOWN_HORIZON + 5
    calendar = pd.bdate_range("2015-01-01", periods=n_post + 10)
    decision_date = calendar[5]
    decision_idx = calendar.get_loc(decision_date)

    prices = np.full(len(calendar), 10.0)
    prices[decision_idx:decision_idx + n_post] = np.linspace(10.0, 12.0, n_post)  # steady climb
    prices[decision_idx + 50] = 0.0  # vendor-rounding zero, mid-window (not on decision_date itself)

    prices_wide = pd.DataFrame({"ZEROED": prices}, index=calendar)
    cdi_index = pd.Series(1.0, index=calendar)  # flat CDI
    decision_dates = pd.DatetimeIndex([decision_date])
    universe = pd.DataFrame({"decision_date": decision_dates, "ticker": ["ZEROED"]})

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            out = compute_drawdown_severity(prices_wide, cdi_index, decision_dates, universe)
            no_warning = True
        except RuntimeWarning:
            no_warning = False

    severity = float(out["drawdown_severity"].iloc[0]) if no_warning else None
    ok = no_warning and severity is not None and np.isfinite(severity)
    print_check("compute_drawdown_severity: an interior zero price is masked to NaN, not log(0)=-inf "
                "poisoning severity to inf", ok,
                "raised RuntimeWarning" if not no_warning else f"severity={severity}")
    return passed + ok, failed + (not ok)


def test_risk_adjusted_return_masks_degenerate_zero_vol(passed, failed):
    # A pinned-price artifact: adj_close is EXACTLY flat (a stuck vendor-rounding
    # constant, see adj_close_precision_degraded) for the entire trailing_vol lookback
    # window before decision_date -- trailing_vol is exactly 0.0 (finite, past warm-up,
    # NOT NaN). Dividing the forward return by that must be masked to NaN, not blown up
    # to +-inf.
    n_pre = VOL_LOOKBACK_DAYS + 5
    n_post = 30
    calendar = pd.bdate_range("2018-01-01", periods=n_pre + n_post)
    decision_date = calendar[n_pre - 1]

    prices = np.concatenate([
        np.full(n_pre, 5.0),               # pinned/flat -> trailing_vol == 0.0 exactly
        np.linspace(5.0, 5.5, n_post),      # real forward move after decision_date
    ])
    prices_wide = pd.DataFrame({"PINNED": prices}, index=calendar)
    cdi_index = pd.Series(1.0, index=calendar)  # flat CDI
    decision_dates = pd.DatetimeIndex([decision_date])
    universe = pd.DataFrame({"decision_date": decision_dates, "ticker": ["PINNED"]})

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            out = compute_risk_adjusted_excess_returns(prices_wide, cdi_index, decision_dates, universe,
                                                         horizons=(21,))
            no_warning = True
        except RuntimeWarning:
            no_warning = False

    val = out["risk_adj_excess_return_k21"].iloc[0] if no_warning else None
    ok = no_warning and pd.isna(val)
    print_check("compute_risk_adjusted_excess_returns: a pinned (trailing_vol==0) ticker's ratio is "
                "masked to NaN before dividing, not blown up to inf via a divide-by-zero",
                ok, "raised RuntimeWarning" if not no_warning else f"got {val}")
    return passed + ok, failed + (not ok)


def test_cdi_cumulative_index_compounds_correctly(passed, failed):
    # 11 calendar points, every one carrying rate r -> cumprod applies r 11
    # times by the last position (index 10 is the 11th element), not 10 --
    # an earlier version of this test got that off by one.
    calendar = pd.bdate_range("2020-01-01", periods=11)
    r = 0.0005  # daily decimal rate
    cdi_daily = pd.Series(r, index=calendar)
    idx = build_cdi_cumulative_index(cdi_daily, calendar)
    expected = (1.0 + r) ** 11
    ok = abs(float(idx.iloc[10]) - expected) < 1e-9
    print_check("build_cdi_cumulative_index: constant daily rate compounds geometrically",
                ok, f"got {idx.iloc[10]:.8f}, expected {expected:.8f}")
    return passed + ok, failed + (not ok)


def test_cdi_bench_integration(passed, failed):
    # Same pattern as h_series/test_spine.py's test_build_forward_targets_known_returns,
    # but the bench comes from THIS module's CDI construction, not a hand-built Series --
    # proves load_cdi_daily_decimal -> build_cdi_cumulative_index produces a usable,
    # correctly-shaped bench for build_forward_targets, not just plausible-looking output.
    calendar = pd.bdate_range("2020-01-01", periods=10)
    cdi_daily = pd.Series(0.0, index=calendar)  # flat CDI: 0%/day
    cdi_index = build_cdi_cumulative_index(cdi_daily, calendar)

    prices_wide = pd.DataFrame({"DOUBLE": np.linspace(10.0, 10.0, 10)}, index=calendar)
    prices_wide.loc[calendar[5], "DOUBLE"] = 20.0  # +100% from decision_date to decision_date+k

    decision_dates = pd.DatetimeIndex([calendar[2]])
    universe = pd.DataFrame({"decision_date": decision_dates, "ticker": ["DOUBLE"]})
    out = build_forward_targets(prices_wide, cdi_index, decision_dates, k=3, universe=universe)

    ret = float(out["fwd_rel_return"].iloc[0])
    ok = abs(ret - 1.0) < TOL  # CDI flat -> relative return == absolute return == +100%
    print_check("CDI bench wiring: flat CDI index behaves as a neutral benchmark",
                ok, f"fwd_rel_return={ret:.4f} (expect 1.0)")
    return passed + ok, failed + (not ok)


def _build_worked_example_prices():
    """COMPOUND: flat, then a single +40% jump at the very last day (k=504) --
    near-zero return at every earlier horizon, dominant at the longest one.
    REVERSAL: jumps +15% early (by k=21), holds through k=126, then declines
    linearly to -10% by k=504 -- a real peak-then-decline, unlike COMPOUND's
    monotonic path. Both tickers share an IDENTICAL pre-decision price path
    (same alternating oscillation), so their trailing_vol at the decision
    date is exactly equal -- any difference in risk-adjusted output is
    attributable purely to the differing forward paths, not the denominator.
    CDI is flat (0%/day) throughout, so excess return == raw return exactly,
    keeping the hand-reasoning simple."""
    n_pre, n_post = 120, DRAWDOWN_HORIZON + 20
    calendar = pd.bdate_range("2010-01-01", periods=n_pre + n_post)
    decision_idx = n_pre
    decision_date = calendar[decision_idx]

    # identical pre-decision oscillation for both tickers (for equal trailing_vol)
    a = 0.01
    osc = np.array([a if i % 2 == 0 else -a for i in range(n_pre)])
    pre_log = np.concatenate([[0.0], np.cumsum(osc)])  # length n_pre+1, ends back at 0

    # COMPOUND: flat post-decision except the very last point (+40%)
    compound_post = np.zeros(n_post)
    compound_post[DRAWDOWN_HORIZON] = np.log(1.40)
    compound_log = np.concatenate([pre_log, pre_log[-1] + compound_post[1:]])

    # REVERSAL: linear up to +15% by day 21, flat through day 126, linear down to -10% by day 504
    checkpoints_idx = np.array([0, 21, 126, DRAWDOWN_HORIZON])
    checkpoints_val = np.array([0.0, np.log(1.15), np.log(1.15), np.log(0.90)])
    xs = np.arange(0, n_post)
    reversal_post = np.interp(xs, checkpoints_idx, checkpoints_val)
    reversal_log = np.concatenate([pre_log, pre_log[-1] + reversal_post[1:]])

    prices_wide = pd.DataFrame(
        {"COMPOUND": np.exp(compound_log), "REVERSAL": np.exp(reversal_log)}, index=calendar,
    )
    cdi_index = pd.Series(1.0, index=calendar)  # flat CDI (0%/day)
    decision_dates = pd.DatetimeIndex([decision_date])
    universe = pd.DataFrame({"decision_date": decision_dates.tolist() * 2, "ticker": ["COMPOUND", "REVERSAL"]})
    return prices_wide, cdi_index, decision_dates, universe


def test_worked_examples_no_aggregation_six_columns(passed, failed):
    prices_wide, cdi_index, decision_dates, universe = _build_worked_example_prices()
    out = build_conviction_labels(prices_wide, cdi_index, decision_dates, universe)
    target_cols = [f"risk_adj_excess_return_k{k}" for k in HORIZONS] + ["drawdown_severity"]
    ok = out.shape[0] == 2 and all(c in out.columns for c in target_cols) and len(target_cols) == 6
    print_check("build_conviction_labels: exactly 6 independent target columns, no aggregation step",
                ok, f"columns={list(out.columns)}")
    return passed + ok, failed + (not ok)


def test_worked_example_compounder_dominated_by_long_horizon(passed, failed):
    prices_wide, cdi_index, decision_dates, universe = _build_worked_example_prices()
    out = build_conviction_labels(prices_wide, cdi_index, decision_dates, universe)
    row = out.loc[out["ticker"] == "COMPOUND"].iloc[0]
    k21 = abs(row["risk_adj_excess_return_k21"])
    k504 = abs(row["risk_adj_excess_return_k504"])
    ok = k504 > 10 * k21  # near-zero early, dominant at the 24-month horizon
    print_check("COMPOUND: k=504 output dominates k=21 (single late jump, flat before it)",
                ok, f"|k21|={k21:.4f}, |k504|={k504:.4f}")
    return passed + ok, failed + (not ok)


def test_worked_example_reversal_has_larger_drawdown(passed, failed):
    prices_wide, cdi_index, decision_dates, universe = _build_worked_example_prices()
    out = build_conviction_labels(prices_wide, cdi_index, decision_dates, universe)
    dd_compound = float(out.loc[out["ticker"] == "COMPOUND", "drawdown_severity"].iloc[0])
    dd_reversal = float(out.loc[out["ticker"] == "REVERSAL", "drawdown_severity"].iloc[0])
    ok = dd_reversal > dd_compound + 0.10  # REVERSAL round-trips ~25pp; COMPOUND is ~monotonic
    print_check("REVERSAL (peaks then gives it back) has materially higher drawdown_severity than "
                "COMPOUND (monotonic)", ok, f"COMPOUND={dd_compound:.4f}, REVERSAL={dd_reversal:.4f}")
    return passed + ok, failed + (not ok)


def test_worked_examples_are_visibly_different_vectors(passed, failed):
    prices_wide, cdi_index, decision_dates, universe = _build_worked_example_prices()
    out = build_conviction_labels(prices_wide, cdi_index, decision_dates, universe)
    cols = [f"risk_adj_excess_return_k{k}" for k in HORIZONS] + ["drawdown_severity"]
    v_compound = out.loc[out["ticker"] == "COMPOUND", cols].to_numpy()[0]
    v_reversal = out.loc[out["ticker"] == "REVERSAL", cols].to_numpy()[0]
    ok = not np.allclose(v_compound, v_reversal, atol=0.05)
    print_check("COMPOUND and REVERSAL land as visibly different 6-vectors, not collapsed together",
                ok, f"COMPOUND={np.round(v_compound, 3)}, REVERSAL={np.round(v_reversal, 3)}")
    return passed + ok, failed + (not ok)


def main() -> int:
    print_header("conviction_model/labels.py")
    passed = failed = 0
    for test_fn in [
        test_trailing_volatility_exact,
        test_trailing_volatility_warmup_is_nan,
        test_trailing_volatility_masks_zero_price,
        test_drawdown_severity_masks_interior_zero_price,
        test_risk_adjusted_return_masks_degenerate_zero_vol,
        test_cdi_cumulative_index_compounds_correctly,
        test_cdi_bench_integration,
        test_worked_examples_no_aggregation_six_columns,
        test_worked_example_compounder_dominated_by_long_horizon,
        test_worked_example_reversal_has_larger_drawdown,
        test_worked_examples_are_visibly_different_vectors,
    ]:
        passed, failed = test_fn(passed, failed)
    print_section_end(passed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
