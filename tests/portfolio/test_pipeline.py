"""
test_pipeline.py -- integration checks for pipeline.make_full_weights_fn:
wiring sanity (weights feasible), the two equal-weight fallbacks (no alpha
prediction yet / not enough risk-model history yet), a tilt toward the
higher-alpha name, then a full run through backtest.run_backtest() end to
end.

Fast group (synthetic only). Run: python tests/portfolio/test_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.portfolio.backtest import run_backtest  # noqa: E402
from src.portfolio.pipeline import make_full_weights_fn, scaled_sigma  # noqa: E402
from src.portfolio.risk import shrinkage_cov  # noqa: E402
from tests.test_utils import print_check, print_header, print_section_end  # noqa: E402


def _synthetic_prices(n_days=400, n_tickers=5, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [chr(ord("A") + i) for i in range(n_tickers)]
    drifts = np.linspace(-0.0003, 0.0008, n_tickers)
    rows = []
    for tkr, drift in zip(tickers, drifts):
        prices = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n_days)))
        rows.extend({"ticker": tkr, "trade_date": d, "adj_close": p} for d, p in zip(dates, prices))
    return pd.DataFrame(rows), dates, tickers


def main():
    print_header("test_pipeline")
    passed = failed = 0

    prices_df, dates, tickers = _synthetic_prices()
    price_wide = prices_df.pivot(index="trade_date", columns="ticker", values="adj_close")

    reb_dates = [dates[i] for i in (50, 150, 250, 350)]
    # only the LAST two rebalances have an alpha prediction -- the first two
    # exercise the "no prediction yet" fallback.
    alpha_by_date = {
        reb_dates[2]: {"A": -0.05, "B": -0.02, "C": 0.0, "D": 0.03, "E": 0.08},
        reb_dates[3]: {"A": -0.05, "B": -0.02, "C": 0.0, "D": 0.03, "E": 0.08},
    }
    weights_fn = make_full_weights_fn(alpha_by_date, price_wide, sigma_window=100, c1=0.001)

    # --- fallback: no alpha prediction yet -> equal weight
    w = weights_fn(reb_dates[0], set(tickers), {"prev_weights": {}})
    equal_ok = all(np.isclose(v, 1 / len(tickers)) for v in w.values()) and len(w) == len(tickers)
    print_check("falls back to equal-weight when no alpha prediction exists yet", bool(equal_ok))
    passed, failed = passed + equal_ok, failed + (not equal_ok)

    # --- with alpha + enough sigma history: feasible, sums to <=1 (ex-cash residual)
    w2 = weights_fn(reb_dates[2], set(tickers), {"prev_weights": {}})
    feasible_ok = all(v >= -1e-9 for v in w2.values()) and sum(w2.values()) <= 1 + 1e-3
    print_check("with alpha + sigma available, weights are feasible (>=0, sum<=1)", bool(feasible_ok))
    passed, failed = passed + feasible_ok, failed + (not feasible_ok)

    tilts_toward_e_ok = w2.get("E", 0) >= w2.get("A", 0)
    print_check("tilts toward the ticker with the highest supplied alpha (E > A)",
                bool(tilts_toward_e_ok), f"w={ {k: round(v, 4) for k, v in w2.items()} }")
    passed, failed = passed + tilts_toward_e_ok, failed + (not tilts_toward_e_ok)

    # --- Phase 1.3: shrink_factor pulls the allocation toward the
    # equal-weight/risk-parity solution as it goes to 1 (alpha zeroed out ->
    # nothing left to justify tilting away from diversification).
    weights_fn_shrunk = make_full_weights_fn(alpha_by_date, price_wide, sigma_window=100,
                                              c1=0.001, shrink_factor=1.0)
    w_shrunk = weights_fn_shrunk(reb_dates[2], set(tickers), {"prev_weights": {}})
    spread_unshrunk = w2.get("E", 0) - w2.get("A", 0)
    spread_shrunk = w_shrunk.get("E", 0) - w_shrunk.get("A", 0)
    shrink_ok = spread_shrunk < spread_unshrunk
    print_check("shrink_factor=1.0 narrows the E-vs-A tilt vs shrink_factor=0.0 (default)",
                bool(shrink_ok), f"unshrunk spread={spread_unshrunk:.4f}, shrunk spread={spread_shrunk:.4f}")
    passed, failed = passed + shrink_ok, failed + (not shrink_ok)

    # --- regression: Sigma must be scaled to the same horizon as alpha, not
    # left as a raw daily covariance (found empirically 2026-07-24 -- an
    # unscaled Sigma is ~100-300x too small relative to a 252-day alpha,
    # making the risk-aversion term negligible and the optimizer chase
    # whichever ticker has the highest apparent alpha with almost no
    # diversification). Checked directly against an independently-computed
    # shrinkage_cov, not indirectly through a weights_fn call -- a synthetic
    # alpha spread large enough to matter for the OTHER checks above turned
    # out to swamp the risk term either way, making an indirect behavioral
    # check here unreliable (both scaled and unscaled cornered to ~100% in
    # one name, a ~1e-5 difference that isn't a meaningful validation).
    as_of = reb_dates[2]
    expected_daily_sigma = shrinkage_cov(
        price_wide.loc[:as_of, tickers].ffill(limit=5).pct_change(fill_method=None).iloc[1:].tail(100)
    )
    sigma_h1 = scaled_sigma(price_wide, as_of, tickers, sigma_window=100, horizon_td=1)
    sigma_h252 = scaled_sigma(price_wide, as_of, tickers, sigma_window=100, horizon_td=252)

    unscaled_matches_raw_ok = np.allclose(sigma_h1.to_numpy(), expected_daily_sigma.to_numpy())
    print_check("horizon_td=1 reproduces the raw daily shrinkage_cov exactly (sanity)",
                bool(unscaled_matches_raw_ok))
    passed, failed = passed + unscaled_matches_raw_ok, failed + (not unscaled_matches_raw_ok)

    scaling_ok = np.allclose(sigma_h252.to_numpy(), expected_daily_sigma.to_numpy() * 252)
    print_check("scaled_sigma(horizon_td=252) is exactly 252x the raw daily covariance",
                bool(scaling_ok))
    passed, failed = passed + scaling_ok, failed + (not scaling_ok)

    # --- fallback: no trailing return history at all yet (the very first day
    # of history -- pct_change() has nothing to compute from). Give this date
    # an alpha entry too, so a passing check here is actually exercising the
    # "no sigma history" path, not just reusing the "no alpha" path above.
    alpha_by_date_with_day0 = {**alpha_by_date, dates[0]: {t: 0.05 for t in tickers}}
    weights_fn_day0 = make_full_weights_fn(alpha_by_date_with_day0, price_wide, sigma_window=100, c1=0.001)
    w3 = weights_fn_day0(dates[0], set(tickers), {"prev_weights": {}})
    no_history_ok = all(np.isclose(v, 1 / len(tickers)) for v in w3.values())
    print_check("falls back to equal-weight when there's no trailing return history yet",
                bool(no_history_ok), f"w={ {k: round(v, 4) for k, v in w3.items()} }")
    passed, failed = passed + no_history_ok, failed + (not no_history_ok)

    # --- end-to-end through the real backtest harness. Membership needs a
    # period boundary at each of our chosen reb_dates -- rebalance_dates()
    # reads its calendar straight from membership's distinct `start` values.
    membership = pd.DataFrame([
        {"ticker": t, "start": reb_dates[i],
         "end": reb_dates[i + 1] if i + 1 < len(reb_dates) else dates[-1] + pd.Timedelta(days=1)}
        for i in range(len(reb_dates)) for t in tickers
    ])
    cdi_df = pd.DataFrame({"trade_date": dates, "cdi": 0.02})
    curve, log = run_backtest(prices_df, cdi_df, membership, weights_fn, one_way_cost=0.001)
    e2e_ok = curve.notna().all() and len(log) == len(reb_dates)
    print_check("runs end-to-end through run_backtest with no NaN in the equity curve",
                bool(e2e_ok), f"{len(curve)} daily obs, final value={curve.iloc[-1]:.4f}")
    passed, failed = passed + e2e_ok, failed + (not e2e_ok)

    print_section_end(passed, failed)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
