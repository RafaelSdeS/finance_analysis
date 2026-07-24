"""
test_backtest.py -- checks run_backtest against an independent, dead-simple
non-vectorized reference simulator (an oracle, not the implementation under
test), on a 2-ticker synthetic panel with a forced universe exit (proposal
§5 pothole P6), plus a zero-cost/no-op run reproducing buy-and-hold exactly.

cdi=0 throughout (isolates equity/turnover mechanics); cdi_curve's own cash
compounding is checked separately and directly below.

Fast group (synthetic only). Run: python tests/portfolio/test_backtest.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.portfolio.backtest import (  # noqa: E402
    buy_and_hold_curve, cdi_curve, equal_weight_fn, run_backtest,
)
from tests.test_utils import print_check, print_header, print_section_end  # noqa: E402


def _reference_sim(prices: dict, membership_periods: dict, weights_of, one_way_cost, initial_value=1.0):
    """Independent, non-vectorized oracle mirroring run_backtest's intended
    semantics. Cash is assumed to earn 0% (every scenario below uses cdi=0),
    so this deliberately skips cash-compounding -- that's covered separately
    against cdi_curve(). prices: {ticker: {date: price}}. membership_periods:
    {start_date: set(tickers)}. weights_of(universe) -> dict[ticker, weight]."""
    reb_dates = sorted(membership_periods)
    all_dates = sorted({d for p in prices.values() for d in p})
    prev_shares, prev_cash = {}, initial_value
    curve, log = {}, []
    for k, t in enumerate(reb_dates):
        price_t = {tkr: prices[tkr][t] for tkr in prices if t in prices[tkr]}
        equity_value = sum(sh * price_t[tkr] for tkr, sh in prev_shares.items() if tkr in price_t)
        port_value = prev_cash + equity_value
        prev_w = ({tkr: sh * price_t[tkr] / port_value for tkr, sh in prev_shares.items() if tkr in price_t}
                  if port_value > 0 else {})
        new_w = weights_of(membership_periods[t])
        full = set(prev_w) | set(new_w)
        turnover = sum(abs(new_w.get(tkr, 0.0) - prev_w.get(tkr, 0.0)) for tkr in full)
        cost = one_way_cost * turnover
        capital = port_value * (1 - cost)
        new_shares = {tkr: capital * w / price_t[tkr] for tkr, w in new_w.items()}
        cash_after = capital * (1 - sum(new_w.values()))
        log.append({"date": t, "turnover": turnover, "cost": cost, "n_holdings": len(new_shares)})

        next_t = reb_dates[k + 1] if k + 1 < len(reb_dates) else all_dates[-1]
        for d in all_dates:
            if t <= d <= next_t:
                eq = sum(sh * prices[tkr][d] for tkr, sh in new_shares.items() if d in prices[tkr])
                curve[d] = eq + cash_after
        prev_shares, prev_cash = new_shares, cash_after
    return curve, log


def _two_ticker_panel():
    dates = pd.bdate_range("2020-01-01", periods=10)
    a_prices = {d: 100.0 + i for i, d in enumerate(dates)}
    b_prices = {d: (50.0 if i < 5 else 55.0) for i, d in enumerate(dates)}
    rows = (
        [{"ticker": "A", "trade_date": d, "adj_close": p} for d, p in a_prices.items()]
        + [{"ticker": "B", "trade_date": d, "adj_close": p} for d, p in b_prices.items()]
    )
    prices_df = pd.DataFrame(rows)
    cdi_df = pd.DataFrame({"trade_date": dates, "cdi": 0.0})
    return dates, a_prices, b_prices, prices_df, cdi_df


def test_forced_exit_and_compounding():
    passed = failed = 0
    dates, a_prices, b_prices, prices_df, cdi_df = _two_ticker_panel()

    membership_df = pd.DataFrame([
        {"ticker": "A", "start": dates[0], "end": dates[5]},
        {"ticker": "B", "start": dates[0], "end": dates[5]},
        {"ticker": "A", "start": dates[5], "end": dates[-1] + pd.Timedelta(days=1)},
    ])
    one_way_cost = 0.001
    curve, log = run_backtest(prices_df, cdi_df, membership_df, equal_weight_fn, one_way_cost=one_way_cost)

    ref_prices = {"A": a_prices, "B": b_prices}
    ref_periods = {dates[0]: {"A", "B"}, dates[5]: {"A"}}
    ref_weights_of = lambda uni: ({t: 1.0 / len(uni) for t in uni} if uni else {})  # noqa: E731
    ref_curve, ref_log = _reference_sim(ref_prices, ref_periods, ref_weights_of, one_way_cost)

    curve_ok = all(np.isclose(curve[d], ref_curve[d]) for d in dates)
    print_check("equity curve matches independent reference simulator", bool(curve_ok))
    passed, failed = passed + curve_ok, failed + (not curve_ok)

    turnover_ok = all(
        np.isclose(log.iloc[i]["turnover"], ref_log[i]["turnover"]) for i in range(len(ref_log))
    )
    print_check("per-rebalance turnover matches reference", bool(turnover_ok))
    passed, failed = passed + turnover_ok, failed + (not turnover_ok)

    forced_exit_ok = log.iloc[1]["turnover"] > 0.9 and log.iloc[1]["n_holdings"] == 1
    print_check(
        "B's forced exit at the 2nd rebalance is counted as real turnover "
        "even though equal_weight_fn never mentioned B",
        bool(forced_exit_ok), f"turnover={log.iloc[1]['turnover']:.4f}, n_holdings={log.iloc[1]['n_holdings']}",
    )
    passed, failed = passed + forced_exit_ok, failed + (not forced_exit_ok)

    return passed, failed


def test_zero_cost_noop_reproduces_buy_and_hold():
    passed = failed = 0
    dates, a_prices, b_prices, prices_df, cdi_df = _two_ticker_panel()

    # both tickers stay in the universe for the whole span -- no forced exits
    membership_df = pd.DataFrame([
        {"ticker": "A", "start": dates[0], "end": dates[-1] + pd.Timedelta(days=1)},
        {"ticker": "B", "start": dates[0], "end": dates[-1] + pd.Timedelta(days=1)},
    ])

    def noop_fn(date, uni, state):
        if state["prev_weights"]:
            return state["prev_weights"]
        n = len(uni)
        return {t: 1.0 / n for t in uni} if n else {}

    curve, log = run_backtest(prices_df, cdi_df, membership_df, noop_fn, one_way_cost=0.0)

    expected = {
        d: 0.5 * (a_prices[d] / a_prices[dates[0]]) + 0.5 * (b_prices[d] / b_prices[dates[0]])
        for d in dates
    }
    curve_ok = all(np.isclose(curve[d], expected[d]) for d in dates)
    print_check("zero-cost no-op run reproduces a static 50/50 buy-and-hold exactly", bool(curve_ok))
    passed, failed = passed + curve_ok, failed + (not curve_ok)

    no_trade_ok = (log["turnover"].iloc[1:] < 1e-9).all()
    print_check("turnover is ~0 at every rebalance after the first (true no-op)", bool(no_trade_ok))
    passed, failed = passed + no_trade_ok, failed + (not no_trade_ok)

    return passed, failed


def test_cdi_curve():
    dates = pd.bdate_range("2021-01-01", periods=5)
    cdi_df = pd.DataFrame({"trade_date": dates, "cdi": [0.01] * 5})
    curve = cdi_curve(cdi_df, initial_value=2.0)
    expected = 2.0 * np.cumprod(1 + cdi_df["cdi"].to_numpy() / 100)
    ok = np.allclose(curve.to_numpy(), expected)
    print_check("cdi_curve matches independent cumprod", bool(ok))
    return int(ok), int(not ok)


def test_buy_and_hold_curve():
    dates = pd.bdate_range("2021-01-01", periods=5)
    prices = pd.Series([10.0, 11.0, 9.0, 12.0, 12.0], index=dates)
    curve = buy_and_hold_curve(prices, initial_value=3.0)
    expected = 3.0 * prices / prices.iloc[0]
    ok = np.allclose(curve.to_numpy(), expected.to_numpy())
    print_check("buy_and_hold_curve matches direct normalization", bool(ok))
    return int(ok), int(not ok)


def main():
    print_header("test_backtest")
    p1, f1 = test_forced_exit_and_compounding()
    p2, f2 = test_zero_cost_noop_reproduces_buy_and_hold()
    p3, f3 = test_cdi_curve()
    p4, f4 = test_buy_and_hold_curve()
    passed, failed = p1 + p2 + p3 + p4, f1 + f2 + f3 + f4
    print_section_end(passed, failed)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
