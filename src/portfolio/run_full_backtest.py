"""
run_full_backtest.py -- proposal Phase 2.7 "Done when" deliverable: the
full predict-then-optimize pipeline (alpha -> Sigma -> optimizer) wired end
to end through the backtest harness, on the real dataset. Reports the full
§8 panel including the cost-sensitivity curve, vs. all three baselines.

Run: python -m src.portfolio.run_full_backtest [--top-n 50] [--horizon-td 252] [--rebalance-freq Q]
    [--lam 5.0] [--c2 2.0] [--shrink-factor 0.3]

--lam/--c2/--shrink-factor set plan Phase 1.2/1.3's "boring" candidate
config (raise risk-aversion, penalize turnover quadratically, shrink alpha
toward 0), printed as its own block AFTER the cost-sensitivity sweep, at
the true 0.03% fee -- directly comparable to that sweep's first row
("c1=0.0300%", which is today's lam=1/c2=0/shrink=0 config at the same
real cost). Defaults are an ORDER-OF-MAGNITUDE GUESS, not empirically
tuned (no backtest run was executed to pick these; see
pipeline.py/optimizer.py docstrings for the sizing reasoning) -- read
Gate B (excess-CDI Sharpe >= max(EW, quintile-sort) AND turnover <= 2x)
off the two rows, then sweep --lam/--c2/--shrink-factor from there.
"""

import argparse

import pandas as pd

from src.build_dataset.paths import MACRO_DIR, OUTPUT_PATH, PRICES_DIR
from src.portfolio import alpha, contrarian, universe
from src.portfolio.backtest import buy_and_hold_curve, cdi_curve, equal_weight_fn, run_backtest
from src.portfolio.features import feature_columns
from src.portfolio.labels import forward_excess_return
from src.portfolio.metrics import CRISIS_WINDOWS, active_return_report, full_report, print_regime_slices, print_report
from src.portfolio.pipeline import make_full_weights_fn

ONE_WAY_COST_SWEEP = [0.0003, 0.0015, 0.003]  # 0.03% floor -> 0.15% -> 0.3%
_NO_REBALANCE_LOG = pd.DataFrame({"turnover": [0.0]})


def main(top_n: int = 50, horizon_td: int = 252, rebalance_freq: str = "Q",
         lam: float = 5.0, c2: float = 2.0, shrink_factor: float = 0.3):
    print("Loading dataset (full feature set)...")
    base_cols = ["ticker", "trade_date", "adj_close", "traded_amount"]
    all_cols = sorted(set(base_cols) | set(feature_columns(include_sector=False)))
    df = pd.read_parquet(OUTPUT_PATH, columns=all_cols)

    print(f"Building point-in-time liquid universe (top_n={top_n}, rebalance_freq={rebalance_freq})...")
    membership = universe.liquid_universe(df[["ticker", "trade_date", "traded_amount"]],
                                           top_n=top_n, rebalance_freq=rebalance_freq)
    df = universe.restrict_to_universe(df, membership)
    print(f"  restricted to {len(df)} rows, {df['ticker'].nunique()} tickers ever in the universe")

    print(f"Building the {horizon_td}-day forward-excess-return label...")
    df["label"] = forward_excess_return(df, horizon_td=horizon_td)

    reb_dates = universe.rebalance_dates(membership)
    # Measured from the actual calendar, not looked up from rebalance_freq's
    # alias string -- correct regardless of what pandas period alias was used,
    # and avoids a second place that would need updating if a new alias is
    # ever passed. Bug found 2026-07-25: every full_report() call below used
    # to rely on its hardcoded rebalances_per_year=4 default even when
    # rebalance_freq="M" -- silently under-annualizing turnover ~3x.
    years_spanned = (reb_dates.max() - reb_dates.min()).days / 365.25
    rebalances_per_year = (len(reb_dates) - 1) / years_spanned
    print(f"  measured {rebalances_per_year:.1f} rebalances/year from the membership calendar")
    print(f"Walk-forward alpha training/prediction across {len(reb_dates)} rebalance dates "
          "(reused unchanged across the cost sweep below -- alpha doesn't depend on cost)...")
    preds = alpha.walk_forward_predict(df, reb_dates, horizon_td=horizon_td)  # n_estimators: alpha.DEFAULT_N_ESTIMATORS
    preds_by_date = {d: g.set_index("ticker")["alpha"].to_dict() for d, g in preds.groupby("date")}
    print(f"  {len(preds_by_date)} dates produced a prediction")

    # Layer 2: contrarian equity-vs-CDI exposure ("buy the cannons, sell the
    # violins"). Valuation-driven (aggregate ERP), gentle 50-100% band, causal.
    # Cost-independent, so computed once and reused across the sweep like preds.
    exposure_by_date = contrarian.equity_exposure(df, reb_dates)
    _exp = pd.Series(exposure_by_date)
    print(f"  contrarian exposure: {_exp.min():.0%}..{_exp.max():.0%} equity "
          f"(median {_exp.median():.0%}) across {len(_exp)} rebalances")

    prices = df[["ticker", "trade_date", "adj_close"]]
    price_wide = prices.pivot(index="trade_date", columns="ticker", values="adj_close")  # NOT ffilled -- see pipeline.py
    cdi = pd.read_parquet(OUTPUT_PATH, columns=["trade_date", "cdi"]).drop_duplicates().sort_values("trade_date")
    cdi_daily = cdi.set_index("trade_date")["cdi"]  # excess-over-CDI Sharpe (plan Phase 0.1)
    selic = pd.read_parquet(MACRO_DIR / "selic.parquet").rename(columns={"reference_date": "trade_date"})
    selic_daily = selic.set_index("trade_date")["selic"]

    print("\nRunning equal-weight baseline...")
    eq_curve, eq_log = run_backtest(prices, cdi, membership, equal_weight_fn)
    eq_returns = eq_curve.pct_change().dropna()

    print("Loading BOVA11 / 100% CDI baselines...")
    bova = pd.read_parquet(PRICES_DIR / "BOVA11.parquet", columns=["trade_date", "adj_close"])
    bova_curve = buy_and_hold_curve(bova.set_index("trade_date")["adj_close"].reindex(eq_curve.index).ffill())
    cdi_only_curve = cdi_curve(cdi[cdi["trade_date"].isin(eq_curve.index)])

    print_report("Equal-weight baseline", full_report(
        eq_curve, eq_log, selic_daily=selic_daily, cdi_daily=cdi_daily, rebalances_per_year=rebalances_per_year))
    print_report("BOVA11 buy-and-hold", full_report(bova_curve, _NO_REBALANCE_LOG, selic_daily=selic_daily, cdi_daily=cdi_daily))
    print_report("100% CDI", full_report(cdi_only_curve, _NO_REBALANCE_LOG, selic_daily=selic_daily, cdi_daily=cdi_daily))

    # Plan Phase 0.4: sanity-check Layer 2's contrarian timing BEFORE trusting
    # its backtest numbers -- does exposure actually rise near crisis troughs
    # ("cannons") and fall near market highs ("violins"), or is it inverted?
    print("\n" + "=" * 60)
    print("Layer 2 sanity check: contrarian exposure vs BOVA11 drawdown, by regime")
    print("=" * 60)
    exp_series = pd.Series(exposure_by_date).sort_index()
    bova_dd = (bova_curve / bova_curve.cummax() - 1).reindex(exp_series.index, method="ffill")
    print(f"  overall mean exposure: {exp_series.mean():.1%}  (band is 50-100%)")
    for crisis_name, (start, end) in CRISIS_WINDOWS.items():
        window = exp_series[(exp_series.index >= start) & (exp_series.index <= end)]
        if len(window):
            print(f"  {crisis_name:<20} mean={window.mean():.1%}  min={window.min():.1%}  "
                  f"max={window.max():.1%}  n={len(window)}   <- want HIGH (buying the cannons)")
    near_ath = bova_dd >= -0.01
    if near_ath.any():
        print(f"  {'near BOVA11 all-time-highs':<20} mean={exp_series[near_ath].mean():.1%}  "
              f"n={int(near_ath.sum())}   <- want LOW (selling at the violins)")

    print("\n" + "=" * 60)
    print("Full pipeline (alpha -> Sigma -> optimizer): cost-sensitivity curve")
    print("=" * 60)
    for c1 in ONE_WAY_COST_SWEEP:
        weights_fn = make_full_weights_fn(preds_by_date, price_wide, sigma_window=252,
                                           horizon_td=horizon_td, c1=c1, lam=1.0, w_max=0.1,
                                           exposure_by_date=exposure_by_date)
        curve, log = run_backtest(prices, cdi, membership, weights_fn, one_way_cost=c1)
        report = full_report(curve, log, selic_daily=selic_daily, cdi_daily=cdi_daily,
                              benchmark_returns=eq_returns, rebalances_per_year=rebalances_per_year)
        print_report(f"one-way c1={c1:.4%} (round-trip {2 * c1:.4%})", report)

        # Plan Phase 0.3: active return vs equal-weight, by regime -- is the
        # extra machinery (alpha+optimizer+overlay) adding anything, and where.
        active_slices = active_return_report(curve.pct_change().dropna(), eq_returns, selic_daily)
        print("  Active return vs equal-weight, by regime:")
        print_regime_slices(active_slices)

    print("\n" + "=" * 60)
    print("Phase 1 boring candidate: turnover control + alpha shrinkage")
    print("=" * 60)
    print(f"  lam={lam}  c2={c2}  shrink_factor={shrink_factor}  (vs Phase 0's lam=1.0 c2=0.0 "
          f"shrink=0.0 -- compare against the 'c1=0.0300%' row above, same true cost)")
    boring_c1 = ONE_WAY_COST_SWEEP[0]
    boring_weights_fn = make_full_weights_fn(preds_by_date, price_wide, sigma_window=252,
                                              horizon_td=horizon_td, c1=boring_c1, lam=lam, c2=c2,
                                              w_max=0.1, exposure_by_date=exposure_by_date,
                                              shrink_factor=shrink_factor)
    boring_curve, boring_log = run_backtest(prices, cdi, membership, boring_weights_fn, one_way_cost=boring_c1)
    boring_report = full_report(boring_curve, boring_log, selic_daily=selic_daily, cdi_daily=cdi_daily,
                                 benchmark_returns=eq_returns, rebalances_per_year=rebalances_per_year)
    print_report("Boring candidate (Phase 1.2/1.3)", boring_report)
    boring_active = active_return_report(boring_curve.pct_change().dropna(), eq_returns, selic_daily)
    print("  Active return vs equal-weight, by regime:")
    print_regime_slices(boring_active)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--horizon-td", type=int, default=252)
    parser.add_argument("--rebalance-freq", type=str, default="Q",
                         help="pandas period alias, e.g. Q (quarterly) or M (monthly)")
    parser.add_argument("--lam", type=float, default=5.0, help="Phase 1.2 boring-candidate risk aversion")
    parser.add_argument("--c2", type=float, default=2.0, help="Phase 1.2 boring-candidate quadratic turnover penalty")
    parser.add_argument("--shrink-factor", type=float, default=0.3,
                         help="Phase 1.3 boring-candidate alpha shrinkage toward 0, in [0, 1]")
    args = parser.parse_args()
    main(top_n=args.top_n, horizon_td=args.horizon_td, rebalance_freq=args.rebalance_freq,
         lam=args.lam, c2=args.c2, shrink_factor=args.shrink_factor)
