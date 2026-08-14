"""
run_alpha_diagnostic.py -- proposal Phase 2.6 "Done when" deliverable: runs
the LightGBM walk-forward forecaster on the REAL point-in-time liquid
universe, reports out-of-sample rank-IC, and compares a simple alpha-ranked
(top-half, equal-weight -- "still equal-ish, pre-optimizer") strategy
against the Phase 2.3 equal-weight floor, same universe/dates/costs.

Run: python -m src.portfolio.run_alpha_diagnostic [--top-n 50] [--horizon-td 252]
"""

import argparse
import json

import numpy as np
import pandas as pd

from src.build_dataset.paths import OUTPUT_PATH, PRICES_DIR, SPLIT_CONFIG_PATH
from src.portfolio import alpha, artifacts, contrarian, universe
from src.portfolio.backtest import buy_and_hold_curve, cdi_curve, equal_weight_fn, run_backtest
from src.portfolio.features import feature_columns
from src.portfolio.labels import forward_excess_return
from src.portfolio.metrics import (
    deflated_sharpe_ratio, excess_over_cdi_sharpe, full_report, information_ratio,
    newey_west_tstat, print_report,
)
from src.portfolio.visualize_portfolio import build_dashboard, save_dashboard

WINDOW_CHOICES = ("train", "trainval", "test", "full")

# Plan V.1a: `pl` (and everything derived from it -- earnings_yield, and every P/L-based
# fundamental ratio anywhere in the dataset) is exactly 0% populated before this date
# (CLAUDE.md, confirmed dataset-wide, not a gradual ramp: 2010=0.0%, 2011=62.3%). Predictions
# before it come from a price-features-only model (LightGBM's native NaN handling lets it
# train regardless, per alpha.py's docstring), silently averaged into every headline number
# unless explicitly split out here.
FUNDAMENTALS_COVERAGE_START = pd.Timestamp("2011-01-31")


def window_bounds(window: str) -> tuple:
    """(truncate_end, eval_start) per plan Phase V.0c/V.3a-b -- deliberately
    two DIFFERENT levers, not one cutoff applied twice:

    `truncate_end`: restricts what the model is ALLOWED TO SEE. Used by
    train/trainval to simulate "as of" a past date -- nothing after this
    date exists anywhere in the run, so universe construction, walk-forward
    training, AND backtest trading all happen blind to the future. This is
    what a real train/val-only design search needs.

    `eval_start`: restricts which dates get SCORED, without touching what
    the model trained on. Used by test -- the walk-forward model keeps
    training continuously through the full history exactly like production
    (`alpha.walk_forward_predict` already retrains at every rebalance date),
    and only the reported metrics are sliced to the held-out tail. An
    artificially-blinded model would understate what a real frozen-test
    deployment looks like.

    `full` (default): both None -- today's original behavior, unrestricted,
    for exploratory diagnostics that intentionally want the whole sample
    (e.g. the Phase V deflated-Sharpe checks already run).
    """
    if window == "full":
        return None, None
    split = json.loads(SPLIT_CONFIG_PATH.read_text())
    train_end, val_end = pd.Timestamp(split["train_end"]), pd.Timestamp(split["val_end"])
    if window == "train":
        return train_end, None
    if window == "trainval":
        return val_end, None
    if window == "test":
        return None, val_end
    raise ValueError(f"unknown window {window!r}, choose from {WINDOW_CHOICES}")


def make_alpha_weighted_fn(preds_by_date: dict, top_frac: float = 0.5, hold_frac: float | None = None,
                            exposure_by_date: dict | None = None):
    """The honest baseline bar (plan Phase 1.1): quintile/top-half sort,
    equal-weight, PLUS a no-trade band -- a name already held stays as long
    as it's still within the looser `hold_frac` cut, not just the tighter
    `top_frac` entry cut. Without this, a name bouncing between rank 49 and
    51 out of 100 gets fully sold and rebought every quarter on pure ranking
    noise near the cutoff -- exactly the churn a Buffett-shaped "own the
    best, hold" bar should not have. `hold_frac=None` (default) reproduces
    the pre-band behavior (hold_frac == top_frac, no slack).
    Falls back to equal-weight on any date without a prediction yet (early
    history, before the model has min_train_rows).

    `exposure_by_date` (plan Phase 3.1c): optional {date: equity_cap} from
    contrarian.equity_exposure(). Without it (default), weights sum to 1.0 --
    this strategy is a pure stock-picker, never holds cash by construction.
    With it, the chosen set is scaled by the cap and the residual falls
    through to cash via run_backtest's "weights need not sum to 1" convention
    (backtest.py) -- no optimizer/cvxpy needed, just a scalar on top of the
    existing equal-weight sort."""
    hold_frac = top_frac if hold_frac is None else hold_frac
    def weights_fn(date, uni, state):
        preds = preds_by_date.get(date, {})
        ranked = sorted((t for t in uni if t in preds), key=lambda t: preds[t], reverse=True)
        if not ranked:
            n = len(uni)
            return {t: 1.0 / n for t in uni} if n else {}
        n_buy = max(1, int(len(ranked) * top_frac))
        n_hold = max(n_buy, int(len(ranked) * hold_frac))
        buy_set = set(ranked[:n_buy])
        hold_set = set(ranked[:n_hold])
        currently_held = set(state.get("prev_weights", {}))
        chosen = buy_set | (currently_held & hold_set)
        exposure = exposure_by_date.get(date, 1.0) if exposure_by_date else 1.0
        w = exposure / len(chosen)
        return {t: w for t in chosen}
    return weights_fn


def main(top_n: int = 50, horizon_td: int = 252, top_frac: float = 0.6, hold_frac: float = 0.75,
         use_exposure: bool = False, n_trials: int = 16, window: str = "full"):
    truncate_end, eval_start = window_bounds(window)

    print("Loading dataset (full feature set)...")
    base_cols = ["ticker", "trade_date", "adj_close", "traded_amount"]
    # net_income/market_cap/reference_date aren't in the ML feature keep-list
    # (feature_columns()) but are needed for the contrarian exposure signal.
    exposure_cols = ["net_income", "market_cap", "reference_date"]
    all_cols = sorted(set(base_cols) | set(feature_columns(include_sector=False)) | set(exposure_cols))
    df = pd.read_parquet(OUTPUT_PATH, columns=all_cols)

    if truncate_end is not None:
        df = df[df["trade_date"] <= truncate_end]
        print(f"  --window={window}: truncated input to <= {truncate_end.date()} "
              f"({len(df)} rows) -- simulates design-time, nothing after this date exists yet")

    print(f"Building point-in-time liquid universe (top_n={top_n})...")
    membership = universe.liquid_universe(df[["ticker", "trade_date", "traded_amount"]], top_n=top_n)
    df = universe.restrict_to_universe(df, membership)
    print(f"  restricted to {len(df)} rows, {df['ticker'].nunique()} tickers ever in the universe")

    print(f"Building the {horizon_td}-day forward-excess-return label...")
    df["label"] = forward_excess_return(df, horizon_td=horizon_td)

    reb_dates = universe.rebalance_dates(membership)
    print(f"Walk-forward training/prediction across {len(reb_dates)} rebalance dates...")
    preds = alpha.walk_forward_predict(df, reb_dates, horizon_td=horizon_td)  # n_estimators: alpha.DEFAULT_N_ESTIMATORS
    print(f"  {preds['date'].nunique()} dates produced a prediction "
          f"(first {len(reb_dates) - preds['date'].nunique()} skipped -- not enough training history yet)")

    # Plan V.1b: before the first real prediction, make_alpha_weighted_fn falls back to
    # plain equal-weight, so the alpha-sort is LITERALLY the EW baseline on those dates --
    # active-vs-EW is exactly zero there, diluting info-ratio/DSR by counting "no strategy
    # exists yet" as "strategy added nothing." Fold this into the same eval floor as the
    # --window restriction (whichever is later wins) rather than a second, separate cutoff.
    first_pred_date = preds["date"].min() if not preds.empty else None
    floors = [d for d in (eval_start, first_pred_date) if d is not None]
    restrict_start = max(floors) if floors else None
    if first_pred_date is not None and (eval_start is None or first_pred_date > eval_start):
        print(f"  first real prediction at {first_pred_date.date()} -- metrics restricted from "
              f"there (pre-prediction dates are an EW clone by construction, not strategy skill)")

    ic = alpha.rank_ic(preds, df)
    if restrict_start is not None:
        ic = ic[ic.index >= restrict_start]
        print(f"  rank-IC restricted to dates >= {restrict_start.date()}")
    print("\n=== Out-of-sample rank-IC ===")
    print(f"mean={ic.mean():.4f}  median={ic.median():.4f}  "
          f"fraction of dates with IC>0={float((ic > 0).mean()):.2f}  n_dates={ic.notna().sum()}")

    # Plan V.1a: is the headline number understating a design that actually works once
    # fundamentals exist? Split, don't just flag -- the earlier full-sample numbers blend a
    # price-only era in with the era the model was actually designed for.
    ic_pre = ic[ic.index < FUNDAMENTALS_COVERAGE_START]
    ic_post = ic[ic.index >= FUNDAMENTALS_COVERAGE_START]
    if len(ic_pre) and len(ic_post):
        print(f"  pre-{FUNDAMENTALS_COVERAGE_START.date()} (price-only features): "
              f"mean={ic_pre.mean():.4f}  n_dates={ic_pre.notna().sum()}")
        print(f"  post-{FUNDAMENTALS_COVERAGE_START.date()} (fundamentals available): "
              f"mean={ic_post.mean():.4f}  n_dates={ic_post.notna().sum()}")

    # Overlap-corrected t-stat (plan V.1c): consecutive rebalance dates' 252-trading-day
    # label windows overlap heavily on a quarterly calendar (~4 quarters per horizon), so
    # the naive t-stat below overstates significance by treating each date as independent.
    # max_lag derived from the rebalance calendar's own spacing, not hardcoded to "quarterly"
    # -- calendar-day gap converted to trading days (252/365), a deliberately simple
    # approximation (same convention as alpha.py's own purge-boundary date math).
    ic_dates = pd.Series(ic.dropna().index).sort_values()
    if len(ic_dates) < 2:
        print("(too few rank-IC dates in this window to compute a t-stat)")
    else:
        median_gap_days = ic_dates.diff().dt.days.median()
        gap_trading_days = median_gap_days * 252 / 365
        max_lag = max(0, round(horizon_td / gap_trading_days) - 1)
        naive_t = ic.mean() / (ic.std(ddof=1) / np.sqrt(ic.notna().sum()))
        nw_t = newey_west_tstat(ic, max_lag=max_lag)
        print(f"naive t-stat={naive_t:.2f}  Newey-West t-stat (max_lag={max_lag}, ~{max_lag + 1} "
              f"overlapping periods/horizon)={nw_t:.2f}")

    preds_by_date = {
        d: g.set_index("ticker")["alpha"].to_dict() for d, g in preds.groupby("date")
    }

    exposure_by_date = None
    exposure_label = ""
    if use_exposure:
        print("Computing contrarian equity exposure (smoothed earnings yield vs SELIC)...")
        df = contrarian.add_smoothed_earnings_yield(df)
        exposure_by_date = contrarian.equity_exposure(df, reb_dates, col=contrarian.SMOOTHED_SIGNAL_COL)
        exp = pd.Series(exposure_by_date)
        print(f"  exposure: {exp.min():.0%}..{exp.max():.0%} (median {exp.median():.0%}) "
              f"across {len(exp)} rebalances")
        exposure_label = ", contrarian cash overlay"

    alpha_fn = make_alpha_weighted_fn(preds_by_date, top_frac=top_frac, hold_frac=hold_frac,
                                       exposure_by_date=exposure_by_date)

    # cdi is already a loaded column (feature_columns()'s MACRO group) -- no need to
    # re-open OUTPUT_PATH for a column already sitting in df.
    cdi = df[["trade_date", "cdi"]].drop_duplicates().sort_values("trade_date")
    prices = df[["ticker", "trade_date", "adj_close"]]

    print("\nRunning equal-weight baseline (same universe/dates/costs)...")
    eq_curve, eq_log = run_backtest(prices, cdi, membership, equal_weight_fn)
    print(f"Running alpha-sort (honest baseline bar: top {top_frac:.0%} buy / "
          f"hold to {hold_frac:.0%}, equal-weight{exposure_label})...")
    alpha_curve, alpha_log = run_backtest(prices, cdi, membership, alpha_fn)

    if restrict_start is not None:
        n_before = len(alpha_curve)
        eq_curve = eq_curve[eq_curve.index >= restrict_start]
        eq_log = eq_log[eq_log["date"] >= restrict_start]
        alpha_curve = alpha_curve[alpha_curve.index >= restrict_start]
        alpha_log = alpha_log[alpha_log["date"] >= restrict_start]
        print(f"  evaluation restricted to dates >= {restrict_start.date()} "
              f"({len(alpha_curve)} of {n_before} days) -- training was NOT restricted, only "
              f"which dates get scored (the model trained continuously through the full history, "
              f"exactly as it would in production)")

    cdi_series = cdi.set_index("trade_date")["cdi"]
    eq_returns = eq_curve.pct_change().dropna()
    print_report("Equal-weight baseline", full_report(eq_curve, eq_log, cdi_daily=cdi_series))
    print_report(f"Alpha-sort (top {top_frac:.0%} buy / hold {hold_frac:.0%}{exposure_label})",
                 full_report(alpha_curve, alpha_log, cdi_daily=cdi_series, benchmark_returns=eq_returns,
                             n_trials=n_trials))

    # Plan V.1a, active-return side: same pre/post-2011 split, now on the actual
    # construction (not just the raw signal) -- does the "beats EW" edge concentrate in
    # the era the model has real features for?
    active_all = (alpha_curve.pct_change() - eq_returns).dropna()
    active_pre = active_all[active_all.index < FUNDAMENTALS_COVERAGE_START]
    active_post = active_all[active_all.index >= FUNDAMENTALS_COVERAGE_START]
    if len(active_pre) and len(active_post):
        print(f"\n  active-return-vs-EW, pre-{FUNDAMENTALS_COVERAGE_START.date()}: "
              f"ann.mean={active_pre.mean() * 252:.2%}  n_days={len(active_pre)}")
        print(f"  active-return-vs-EW, post-{FUNDAMENTALS_COVERAGE_START.date()}: "
              f"ann.mean={active_post.mean() * 252:.2%}  n_days={len(active_post)}")
    # n_trials sensitivity (2026-07-26): the honest count above is an estimate of how many
    # distinct configs were compared across all Phase 1/3 sweeps before picking this one --
    # show how the deflated Sharpe moves if that count is off, rather than reporting one
    # falsely-precise number. Deflated on EXCESS-CDI returns, not raw returns: raw equity
    # returns clear "beats zero" almost by construction over a 26y positive-drift sample
    # (CDI/BOVA11 would too) -- the mandate-relevant question is whether the excess-CDI edge
    # (the actual Gate B/D metric) survives the same cherry-picking correction.
    # n_trials=1 alongside the honest count (2026-07-26, plan V.0d): PSR@1 answers "is the
    # edge real at all" (no search-bias correction); DSR@n_trials answers "does it survive
    # having tried n_trials configs" -- printing both decomposes the two separate questions
    # instead of only showing the harsher, already-corrected number.
    alpha_excess_cdi = (alpha_curve.pct_change() - cdi_series.reindex(alpha_curve.index).ffill() / 100).dropna()
    print(f"\n  Deflated Sharpe (on excess-CDI returns) sensitivity to n_trials estimate "
          f"({n_trials} counted from PLAN sweeps):")
    for nt in sorted({1, n_trials, 20, 25}):
        print(f"    n_trials={nt:<4}{deflated_sharpe_ratio(alpha_excess_cdi, n_trials=nt):>10.3f}")

    # Same check on the active-vs-EW series (2026-07-26): the most literal test of "is the
    # construction adding real skill on top of just being a boring EW investor" -- isolates
    # our top_frac/hold_frac/overlay choices from generic Brazilian-equity beta, which both
    # the raw-return and excess-CDI checks above still partly carry.
    alpha_active_vs_ew = (alpha_curve.pct_change() - eq_returns).dropna()
    print(f"\n  Deflated Sharpe (on active-return-vs-EW) sensitivity to n_trials estimate "
          f"({n_trials} counted from PLAN sweeps):")
    for nt in sorted({1, n_trials, 20, 25}):
        print(f"    n_trials={nt:<4}{deflated_sharpe_ratio(alpha_active_vs_ew, n_trials=nt):>10.3f}")

    # Persist the run (plan V.0a/b): every DSR check to date has cost a full walk-forward
    # retrain because nothing survived past the in-memory run. Save once, re-analyze many
    # times; the trial log also makes n_trials a counted fact for the NEXT deflation instead
    # of a hand re-count of this document's sweep tables.
    run_config = {
        "top_n": top_n, "horizon_td": horizon_td, "top_frac": top_frac, "hold_frac": hold_frac,
        "use_exposure": use_exposure, "window": window,
    }
    run_path = artifacts.save_run(run_config, alpha_curve=alpha_curve, alpha_log=alpha_log,
                                   eq_curve=eq_curve, eq_log=eq_log, cdi_series=cdi_series)
    print(f"\nRun artifacts saved to {run_path}")
    artifacts.append_trial_log(run_config, {
        "excess_cdi_sharpe": excess_over_cdi_sharpe(alpha_curve.pct_change().dropna(), cdi_series),
        "info_ratio_vs_ew": information_ratio(alpha_curve.pct_change().dropna(), eq_returns),
        "rank_ic_mean": float(ic.mean()),
    })

    print("\nBuilding dashboard...")
    bova = pd.read_parquet(PRICES_DIR / "BOVA11.parquet", columns=["trade_date", "adj_close"])
    bova_curve = buy_and_hold_curve(bova.set_index("trade_date")["adj_close"].reindex(alpha_curve.index).ffill())
    cdi_only_curve = cdi_curve(cdi[cdi["trade_date"].isin(alpha_curve.index)])
    fig = build_dashboard(alpha_curve, alpha_log, eq_curve, bova_curve, cdi_only_curve, top_frac, hold_frac)
    path = save_dashboard(fig)
    print(f"Dashboard saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--horizon-td", type=int, default=252)
    parser.add_argument("--top-frac", type=float, default=0.6,
                         help="buy cutoff, e.g. 0.6 = top 60%% (2026-07-25 sweep: strictly beats 0.5 "
                              "on every metric -- see PORTFOLIO_IMPROVEMENT_PLAN.md)")
    parser.add_argument("--hold-frac", type=float, default=0.75,
                         help="no-trade band: an already-held name stays until it drops below this "
                              "looser cutoff, not just below --top-frac")
    parser.add_argument("--use-exposure", action="store_true",
                         help="scale weights by contrarian.equity_exposure() (smoothed earnings yield "
                              "vs SELIC), leaving the residual in cash -- off by default, this strategy "
                              "is a pure stock-picker unless opted in (plan Phase 3.1c)")
    parser.add_argument("--n-trials", type=int, default=16,
                         help="count of distinct configs compared across all Phase 1/3 sweeps before "
                              "picking top_frac/hold_frac/the overlay -- corrects deflated Sharpe for "
                              "that selection bias (2026-07-26 methodology check, see PLAN)")
    parser.add_argument("--window", choices=WINDOW_CHOICES, default="full",
                         help="train/trainval TRUNCATE the input data (design-time blindness, for a "
                              "leak-free parameter search); test does NOT truncate training (the model "
                              "keeps learning through the full history, like production) but restricts "
                              "which dates get SCORED to the held-out tail; full (default) is today's "
                              "original unrestricted behavior (plan Phase V.0c)")
    args = parser.parse_args()
    main(top_n=args.top_n, horizon_td=args.horizon_td, top_frac=args.top_frac, hold_frac=args.hold_frac,
         use_exposure=args.use_exposure, n_trials=args.n_trials, window=args.window)
