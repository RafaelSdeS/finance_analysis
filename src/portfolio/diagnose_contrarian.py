"""
diagnose_contrarian.py -- Phase 3 pre-work / validation for contrarian.py's
smoothed-earnings-yield fix (PORTFOLIO_IMPROVEMENT_PLAN.md Phase 3.1a). No
alpha training needed here (the overlay is a pure macro/valuation rule,
independent of stock selection) so this runs in seconds, not minutes --
compares the raw (trailing point-in-time) and smoothed (multi-year CAPE-
style) earnings-yield signals side by side, across the crisis windows the
raw signal got wrong (recession_2015_16 pinned at the floor, near-ATH not
low enough).

Run: python -m src.portfolio.diagnose_contrarian [--top-n 50]
"""

import argparse

import pandas as pd

from src.build_dataset.paths import OUTPUT_PATH, PRICES_DIR
from src.portfolio import contrarian, universe
from src.portfolio.metrics import CRISIS_WINDOWS
from src.portfolio.run_alpha_diagnostic import WINDOW_CHOICES, window_bounds


def main(top_n: int = 50, window: str = "full"):
    # No ML training here (the overlay is a pure macro/valuation rule, no
    # walk-forward model to protect from seeing the future) -- so unlike
    # run_alpha_diagnostic.py, train/trainval/test all collapse to the same
    # mechanism, a plain date filter on `df` (plan Phase V.0c).
    truncate_end, eval_start = window_bounds(window)

    print("Loading dataset (valuation + net_income/market_cap -- no features, no alpha)...")
    cols = ["ticker", "trade_date", "traded_amount", "reference_date", "pl", "selic",
            "earnings_yield", "earnings_yield_vs_selic", "net_income", "market_cap"]
    df = pd.read_parquet(OUTPUT_PATH, columns=cols)

    if truncate_end is not None:
        df = df[df["trade_date"] <= truncate_end]
        print(f"  --window={window}: restricted to <= {truncate_end.date()} ({len(df)} rows)")
    elif eval_start is not None:
        df = df[df["trade_date"] >= eval_start]
        print(f"  --window={window}: restricted to >= {eval_start.date()} ({len(df)} rows)")

    print(f"Building point-in-time liquid universe (top_n={top_n})...")
    membership = universe.liquid_universe(df[["ticker", "trade_date", "traded_amount"]], top_n=top_n)
    df = universe.restrict_to_universe(df, membership)
    reb_dates = universe.rebalance_dates(membership)

    print("Computing smoothed (multi-year) earnings yield...")
    df = contrarian.add_smoothed_earnings_yield(df)

    exposure_raw = pd.Series(contrarian.equity_exposure(df, reb_dates, col=contrarian.SIGNAL_COL)).sort_index()
    exposure_smoothed = pd.Series(
        contrarian.equity_exposure(df, reb_dates, col=contrarian.SMOOTHED_SIGNAL_COL)).sort_index()

    daily = df.groupby("trade_date")[["earnings_yield", "earnings_yield_smoothed"]].median()

    bova = pd.read_parquet(PRICES_DIR / "BOVA11.parquet", columns=["trade_date", "adj_close"]).set_index("trade_date")["adj_close"]
    bova_dd = (bova / bova.cummax() - 1).reindex(exposure_raw.index, method="ffill")

    print("\n=== Raw vs smoothed earnings-yield signal, by crisis window ===")
    print(f"{'window':<20}{'BOVA11 dd':>11}{'exp_raw':>10}{'exp_smooth':>12}"
          f"{'ey_raw':>9}{'ey_smooth':>11}")

    def _row(name, exp_mask_r, exp_mask_s, daily_mask, dd_mask):
        exp_r, exp_s = exposure_raw[exp_mask_r], exposure_smoothed[exp_mask_s]
        daily_w, dd_w = daily[daily_mask], bova_dd[dd_mask]
        if len(exp_r) == 0 or len(daily_w) == 0:
            print(f"{name:<20}  (no data)")
            return
        print(f"{name:<20}{dd_w.mean():>10.1%}{exp_r.mean():>9.1%}{exp_s.mean():>11.1%}"
              f"{daily_w['earnings_yield'].mean():>8.1%}{daily_w['earnings_yield_smoothed'].mean():>10.1%}")

    for name, (start, end) in CRISIS_WINDOWS.items():
        in_window = lambda idx: (idx >= start) & (idx <= end)  # noqa: E731
        _row(name, in_window(exposure_raw.index), in_window(exposure_smoothed.index),
             in_window(daily.index), in_window(bova_dd.index))

    near_ath = bova_dd >= -0.01
    if near_ath.any():
        ath_dates = bova_dd[near_ath].index
        _row("near BOVA11 ATH", exposure_raw.index.isin(ath_dates), exposure_smoothed.index.isin(ath_dates),
             daily.index.isin(ath_dates), bova_dd.index.isin(ath_dates))

    print(f"\n{'full sample mean':<20}{'':>11}{'':>10}{'':>12}"
          f"{daily['earnings_yield'].mean():>8.1%}{daily['earnings_yield_smoothed'].mean():>10.1%}")
    print("\nRead: want exp_smooth HIGHER than exp_raw during recession_2015_16 (was pinned at the 50%")
    print("floor) and LOWER than exp_raw near BOVA11 ATH (was 82.7%, too high). ey_smooth should sit")
    print("closer to (or above) the full-sample mean during the recession if the multi-year average is")
    print("successfully smoothing over the single-quarter earnings collapse that value-trapped ey_raw.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--window", choices=WINDOW_CHOICES, default="full",
                         help="restrict to a split_config.json era (plan Phase V.0c); full (default) "
                              "is today's original unrestricted behavior")
    args = parser.parse_args()
    main(top_n=args.top_n, window=args.window)
