"""
contrarian.py -- Layer 2 of the two-layer design: the "buy at the sound of
cannons, sell at the sound of violins" market-timing overlay.

NOT a learned model -- deliberately. With ~3 crisis episodes in the sample
(2008, 2015-16, covid) a learned regime classifier is fit on ~3 data points
and overfits by construction. The ML stays on cross-sectional stock SELECTION
(data-rich); this layer is a simple, 1-parameter economic rule (data-poor):

    signal_t = median over the current universe of `earnings_yield_vs_selic`
               (the aggregate equity risk premium -- SELIC ~= CDI in Brazil,
                so this is stocks' earnings yield vs the ~riskless carry)
    z_t      = causal ROLLING z-score of signal_t (trailing `window` periods,
               not expanding-since-inception -- see 2026-07-25 fix below)
    exposure = clip(base + k * z_t, floor, ceil)     -> total equity weight cap

    Rolling, not expanding (fixed 2026-07-25): an expanding mean/std drifts
    its own reference along with a slow, sustained trend in signal_t, muting
    how unusual a multi-year decline looks by the time it matters -- found
    empirically investigating why the recession_2015-16 exposure stayed
    near-neutral despite the (smoothed) spread falling steadily from -2.6%
    (2012-12) to -10.6% (2015-06): the expanding mean chased the decline
    down with it. A trailing window judges each point against its recent
    past instead of its entire history since 2011. See
    PORTFOLIO_IMPROVEMENT_PLAN.md Phase 3.1b for the full diagnosis.

High ERP (stocks cheap vs the carry) -> z>0 -> more equity (cannons).
ERP compresses (stocks expensive)     -> z<0 -> shift to CDI  (violins).

"Gentle band" per the user's choice: base=0.75, floor=0.50, ceil=1.00 -- the
book never goes below 50% equity and never levers past 100%. `k` sets how
hard it leans; k=0.15 saturates the band around +/-1.7 sigma (slow, robust).
The residual (1 - exposure) falls to CDI via the optimizer's cash asset.
"""

import numpy as np
import pandas as pd

SIGNAL_COL = "earnings_yield_vs_selic"

SMOOTHED_SIGNAL_COL = "earnings_yield_vs_selic_smoothed"
SMOOTHED_WINDOW_QUARTERS = 20   # 5y of filings -- same convention as
SMOOTHED_MIN_QUARTERS = 8       # build_dataset/features.py's FUND_ZHIST_*


def add_smoothed_earnings_yield(df: pd.DataFrame, window_quarters: int = SMOOTHED_WINDOW_QUARTERS,
                                 min_quarters: int = SMOOTHED_MIN_QUARTERS) -> pd.DataFrame:
    """CAPE-style fix for the trailing-earnings lag (2026-07-25 finding):
    point-in-time `earnings_yield` (=1/pl) uses only the LAST filed
    quarter's net income, so during an earnings recession (net income
    falling as fast as or faster than price) P/E never re-rates 'cheap'
    even as the market crashes -- confirmed empirically in
    recession_2015_16 (earn_yield 1.7%, BELOW its 3.1% full-sample mean,
    despite a 31% BOVA11 drawdown; see PORTFOLIO_IMPROVEMENT_PLAN.md Phase
    3.1). Fix: average net_income over a trailing multi-year window of
    FILINGS (not calendar days -- fundamentals are quarterly step
    functions forward-filled ~63x redundantly across daily rows; same
    dedup-then-roll-then-map-back pattern as
    build_dataset/features.py::compute_history_relative_features) before
    dividing by the already daily-re-anchored market_cap. Needs
    ticker/trade_date/reference_date/net_income/market_cap/selic in df.
    Adds `earnings_yield_smoothed` and `earnings_yield_vs_selic_smoothed`.

    Units gotcha (found 2026-07-25, undocumented by the vendor): `net_income`
    is reported in R$ thousands while `market_cap`/`lpa`/`shares_outstanding`
    are raw BRL -- a ~1000x mismatch. Verified two ways on a real row
    (PETR4, 2026-07-10): `earnings_yield / (net_income/market_cap) = 1000.14`
    and `(lpa * shares_outstanding) / net_income = 1000.35` (the ~0.1-0.35%
    residual is `market_cap`/`pl` being re-anchored to the CURRENT close
    while net_income/lpa are the last FILED quarter's values, not a units
    error). Never surfaced before because the only existing net_income
    consumer (`earnings_growth_yoy` in build_dataset/features.py) is a
    same-column YoY self-ratio where the units cancel out.
    """
    result = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("trade_date").copy()
        q = g.drop_duplicates("reference_date").set_index("reference_date").sort_index()
        avg_net_income = q["net_income"].rolling(window_quarters, min_periods=min_quarters).mean()
        g["net_income_smoothed"] = g["reference_date"].map(avg_net_income)
        result.append(g)
    df = pd.concat(result, ignore_index=True)

    df["earnings_yield_smoothed"] = (df["net_income_smoothed"] * 1000) / df["market_cap"]
    selic_annualized = (1 + df["selic"] / 100) ** 252 - 1
    df[SMOOTHED_SIGNAL_COL] = df["earnings_yield_smoothed"] - selic_annualized
    return df


def equity_exposure(df: pd.DataFrame, reb_dates: pd.DatetimeIndex,
                     col: str = SIGNAL_COL, base: float = 0.75, k: float = 0.15,
                     floor: float = 0.50, ceil: float = 1.00,
                     window: int = 20, min_periods: int = 12) -> dict:
    """
    df: the universe-restricted dataset; needs `trade_date` and `col`. The
        cross-`trade_date` median of `col` is the aggregate ERP each day
        (df is already restricted to the point-in-time universe upstream, so
        the median is over universe members only -- no separate lookup).
    reb_dates: the rebalance calendar (universe.rebalance_dates()).
    Returns {rebalance_date: equity_exposure_cap in [floor, ceil]} for use as
        optimizer.solve(max_equity=...). Warm-up dates (< min_periods of ERP
        history) get `base` -- a neutral cap, same leading-prefix convention
        as every other rolling feature in the pipeline.

    `window`/`min_periods` are in REBALANCE PERIODS (quarterly by default,
    so window=20 ~= 5y, matching contrarian.py's own SMOOTHED_WINDOW_QUARTERS).
    Rolling, not expanding (2026-07-25 fix -- see module docstring): a fixed
    trailing window judges each point against its recent past, not its
    entire history since 2011, so it actually reacts to a sustained
    multi-year trend instead of drifting its reference mean along with it.

    Causal by construction either way: at date t the window only ever
    includes dates <= t. `k` and `base` are the only knobs; tune `k` against
    realized exposure swings, not in-sample Sharpe.
    """
    daily = df.groupby("trade_date")[col].median().sort_index()
    # value known as of each rebalance date (last observation <= that date)
    erp = daily.reindex(daily.index.union(reb_dates)).ffill().reindex(reb_dates)
    roll = erp.rolling(window=window, min_periods=min_periods)
    mu = roll.mean()
    sd = roll.std()
    z = (erp - mu) / sd.replace(0.0, np.nan)
    exposure = (base + k * z).clip(floor, ceil).fillna(base)
    return exposure.to_dict()


if __name__ == "__main__":
    # synthetic ERP that ramps up then down; verify direction, bounds, causality
    dates = pd.date_range("2000-01-31", periods=60, freq="ME")
    vals = np.concatenate([np.linspace(-2, 2, 30), np.linspace(2, -2, 30)])
    df = pd.DataFrame({"trade_date": dates, SIGNAL_COL: vals})
    reb = pd.DatetimeIndex(dates)

    e = pd.Series(equity_exposure(df, reb)).sort_index()
    assert e.between(0.50, 1.00).all(), "exposure escaped the [floor, ceil] band"
    # ERP above its running mean (rising half) -> more equity than the falling half
    assert e.iloc[25] > e.iloc[50], "cheaper (higher ERP) must map to more equity"
    # causality: recomputing on truncated history gives the identical value at the cut
    e_trunc = pd.Series(equity_exposure(df.iloc[:40], reb[:40]))
    assert abs(e_trunc[dates[39]] - e[dates[39]]) < 1e-12, "exposure at t used future data"
    print("contrarian self-check OK |", "exposure range:",
          round(e.min(), 3), "..", round(e.max(), 3))

    # add_smoothed_earnings_yield: one ticker, 12 quarterly filings, rising
    # net_income, constant market_cap/selic -- window collapses to a plain
    # rolling mean, easy to hand-verify.
    q_dates = pd.date_range("2015-03-31", periods=12, freq="QE")
    net_income = pd.Series(range(1, 13), dtype=float) * 100  # 100, 200, .. 1200
    ticker_df = pd.DataFrame({
        "ticker": "TEST3", "trade_date": q_dates, "reference_date": q_dates,
        "net_income": net_income, "market_cap": 10_000.0, "selic": 0.05,
    })
    smoothed = add_smoothed_earnings_yield(ticker_df, window_quarters=4, min_quarters=2)
    # last 4 quarters at the final row: (900+1000+1100+1200)/4 = 1050
    assert abs(smoothed["net_income_smoothed"].iloc[-1] - 1050.0) < 1e-9, \
        "trailing 4Q rolling mean of net_income wrong"
    assert smoothed["net_income_smoothed"].iloc[:1].isna().all(), \
        "single quarter (< min_quarters=2) should be NaN, not a fabricated average"
    # causal: recomputing on a truncated history matches the full run at the cut
    smoothed_trunc = add_smoothed_earnings_yield(ticker_df.iloc[:8], window_quarters=4, min_quarters=2)
    assert abs(smoothed_trunc["net_income_smoothed"].iloc[-1]
               - smoothed["net_income_smoothed"].iloc[7]) < 1e-9, \
        "smoothed earnings at t used future filings"
    print("add_smoothed_earnings_yield self-check OK | smoothed net_income (last row):",
          smoothed["net_income_smoothed"].iloc[-1])
