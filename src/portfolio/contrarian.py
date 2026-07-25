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
    z_t      = causal expanding z-score of signal_t (uses only dates <= t)
    exposure = clip(base + k * z_t, floor, ceil)     -> total equity weight cap

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


def equity_exposure(df: pd.DataFrame, reb_dates: pd.DatetimeIndex,
                     col: str = SIGNAL_COL, base: float = 0.75, k: float = 0.15,
                     floor: float = 0.50, ceil: float = 1.00,
                     min_periods: int = 12) -> dict:
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

    Causal by construction: the expanding mean/std at date t include t's own
    value but nothing after it. `k` and `base` are the only knobs; tune `k`
    against realized exposure swings, not in-sample Sharpe.
    """
    daily = df.groupby("trade_date")[col].median().sort_index()
    # value known as of each rebalance date (last observation <= that date)
    erp = daily.reindex(daily.index.union(reb_dates)).ffill().reindex(reb_dates)
    mu = erp.expanding(min_periods=min_periods).mean()
    sd = erp.expanding(min_periods=min_periods).std()
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
