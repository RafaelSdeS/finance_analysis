"""
anchor.py — H3 Design §b: fitted, risk-based portfolio anchor (min-variance /
risk-parity), reusing risk_portfolios.py's estimate_cov/min_variance_weights/
risk_parity_weights verbatim -- they were built and validated for the
R-series and need no changes here.

Anchor TYPE (min-variance vs risk-parity) is selected by pre-2024
anchor-alone Sharpe/IR in milestone_h3.py, never hand-picked -- this module
only provides the per-date weight computation for a GIVEN type.

Step 0's leakage guardrail lives here: the trailing daily-return window
feeding estimate_cov() for a decision_date T strictly ends at T-1. H3's
weights are executed at T's close (same convention milestone_h2's
precomputed-weight backtest already uses), so the covariance that sizes
that trade must never see T's own return.
"""

import pandas as pd

from ..rl_agent.config import RiskConfig
from ..rl_agent.risk_portfolios import estimate_cov, min_variance_weights, risk_parity_weights

ANCHOR_TYPES = ("min_variance", "risk_parity")


def _trailing_returns_asof(prices_wide: pd.DataFrame, decision_date: pd.Timestamp,
                            tickers: list, lookback: int) -> pd.DataFrame:
    """Simple daily returns for `tickers`, strictly ending at decision_date's
    PREVIOUS trading day. decision_date's own return (its close vs the prior
    close) is excluded by construction -- it isn't known until the same
    close the anchor's trade executes at, so including it would be a
    lookahead leak (Step 0)."""
    calendar = prices_wide.index
    pos = calendar.searchsorted(decision_date)
    end = pos - 1  # last return usable: the one ending at T-1
    start = end - lookback
    if start < 0:
        return pd.DataFrame(columns=tickers)
    window = prices_wide.iloc[start:end + 1][tickers]
    return window.pct_change().iloc[1:]


def _eligible(returns: pd.DataFrame, min_history_frac: float) -> list:
    """Tickers with >= min_history_frac real (non-NaN) daily returns in the
    trailing window. prices_wide is a plain pivot of raw per-ticker data
    (unlike rl_agent's PricePanel global space, it is never flat-backfilled
    across a ticker's pre-listing days), so a direct notna() count is enough
    -- no backfill-detection heuristic needed, unlike
    risk_portfolios.eligible_mask."""
    frac_real = returns.notna().mean(axis=0)
    return frac_real[frac_real >= min_history_frac].index.tolist()


def anchor_weights_by_date(panel: pd.DataFrame, prices_wide: pd.DataFrame, anchor_type: str,
                            cfg: RiskConfig = RiskConfig()) -> pd.DataFrame:
    """Per decision_date, per-ticker anchor weight (long format: decision_date,
    ticker, weight) for `anchor_type`, solved from that date's trailing daily
    covariance (Ledoit-Wolf, Step 0's T-1 cutoff). Tickers without enough
    trailing history (cfg.min_history_frac) get weight 0, not a dropped row
    -- downstream target-weight construction expects a dense per-date vector
    over the full panel universe, matching milestone_h2.add_anchor_columns'
    contract."""
    if anchor_type not in ANCHOR_TYPES:
        raise ValueError(f"unknown anchor_type: {anchor_type!r} (available: {ANCHOR_TYPES})")

    rows = []
    for date, g in panel.groupby("decision_date"):
        tickers = list(g["ticker"])
        avail = [t for t in tickers if t in prices_wide.columns]
        returns = _trailing_returns_asof(prices_wide, date, avail, cfg.lookback)
        eligible = _eligible(returns, cfg.min_history_frac)

        w_by_ticker = dict.fromkeys(tickers, 0.0)
        if len(eligible) >= 2:
            r = returns[eligible].fillna(0.0).to_numpy()
            cov = estimate_cov(r, cfg)
            if anchor_type == "min_variance":
                w_eq, _ = min_variance_weights(cov, max_weight=cfg.max_weight)
            else:
                w_eq, _ = risk_parity_weights(cov)
            for t, w in zip(eligible, w_eq):
                w_by_ticker[t] = float(w)

        rows.append(pd.DataFrame({
            "decision_date": date,
            "ticker": tickers,
            "weight": [w_by_ticker[t] for t in tickers],
        }))
    return pd.concat(rows, ignore_index=True)


def _demo() -> None:
    """Runnable self-check (not exercising real data): a 5-ticker, 300-day
    synthetic panel with one high-variance, one low-variance, and one
    barely-seasoned (should be excluded) name -- checks the T-1 cutoff
    excludes the decision date's own return, weights land on the simplex,
    and min-variance tilts away from the high-variance name relative to
    risk-parity."""
    import numpy as np

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=300)
    tickers = ["LOWVOL", "HIVOL", "MID1", "MID2", "NEWCO"]
    prices = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    prices["LOWVOL"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, len(dates))))
    prices["HIVOL"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(dates))))
    prices["MID1"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, len(dates))))
    prices["MID2"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, len(dates))))
    prices["NEWCO"] = np.nan
    prices.loc[dates[-10]:, "NEWCO"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 10)))

    decision_date = dates[250]
    panel = pd.DataFrame({"decision_date": [decision_date] * 5, "ticker": tickers})
    cfg = RiskConfig(lookback=126, min_history_frac=0.8)

    # T-1 cutoff: the return window must exclude decision_date's own return.
    ret = _trailing_returns_asof(prices, decision_date, tickers[:2], cfg.lookback)
    assert decision_date not in ret.index, "leakage: T's own return leaked into the covariance window"
    assert ret.index.max() < decision_date

    for atype in ANCHOR_TYPES:
        out = anchor_weights_by_date(panel, prices, atype, cfg)
        w = out.set_index("ticker")["weight"]
        assert abs(w.sum() - 1.0) < 1e-6, f"{atype}: weights don't sum to 1 ({w.sum()})"
        assert (w >= -1e-9).all(), f"{atype}: negative weight"
        assert w["NEWCO"] == 0.0, f"{atype}: under-seasoned name got nonzero weight"

    w_mv = anchor_weights_by_date(panel, prices, "min_variance", cfg).set_index("ticker")["weight"]
    w_rp = anchor_weights_by_date(panel, prices, "risk_parity", cfg).set_index("ticker")["weight"]
    assert w_mv["HIVOL"] < w_rp["HIVOL"], "min-variance should tilt away from the high-vol name harder than risk-parity"

    print("anchor.py self-check: OK")


if __name__ == "__main__":
    _demo()
