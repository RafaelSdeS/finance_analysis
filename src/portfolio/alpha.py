"""
alpha.py -- Stage A forecaster (proposal §2, §2.6): LightGBM regression on
the forward-excess-return-over-CDI label, retrained at each quarterly
rebalance on an EXPANDING, PURGED + EMBARGOED window (Lopez de Prado): a
training row is only used if its own label window has fully closed at
least `embargo_days` before the rebalance date being predicted for -- no
lookahead, plus a buffer against serial-correlation-induced optimism.

Native NaN handling (no imputation): LightGBM learns a default split
direction for missing values, so the raw feature matrix -- NaNs and all --
goes in directly. The info-age/NaN-explainer flags in features.py's
INFO_AGE_FLAGS group are exactly what let it use "this is missing" as a
signal in its own right.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.portfolio.features import feature_columns

DEFAULT_EMBARGO_DAYS = 21  # ~1 trading month -- absorbs residual serial correlation at the boundary

DEFAULT_N_ESTIMATORS = 50  # not 100 (LightGBM's own default) or 200: measured
# empirically (2026-07-24 diagnostic, last 20 rebalance dates) that
# out-of-sample rank-IC gets monotonically WORSE as n_estimators grows on
# this weak-signal, few-independent-period problem (0.112 at 50 vs 0.042 at
# 2000). Set once here so every caller gets the better setting by default
# instead of re-guessing it per script (run_alpha_diagnostic.py used to
# still hardcode 200 -- the worse setting -- after this was found).

# monotone_constraints (proposal §4.1): forward return should be non-decreasing
# in earnings_yield_vs_selic (the equity-vs-cash spread, the actual decision
# variable) -- a structural regularizer against the tree overfitting a raw
# SELIC-level threshold to one historical hiking episode. Every other
# feature is left unconstrained (0).
MONOTONE_FEATURE = "earnings_yield_vs_selic"


def _global_trading_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(df["trade_date"].unique()))


def _label_close_dates(trade_dates: pd.Series, global_dates: pd.DatetimeIndex,
                        horizon_td: int) -> pd.Series:
    """Safe-side approximate label-window-close date per row: the WHOLE-MARKET
    trading calendar's date `horizon_td` positions after each row's date.
    (labels.py itself uses each ticker's own row sequence for the label
    formula -- this uses the global calendar instead, purely for the purge
    boundary, a deliberately simple approximation; the embargo buffer
    absorbs the gap for any ticker with idiosyncratic trading halts.)
    NaT for rows too near the end of history (no valid close date yet).
    """
    pos = global_dates.searchsorted(trade_dates.to_numpy())
    close_pos = pos + horizon_td
    valid = close_pos < len(global_dates)
    closes = np.where(valid, global_dates[np.clip(close_pos, 0, len(global_dates) - 1)], np.datetime64("NaT"))
    return pd.Series(closes, index=trade_dates.index)


def _purge_embargo_mask(df: pd.DataFrame, as_of: pd.Timestamp, horizon_td: int,
                         embargo_days: int, close_dates: pd.Series | None = None) -> pd.Series:
    """Boolean mask: rows usable to train a model that predicts AT `as_of`
    -- their own label window must have fully closed at least `embargo_days`
    before `as_of` (purge + embargo, Lopez de Prado; see module docstring).

    `close_dates` (default None, computed internally): pass a precomputed
    _label_close_dates(...) result to skip recomputing it here -- it depends
    only on `df`/`horizon_td`, never on `as_of` (only `cutoff` does), so
    walk_forward_predict computes it ONCE outside its retrain loop instead of
    once per rebalance date (2026-08-13: ~65 dates in a typical run, each
    previously re-doing a full unique+sort of the whole panel plus a
    searchsorted over every row from scratch for an identical result)."""
    if close_dates is None:
        global_dates = _global_trading_dates(df)
        close_dates = _label_close_dates(df["trade_date"], global_dates, horizon_td)
    cutoff = as_of - pd.Timedelta(days=embargo_days)
    return df["label"].notna() & close_dates.notna() & (close_dates <= cutoff)


def fit(df: pd.DataFrame, as_of: pd.Timestamp, horizon_td: int,
        embargo_days: int = DEFAULT_EMBARGO_DAYS, min_train_rows: int = 500,
        close_dates: pd.Series | None = None, **lgb_params):
    """Fit a LightGBM regressor on the purged+embargoed training window as of
    `as_of`. `df` needs ticker/trade_date/label (labels.forward_excess_return)
    plus every column features.feature_columns() lists. Returns the fitted
    model, or None if there isn't yet `min_train_rows` of usable history.

    `close_dates` (default None): see _purge_embargo_mask -- walk_forward_
    predict passes its own precomputed result through here.
    """
    train_df = df.loc[_purge_embargo_mask(df, as_of, horizon_td, embargo_days, close_dates)]
    if len(train_df) < min_train_rows:
        return None

    cols = feature_columns(include_sector=False)
    monotone = [1 if c == MONOTONE_FEATURE else 0 for c in cols]
    params = {"objective": "regression", "verbosity": -1, "monotone_constraints": monotone,
              "n_estimators": DEFAULT_N_ESTIMATORS}
    params.update(lgb_params)

    model = lgb.LGBMRegressor(**params)
    model.fit(train_df[cols], train_df["label"])
    return model


def predict(model, df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Predicted alpha, indexed by ticker, for every row exactly at `as_of`.
    Empty if nothing has a row exactly at that date -- not every rebalance
    date necessarily has a matching row for every ticker after universe
    restriction (a ticker can qualify for a period without trading on that
    exact calendar date, e.g. a brief halt)."""
    predict_df = df[df["trade_date"] == as_of]
    if predict_df.empty:
        return pd.Series(dtype=float)
    cols = feature_columns(include_sector=False)
    preds = model.predict(predict_df[cols])
    return pd.Series(preds, index=predict_df["ticker"].to_numpy())


def walk_forward_predict(df: pd.DataFrame, rebalance_dates, horizon_td: int,
                          embargo_days: int = DEFAULT_EMBARGO_DAYS,
                          min_train_rows: int = 500, **lgb_params) -> pd.DataFrame:
    """Retrain (expanding window) at each date in `rebalance_dates`, predict
    that date's cross-section. Silently skips any date without
    `min_train_rows` of usable training history yet (too early). Returns a
    long DataFrame[date, ticker, alpha].

    Computes _label_close_dates ONCE here rather than once per rebalance date
    inside fit -- see _purge_embargo_mask's close_dates docstring."""
    global_dates = _global_trading_dates(df)
    close_dates = _label_close_dates(df["trade_date"], global_dates, horizon_td)
    rows = []
    for t in rebalance_dates:
        model = fit(df, t, horizon_td, embargo_days, min_train_rows, close_dates=close_dates, **lgb_params)
        if model is None:
            print(f"  [{t.date()}] skipped -- not enough purged training history yet")
            continue
        preds = predict(model, df, t)
        print(f"  [{t.date()}] trained, predicted {len(preds)} tickers")
        rows.extend({"date": t, "ticker": tkr, "alpha": a} for tkr, a in preds.items())
    return pd.DataFrame(rows, columns=["date", "ticker", "alpha"])


def shrink_alpha(alpha_series: pd.Series, factor: float) -> pd.Series:
    """Shrink predicted alpha toward 0 by `factor` in [0, 1] -- 0 leaves
    predictions untouched, 1 flattens them to nothing (pure risk-parity,
    ignore the model). Standard fix for mean-variance "error maximization"
    (Michaud 1989): an MVO chases whichever noisy alpha estimate happens to
    be largest this quarter; shrinking the input toward a flat prior caps
    how much any single noisy estimate can dominate the allocation, without
    touching Sigma or the optimizer itself (plan Phase 1.3)."""
    return alpha_series * (1 - factor)


def rank_ic(predictions: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    """Per-rebalance-date Spearman rank correlation between predicted alpha
    and the realized label. `predictions`: [date, ticker, alpha] (e.g. from
    walk_forward_predict). The mean of the returned Series across dates is
    the headline out-of-sample rank-IC diagnostic (proposal §2.6)."""
    merged = predictions.merge(
        df[["ticker", "trade_date", "label"]],
        left_on=["ticker", "date"], right_on=["ticker", "trade_date"], how="left",
    ).dropna(subset=["label"])
    return merged.groupby("date").apply(
        lambda g: g["alpha"].corr(g["label"], method="spearman"), include_groups=False
    )
