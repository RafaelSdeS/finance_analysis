"""
labels.py -- forward excess-return-over-CDI label (proposal §2.2, §3.3).

label_t = (adj_close_{t+H} / adj_close_t - 1) - (compounded CDI return over
the same H-trading-day window). No lookahead by construction: label at row
t only reads adj_close/cdi realized strictly after t, up to and including
t+H. The last H rows per ticker have no forward window and come back NaN
-- that NaN is the leakage boundary and must never be imputed downstream.

`cdi` is %/trading-day (manifest.COLUMN_UNITS), e.g. 0.0534 = 0.0534%/day.

ponytail: the H-row forward window is H trading days *for that ticker's own
row sequence*, not H calendar days of the market. A halted ticker's H rows
span more calendar time than a fully-liquid one's -- same convention every
other rolling/forward feature in build_dataset/features.py already uses
(e.g. volatility_20d), so this isn't a new inconsistency, just naming it.
"""

import numpy as np
import pandas as pd

HORIZON_12M_TD = 252
HORIZON_6M_TD = 126


def forward_excess_return(df: pd.DataFrame, horizon_td: int = HORIZON_12M_TD) -> pd.Series:
    """Returns a Series aligned to df.index (any row order). `df` needs
    ticker, trade_date, adj_close, cdi, adj_close_precision_degraded."""
    working = df[["ticker", "trade_date", "adj_close", "cdi", "adj_close_precision_degraded"]]
    working = working.sort_values(["ticker", "trade_date"])
    g = working.groupby("ticker", sort=False)

    adj_close = working["adj_close"].where(working["adj_close"] > 0)
    fwd_close = g["adj_close"].shift(-horizon_td)
    fwd_close = fwd_close.where(fwd_close > 0)
    fwd_ret = fwd_close / adj_close - 1

    log_cdi = np.log1p(working["cdi"] / 100)
    cum_log_cdi = log_cdi.groupby(working["ticker"], sort=False).cumsum()
    fwd_cum_log_cdi = cum_log_cdi.groupby(working["ticker"], sort=False).shift(-horizon_td)
    fwd_cdi = np.exp(fwd_cum_log_cdi - cum_log_cdi) - 1

    label = (fwd_ret - fwd_cdi).where(working["adj_close_precision_degraded"] != 1)
    return label.reindex(df.index)
