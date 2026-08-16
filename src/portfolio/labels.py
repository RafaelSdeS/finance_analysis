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

terminal_events (optional, build_dataset.terminal_events.load_terminal_events()):
for a ticker that dies inside the panel, the last `horizon_td` rows have no
forward window and would otherwise stay NaN forever even though the real
outcome is known. Those tail rows get the realized terminal_payoff instead,
with the CDI leg compounded only to the ticker's own last date (the position
is liquidated there, not at t+horizon_td -- comparing against a full H days
of CDI would overstate the true opportunity cost). Only rows that are
already NaN for this reason are touched: a live ticker, or a delisted one
with no resolved terminal event, is completely unaffected.
"""

import numpy as np
import pandas as pd

HORIZON_12M_TD = 252
HORIZON_6M_TD = 126


def forward_excess_return(df: pd.DataFrame, horizon_td: int = HORIZON_12M_TD,
                           terminal_events: pd.DataFrame | None = None) -> pd.Series:
    """Returns a Series aligned to df.index (any row order). `df` needs
    ticker, trade_date, adj_close, cdi, adj_close_precision_degraded.
    terminal_events: optional (ticker, terminal_payoff) table -- see module
    docstring."""
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

    if terminal_events is not None and len(terminal_events):
        payoff = working["ticker"].map(terminal_events.set_index("ticker")["terminal_payoff"])
        # tail rows only: positions within horizon_td of that ticker's LAST
        # row -- the exact rows a normal shift(-horizon_td) leaves NaN. A mid-
        # history NaN (e.g. a precision-degraded base row) must never be
        # overwritten with a terminal payoff computed for the series' actual end.
        pos_from_end = g["trade_date"].transform("size") - 1 - g.cumcount()
        is_tail = pos_from_end < horizon_td
        needs_fill = (label.isna() & payoff.notna() & is_tail
                      & (adj_close > 0) & (working["adj_close_precision_degraded"] != 1))

        term_fwd_ret = payoff / adj_close - 1
        last_cum_log_cdi = cum_log_cdi.groupby(working["ticker"], sort=False).transform("last")
        term_fwd_cdi = np.exp(last_cum_log_cdi - cum_log_cdi) - 1
        term_label = term_fwd_ret - term_fwd_cdi

        label = label.where(~needs_fill, term_label)

    return label.reindex(df.index)
