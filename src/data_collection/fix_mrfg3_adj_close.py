"""
fix_mrfg3_adj_close.py — one-off repair for MRFG3's (spliced into MBRF3)
chronic adj_close corruption in BolsAI's raw data.

Background (2026-07-14, TOP50_ML_READINESS_AUDIT.md): MRFG3's raw adj_close
is stored at 2-decimal precision for values well below 1.0 (e.g. 0.01, 0.02),
so an actual adjustment-factor drift of a few percent rounds to a "100%
jump." Confirmed this is not a stale local snapshot -- BolsAI's LIVE
/stocks/MRFG3/history endpoint serves the identical broken values today
(spot-checked 2011-08/09 directly). Also confirmed NOT explained by any real
Marfrig stock split/grupamento in this window (web search: dividends only,
no split events found for 2019-2022).

yfinance's MBRF3.SA series is independently clean across every one of the
flagged dates (spot-checked directly) and its own flat-run fraction is 0%
(not vendor coverage-padding, see _flat_run_fraction in yf_collectors.py).
Rather than replace BolsAI's OHLCV wholesale (which would also discard its
volume/close, not shown to be wrong), this recomputes only the adj_* columns
by applying yfinance's own (adj_close/close) ratio to BolsAI's own raw OHLC
-- same "keep raw, fix only the derived adjustment" pattern as
repair_unadjusted_splits(), just sourcing the correction ratio from an
independent vendor instead of corporate_events.parquet.

Fixes BOTH data/raw/prices/MRFG3.parquet and MBRF3.parquet: BolsAI's raw
store has two independently-collected files with the identical bug (same
2007-06-29..2026-07-10 span, same broken values) -- MRFG3.parquet from the
original pre-rename ticker, MBRF3.parquet from a later full-history
collection run under the post-rename name. ticker_continuity.json's
MRFG3->MBRF3 splice boundary resolves to MBRF3's OWN first trade date (its
earliest date, since it already has full standalone history), so
apply_ticker_continuity() drops essentially all of MRFG3's rows as "past the
boundary" -- MBRF3.parquet is the file that actually reaches ml_dataset.parquet;
fixing MRFG3.parquet alone (first attempt, 2026-07-14) silently had no effect.

This is a targeted fix for this one ticker, verified against two independent
sources (yfinance + a live BolsAI API spot-check) -- not a general "replace
adj_close from yfinance" tool. The same 2-decimal precision-floor issue is
known to exist elsewhere in the dataset (adj_close_precision_degraded flag,
features.py) and is left as an accepted, documented vendor limitation for
tickers outside this audit's top-50 scope; applying this same swap blindly
dataset-wide has not been validated and is not attempted here.

Run from project root: python -m src.data_collection.fix_mrfg3_adj_close
"""

import logging

import pandas as pd
import yfinance as yf

from . import config

log = logging.getLogger(__name__)

TICKERS = ["MRFG3", "MBRF3"]  # both raw files carry the identical BolsAI bug; see module docstring
YF_SYMBOL = "MBRF3.SA"  # yfinance already resolved the rename; MRFG3.SA 404s


def fix_one(ticker: str, yf_close: pd.Series, yf_adj: pd.Series) -> None:
    path = config.PRICES_DIR / f"{ticker}.parquet"
    if not path.exists():
        log.info("%s: no raw file, skipping", ticker)
        return
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    ratio = (yf_adj / yf_close).reset_index()
    ratio.columns = ["trade_date", "ratio"]
    ratio["trade_date"] = ratio["trade_date"].dt.tz_localize(None)

    before = df[["adj_open", "adj_high", "adj_low", "adj_close"]].copy()
    df = df.merge(ratio, on="trade_date", how="left")
    missing = df["ratio"].isna().sum()
    if missing:
        log.warning("%s: %d/%d dates have no yfinance ratio -- forward/back-filling from neighbors",
                    ticker, missing, len(df))
        df["ratio"] = df["ratio"].ffill().bfill()

    for raw_col, adj_col in [("open", "adj_open"), ("high", "adj_high"),
                              ("low", "adj_low"), ("close", "adj_close")]:
        df[adj_col] = df[raw_col] * df["ratio"]
    df = df.drop(columns=["ratio"])

    # sanity: OHLC ordering must still hold after the rescale (uniform ratio
    # per row preserves ordering exactly, but check rather than assume)
    bad = ((df["adj_low"] > df["adj_open"] + 1e-9) | (df["adj_low"] > df["adj_close"] + 1e-9) |
           (df["adj_high"] < df["adj_open"] - 1e-9) | (df["adj_high"] < df["adj_close"] - 1e-9))
    if bad.any():
        raise ValueError(f"{ticker}: {bad.sum()} rows violate adj OHLC ordering after rescale -- aborting write")

    changed = (before["adj_close"] != df["adj_close"]).sum()
    log.info("%s: recomputed adj_* for %d/%d rows from yfinance ratio (%d changed)",
             ticker, len(df), len(df), changed)

    df.to_parquet(path, index=False)
    log.info("%s: written to %s", ticker, path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t = yf.Ticker(YF_SYMBOL)
    start = "2007-01-01"
    yf_close = t.history(start=start, auto_adjust=False)["Close"]
    yf_adj = t.history(start=start, auto_adjust=True)["Close"]

    for ticker in TICKERS:
        fix_one(ticker, yf_close, yf_adj)


if __name__ == "__main__":
    main()
