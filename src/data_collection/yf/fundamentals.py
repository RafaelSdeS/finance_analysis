"""
yf/fundamentals.py — yfinance quarterly fundamentals collector.

Split out of yf_collectors.py (docs/DATA_LAYER_ORGANIZATION_PLAN.md §O3).
Not the default BR fundamentals source (CVM is -- see CLAUDE.md's "Data
sources & limits"); kept importable behind a DATA_SOURCE flip.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

import numpy as np
import pandas as pd
import yfinance as yf

from .. import checkpoint, config, validate
from ..ratios import FUND_FULL_COLS, compute_ratios
from ..storage import _merge_save
from ._common import _retry, _seed_last_date, _yf_symbol

log = logging.getLogger(__name__)


def _shares_outstanding(bs: pd.DataFrame, path) -> pd.Series:
    if "Ordinary Shares Number" in bs.index:
        return bs.loc["Ordinary Shares Number"]
    # carry forward the latest value already on disk — avoids an extra, slower t.info call
    if path.exists():
        existing = pd.read_parquet(path)
        if len(existing):
            return pd.Series(existing.iloc[-1]["shares_outstanding"], index=bs.columns)
    return pd.Series(np.nan, index=bs.columns)


def collect_fundamentals_yf(tickers: list[str], mode: str, workers: int = 1):
    """`workers` (default 1, unchanged sequential behavior) runs multiple tickers
    concurrently -- same ThreadPoolExecutor + cp_lock shape as collect_prices_yf
    and collect_dividends_yf, for the same reason (checkpoint.save()'s lock only
    protects its own file write, not a caller mutating the shared `cp` dict from
    another thread mid-snapshot).
    """
    cp = checkpoint.load("yf_fundamentals", mode)
    cp_lock = Lock()

    def _one(ticker: str) -> None:
        try:
            fund_path = config.FUND_DIR / f"{ticker}.parquet"
            price_path = config.PRICES_DIR / f"{ticker}.parquet"

            t = yf.Ticker(_yf_symbol(ticker))
            qf = _retry(lambda: t.quarterly_income_stmt, f"fundamentals/{ticker} income")
            bs = _retry(lambda: t.quarterly_balance_sheet, f"fundamentals/{ticker} balance")
            if qf.empty or bs.empty:
                log.info("fundamentals %s: no data (delisted/no yfinance coverage?)", ticker)
                return

            dates = sorted(set(qf.columns) & set(bs.columns))
            last = _seed_last_date(cp, ticker, fund_path, "reference_date")
            if last:
                dates = [d for d in dates if d > pd.Timestamp(last)]
            if not dates:
                log.info("fundamentals %s: up to date", ticker)
                return

            shares = _shares_outstanding(bs, fund_path)
            prices = pd.read_parquet(price_path)[["trade_date", "close"]].sort_values("trade_date") \
                if price_path.exists() else pd.DataFrame(columns=["trade_date", "close"])

            def ttm(row_name):
                if row_name not in qf.index:
                    return pd.Series(dtype=float)
                s = pd.Series(qf.loc[row_name], dtype=float).sort_index()
                return s.rolling(4).sum()

            def point(row_name):
                if row_name not in bs.index:
                    return pd.Series(dtype=float)
                return pd.Series(bs.loc[row_name], dtype=float)

            net_revenue, net_income = ttm("Total Revenue"), ttm("Net Income")
            ebitda, ebit = ttm("EBITDA"), ttm("EBIT")
            cost_of_revenue = ttm("Cost Of Revenue")
            equity, total_assets = point("Stockholders Equity"), point("Total Assets")
            total_debt, cash = point("Total Debt"), point("Cash And Cash Equivalents")
            current_assets, current_liabilities = point("Current Assets"), point("Current Liabilities")

            rows = []
            for d in dates:
                close_at_date = prices[prices["trade_date"] <= d]["close"]
                base = {
                    "ticker": ticker,
                    "reference_date": d,
                    "close_price": close_at_date.iloc[-1] if len(close_at_date) else np.nan,
                    "shares_outstanding": shares.get(d, np.nan),
                    "net_income": net_income.get(d, np.nan),
                    "equity": equity.get(d, np.nan),
                    "net_revenue": net_revenue.get(d, np.nan),
                    "total_debt": total_debt.get(d, np.nan),
                    "ebitda": ebitda.get(d, np.nan),
                    "ebit": ebit.get(d, np.nan),
                    "cash": cash.get(d, np.nan),
                    "total_assets": total_assets.get(d, np.nan),
                    "current_assets": current_assets.get(d, np.nan),
                    "current_liabilities": current_liabilities.get(d, np.nan),
                    "cost_of_revenue": cost_of_revenue.get(d, np.nan),
                }
                base["net_debt"] = base["total_debt"] - base["cash"]
                # yfinance figures are already full BRL (§1: DATA_LAYER_CORRECTNESS_PLAN.md) --
                # unlike the BolsAI/CVM path, no thousands->units crossing needed here.
                row = {**base, **compute_ratios(base, unit_scale=1)}
                row.pop("cost_of_revenue", None)  # not part of the on-disk schema
                rows.append(row)

            df = pd.DataFrame(rows)[FUND_FULL_COLS]
            saved = _merge_save(df, fund_path, "reference_date",
                                validate.validate_fundamentals, f"fundamentals/{ticker}")
            if saved is not None:
                with cp_lock:
                    cp[ticker] = {"last_quarter": str(saved["reference_date"].max().date()), "rows": len(saved)}
                    checkpoint.save("yf_fundamentals", mode, cp)
                log.info("fundamentals %s: %d quarters", ticker, len(saved))
        except Exception as e:
            log.warning("fundamentals %s: skipping after error: %s", ticker, e)
        finally:
            sleep(config.YF_RATE_LIMIT_SLEEP)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_one, tickers))
