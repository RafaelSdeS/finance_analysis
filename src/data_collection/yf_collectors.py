"""
yf_collectors.py — yfinance-sourced collectors for prices/fundamentals/dividends.

Mirrors collectors.py's contract exactly: collect_X(tickers, mode) -> validate ->
_merge_save -> checkpoint. Reuses _merge_save, checkpoint.py, validate.py as-is —
yfinance is just another source feeding the same idempotent writer.

company_info and macro have no yfinance equivalent and stay BolsAI/BCB-only
(see collectors.py); not touched here.
"""

import logging
from time import sleep

import numpy as np
import pandas as pd
import yfinance as yf

from . import checkpoint, config, validate
from .collectors import _merge_save

log = logging.getLogger(__name__)

K = 1000  # BolsAI fundamentals are stored in BRL thousands; yfinance reports full BRL.

# Full on-disk fundamentals schema (validate.FUND_COLS only lists the required subset).
FUND_FULL_COLS = [
    "ticker", "reference_date", "close_price", "shares_outstanding", "market_cap",
    "pl", "pvp", "ev_ebitda", "ev_ebit", "p_ebitda", "p_ebit", "p_sr", "lpa", "vpa",
    "gross_margin", "net_margin", "ebitda_margin", "ebit_margin", "roe", "roa", "roic",
    "ebit_over_assets", "asset_turnover", "p_assets", "current_ratio", "debt_equity",
    "net_debt_equity", "net_debt_ebitda", "net_debt_ebit", "cagr_revenue_5y", "cagr_earnings_5y",
    "net_income", "equity", "net_revenue", "total_debt", "ebitda", "ebit", "net_debt",
    "cash", "total_assets", "current_assets", "current_liabilities",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _yf_symbol(ticker: str) -> str:
    return config.TICKER_ALIASES.get(ticker, ticker) + config.YF_SUFFIX


def _retry(fn, label: str):
    """yfinance is a scraper with no typed exceptions worth special-casing —
    a couple of doubling-backoff retries is enough, no need for client.py's
    full httpx retry machinery (different transport entirely)."""
    last_err = None
    for attempt in range(config.YF_RETRIES):
        try:
            return fn()
        except Exception as e:
            last_err = e
            wait = config.YF_RETRY_SLEEP * 2 ** attempt
            log.warning("%s: yfinance error (%s), retry in %ds", label, e, wait)
            sleep(wait)
    raise RuntimeError(f"max retries exceeded for {label}: {last_err}")


def _seed_last_date(cp: dict, ticker: str, path, col: str) -> str | None:
    """Own checkpoint wins; else fall back to the max date already on disk
    (BolsAI backfill or a prior update run), so the first update doesn't
    redownload full history. --mode update keeps its own checkpoint dir,
    decoupled from prototype/full_scale."""
    if ticker in cp:
        return cp[ticker].get("last_date") or cp[ticker].get("last_quarter")
    if path.exists():
        return str(pd.read_parquet(path, columns=[col])[col].max().date())
    return None


# ---------------------------------------------------------------------------
# prices
# ---------------------------------------------------------------------------

def collect_prices_yf(tickers: list[str], mode: str):
    cp = checkpoint.load("yf_prices", mode)
    for ticker in tickers:
        try:
            path = config.PRICES_DIR / f"{ticker}.parquet"
            start = _seed_last_date(cp, ticker, path, "trade_date")
            fetch_start = (pd.to_datetime(start) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") \
                if start else config.START_DATE

            t = yf.Ticker(_yf_symbol(ticker))
            raw = _retry(lambda: t.history(start=fetch_start, auto_adjust=False), f"prices/{ticker}")
            if raw.empty:
                log.info("prices %s: no new rows (delisted/renamed/no yfinance coverage?)", ticker)
                continue

            adj_close = _retry(lambda: t.history(start=fetch_start, auto_adjust=True)["Close"],
                               f"prices/{ticker} adj_close")

            # Split-boundary fix: only the newly-fetched batch is touched, old rows on
            # disk are never rewritten. Always logged loudly so it can be spot-checked.
            splits = t.splits
            if start and len(splits):
                affected = splits[splits.index > pd.Timestamp(start, tz=splits.index.tz)]
                if len(affected):
                    log.warning("prices %s: split(s) in update window %s — reverse-adjusting "
                               "pre-split rows to BolsAI's unadjusted convention",
                               ticker, dict(affected))
                    for split_date, ratio in affected.items():
                        mask = raw.index < split_date
                        raw.loc[mask, ["Open", "High", "Low", "Close"]] *= ratio

            close = raw["Close"]
            ratio = adj_close / close

            df = pd.DataFrame({
                "ticker": ticker,
                "trade_date": raw.index.tz_localize(None),
                "open": raw["Open"].values,
                "high": raw["High"].values,
                "low": raw["Low"].values,
                "close": close.values,
                "adj_open": (raw["Open"] * ratio).values,
                "adj_high": (raw["High"] * ratio).values,
                "adj_low": (raw["Low"] * ratio).values,
                "adj_close": adj_close.values,
                "volume": raw["Volume"].values,
                "volume_adjusted": raw["Volume"].values,  # ponytail: yfinance doesn't split-adjust
                # volume; BolsAI does. Documented divergence, not worth reconstructing from splits.
                "traded_amount": (close * raw["Volume"]).values,  # approximation, no yfinance equivalent
                "num_trades": None,  # no yfinance equivalent at all
            })

            saved = _merge_save(df, path, "trade_date", validate.validate_prices, f"prices/{ticker}")
            if saved is not None:
                cp[ticker] = {"last_date": str(saved["trade_date"].max().date()), "rows": len(saved)}
                checkpoint.save("yf_prices", mode, cp)
                log.info("prices %s: %d total rows", ticker, len(saved))
        except Exception as e:
            log.warning("prices %s: skipping after error: %s", ticker, e)
        finally:
            sleep(config.RATE_LIMIT_SLEEP)


# ---------------------------------------------------------------------------
# fundamentals
# ---------------------------------------------------------------------------

def _compute_ratios(r: dict) -> dict:
    """Recompute BolsAI-equivalent ratios from yfinance raw figures.
    Formulas for market_cap/lpa/vpa/pl/pvp/roe/roa/net_margin/ebitda_margin/
    net_debt/debt_equity/ev_ebitda are the exact ones already verified at 5%
    tolerance against live BolsAI data in tests/raw_data/validate_vs_yfinance.py's
    check_internal_consistency(). The rest extend the same algebraic pattern.
    All divisions propagate NaN naturally on missing/zero inputs — no extra guards needed.
    """
    # np.float64 (not plain float) so x/0 -> inf/nan instead of ZeroDivisionError.
    g = lambda k: np.float64(r.get(k, np.nan))
    net_income, equity, net_revenue = g("net_income"), g("equity"), g("net_revenue")
    total_assets, total_debt, ebitda, ebit = g("total_assets"), g("total_debt"), g("ebitda"), g("ebit")
    cash, current_assets, current_liabilities = g("cash"), g("current_assets"), g("current_liabilities")
    shares, close_price = g("shares_outstanding"), g("close_price")
    cost_of_revenue = g("cost_of_revenue")

    market_cap = close_price * shares
    net_debt = total_debt - cash
    ev = market_cap + net_debt * K

    out = {
        "market_cap": market_cap,
        "lpa": net_income * K / shares,
        "vpa": equity * K / shares,
        "pl": market_cap / (net_income * K),
        "pvp": market_cap / (equity * K),
        "roe": net_income / equity * 100,
        "roa": net_income / total_assets * 100,
        "net_margin": net_income / net_revenue * 100,
        "ebitda_margin": ebitda / net_revenue * 100,
        "net_debt": net_debt,
        "debt_equity": total_debt / equity,
        "ev_ebitda": ev / (ebitda * K),
        "ev_ebit": ev / (ebit * K),
        "p_ebitda": market_cap / (ebitda * K),
        "p_ebit": market_cap / (ebit * K),
        "p_sr": market_cap / (net_revenue * K),
        "ebit_margin": ebit / net_revenue * 100,
        "ebit_over_assets": ebit / total_assets * 100,
        "asset_turnover": net_revenue / total_assets,
        "p_assets": market_cap / (total_assets * K),
        "current_ratio": current_assets / current_liabilities,
        "net_debt_equity": net_debt / equity,
        "net_debt_ebitda": net_debt / ebitda,
        "net_debt_ebit": net_debt / ebit,
        # ponytail: approximation — no tax-effected NOPAT available from yfinance.
        "roic": ebit / (total_debt + equity - cash) * 100,
        "gross_margin": (net_revenue - cost_of_revenue) / net_revenue * 100,
        # filled later by cagr_handler.fill_cagr_columns() over the combined
        # historical series — yfinance alone has ~1.5y depth, not enough for 5y CAGR.
        "cagr_revenue_5y": np.nan,
        "cagr_earnings_5y": np.nan,
    }
    return out


def _shares_outstanding(bs: pd.DataFrame, path) -> pd.Series:
    if "Ordinary Shares Number" in bs.index:
        return bs.loc["Ordinary Shares Number"]
    # carry forward the latest value already on disk — avoids an extra, slower t.info call
    if path.exists():
        existing = pd.read_parquet(path)
        if len(existing):
            return pd.Series(existing.iloc[-1]["shares_outstanding"], index=bs.columns)
    return pd.Series(np.nan, index=bs.columns)


def collect_fundamentals_yf(tickers: list[str], mode: str):
    cp = checkpoint.load("yf_fundamentals", mode)
    for ticker in tickers:
        try:
            fund_path = config.FUND_DIR / f"{ticker}.parquet"
            price_path = config.PRICES_DIR / f"{ticker}.parquet"

            t = yf.Ticker(_yf_symbol(ticker))
            qf = _retry(lambda: t.quarterly_income_stmt, f"fundamentals/{ticker} income")
            bs = _retry(lambda: t.quarterly_balance_sheet, f"fundamentals/{ticker} balance")
            if qf.empty or bs.empty:
                log.info("fundamentals %s: no data (delisted/no yfinance coverage?)", ticker)
                continue

            dates = sorted(set(qf.columns) & set(bs.columns))
            last = _seed_last_date(cp, ticker, fund_path, "reference_date")
            if last:
                dates = [d for d in dates if d > pd.Timestamp(last)]
            if not dates:
                log.info("fundamentals %s: up to date", ticker)
                continue

            shares = _shares_outstanding(bs, fund_path)
            prices = pd.read_parquet(price_path)[["trade_date", "close"]].sort_values("trade_date") \
                if price_path.exists() else pd.DataFrame(columns=["trade_date", "close"])

            def ttm(row_name):
                if row_name not in qf.index:
                    return pd.Series(dtype=float)
                s = pd.Series(qf.loc[row_name], dtype=float).sort_index()
                return s.rolling(4).sum() / K

            def point(row_name):
                if row_name not in bs.index:
                    return pd.Series(dtype=float)
                return pd.Series(bs.loc[row_name], dtype=float) / K

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
                row = {**base, **_compute_ratios(base)}
                row.pop("cost_of_revenue", None)  # not part of the on-disk schema
                rows.append(row)

            df = pd.DataFrame(rows)[FUND_FULL_COLS]
            saved = _merge_save(df, fund_path, "reference_date",
                                validate.validate_fundamentals, f"fundamentals/{ticker}")
            if saved is not None:
                cp[ticker] = {"last_quarter": str(saved["reference_date"].max().date()), "rows": len(saved)}
                checkpoint.save("yf_fundamentals", mode, cp)
                log.info("fundamentals %s: %d quarters", ticker, len(saved))
        except Exception as e:
            log.warning("fundamentals %s: skipping after error: %s", ticker, e)
        finally:
            sleep(config.RATE_LIMIT_SLEEP)


# ---------------------------------------------------------------------------
# dividends
# ---------------------------------------------------------------------------

def collect_dividends_yf(tickers: list[str], mode: str):
    cp = checkpoint.load("yf_dividends", mode)
    for ticker in tickers:
        try:
            path = config.DIVIDENDS_DIR / f"{ticker}.parquet"
            start = _seed_last_date(cp, ticker, path, "ex_date")
            fetch_start = (pd.to_datetime(start) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") \
                if start else config.START_DATE

            t = yf.Ticker(_yf_symbol(ticker))
            hist = _retry(lambda: t.history(start=fetch_start, actions=True), f"dividends/{ticker}")
            if hist.empty or "Dividends" not in hist.columns:
                log.info("dividends %s: no new rows", ticker)
                continue
            divs = hist[hist["Dividends"] > 0]["Dividends"]
            if divs.empty:
                log.info("dividends %s: no new rows", ticker)
                continue

            df = pd.DataFrame({
                "ex_date": divs.index.tz_localize(None),
                "payment_date": None,
                "type": "UNKNOWN",  # ponytail: yfinance can't distinguish JCP vs Dividendo
                "value_per_share": divs.values,
                "adjusted": False,
                "ticker": ticker,
            })

            saved = _merge_save(df, path, "ex_date", validate.validate_dividends, f"dividends/{ticker}")
            if saved is not None:
                cp[ticker] = {"last_date": str(saved["ex_date"].max().date()), "rows": len(saved)}
                checkpoint.save("yf_dividends", mode, cp)
                log.info("dividends %s: %d total payments", ticker, len(saved))
        except Exception as e:
            log.warning("dividends %s: skipping after error: %s", ticker, e)
        finally:
            sleep(config.RATE_LIMIT_SLEEP)


# ---------------------------------------------------------------------------
# self-check (no network)
# ---------------------------------------------------------------------------

def _demo():
    r = {
        "net_income": 100.0, "equity": 500.0, "net_revenue": 1000.0,
        "total_assets": 2000.0, "total_debt": 300.0, "ebitda": 200.0, "ebit": 150.0,
        "cash": 50.0, "current_assets": 400.0, "current_liabilities": 200.0,
        "shares_outstanding": 10.0, "close_price": 100.0, "cost_of_revenue": 600.0,
    }
    out = _compute_ratios(r)
    assert out["market_cap"] == 1000.0
    assert abs(out["roe"] - 20.0) < 1e-9
    assert abs(out["roa"] - 5.0) < 1e-9
    assert abs(out["net_margin"] - 10.0) < 1e-9
    assert out["net_debt"] == 250.0
    assert abs(out["debt_equity"] - 0.6) < 1e-9
    assert abs(out["current_ratio"] - 2.0) < 1e-9
    assert np.isinf(_compute_ratios({**r, "equity": 0.0})["roe"])  # 100/0 -> inf, no crash
    assert np.isnan(_compute_ratios({k: v for k, v in r.items() if k != "ebitda"})["ev_ebitda"])
    print("_compute_ratios: OK")

    cp = {"PETR4": {"last_date": "2026-01-01"}}
    assert _seed_last_date(cp, "PETR4", None, "trade_date") == "2026-01-01"
    print("_seed_last_date: OK")


if __name__ == "__main__":
    _demo()
