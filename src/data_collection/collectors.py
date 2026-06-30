"""
collectors.py — one function per data source.

Each collector: fetch (resilient, resumable) → validate → idempotent save
(append + dedup) → checkpoint. No abstract base class: the four sources have
genuinely different fetch logic and share nothing worth abstracting.

Sources:
  - collect_macro          BCB SGS (SELIC/CDI/IPCA), keyless, date-chunked
  - collect_prices         BolsAI daily OHLCV, date-window paginated
  - collect_fundamentals   BolsAI quarterly, single call
  - collect_company_info   BolsAI metadata, fuzzy search
"""

import logging
from datetime import datetime, timedelta
from time import sleep

import httpx
import pandas as pd

from . import checkpoint, client, config, validate

log = logging.getLogger(__name__)

# BolsAI price API field -> stored parquet column
PRICE_RENAME = {
    "adjusted_open": "adj_open",
    "adjusted_high": "adj_high",
    "adjusted_low": "adj_low",
    "adjusted_close": "adj_close",
    "adjusted_volume": "volume_adjusted",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _chunk_dates(start: str, end: str, years: int):
    """Yield (start, end) ISO windows of <= `years` each, to stay under API caps."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    while s <= e:
        chunk_end = min(datetime(s.year + years, s.month, s.day), e)
        yield s.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        s = chunk_end + timedelta(days=1)


def _merge_save(df_new, path, date_col, validator, ticker_label=""):
    """Append to existing parquet, dedup on date_col, validate, write. Idempotent."""
    if path.exists():
        df_old = pd.read_parquet(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df[date_col] = pd.to_datetime(df[date_col])
    df = (df.drop_duplicates(subset=["ticker", date_col] if "ticker" in df.columns else [date_col],
                             keep="last")
            .sort_values(date_col)
            .reset_index(drop=True))
    vr = validator(df)
    if not vr.passed:
        log.error("%s validation FAILED: %s", ticker_label, vr.errors)
        return None
    for w in vr.warnings:
        log.warning("%s: %s", ticker_label, w)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


# ---------------------------------------------------------------------------
# BolsAI: all tickers (full-scale ticker universe)
# ---------------------------------------------------------------------------

def get_all_tickers() -> list[str]:
    c = client.make_client(config.BOLSAI_BASE, config.BOLSAI_API_KEY)
    try:
        tickers, offset = [], 0
        while True:
            d = client.get_json(c, "/stocks/", {"limit": 500, "offset": offset})
            batch = d.get("data", [])
            if not batch:
                break
            tickers += [s.get("ticker") or s.get("ticker_primary") for s in batch]
            offset += len(batch)
            if len(batch) < 500:
                break
        return sorted(t for t in tickers if t)
    finally:
        c.close()


# ---------------------------------------------------------------------------
# BCB macro
# ---------------------------------------------------------------------------

def collect_macro(mode: str):
    # BCB needs "bcdata.sgs.{id}/dados" (dot, not slash) — base_url joining would
    # mangle it, so use a baseless client and pass the full URL.
    c = client.make_client("")
    cp = checkpoint.load("macro", mode)
    try:
        for name, sid in config.BCB_SERIES.items():
            path = config.MACRO_DIR / f"{name}.parquet"
            start = cp.get(name, {}).get("last_date")
            start = (pd.to_datetime(start) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") \
                if start else config.START_DATE
            end = datetime.now().strftime("%Y-%m-%d")
            if start > end:
                log.info("macro %s: up to date", name)
                continue

            rows = []
            for s, e in _chunk_dates(start, end, 10):
                try:
                    d = client.get_json(c, f"{config.BCB_BASE}.{sid}/dados", {
                        "formato": "json",
                        "dataInicial": datetime.strptime(s, "%Y-%m-%d").strftime("%d/%m/%Y"),
                        "dataFinal": datetime.strptime(e, "%Y-%m-%d").strftime("%d/%m/%Y"),
                    })
                except httpx.HTTPStatusError as ex:
                    # BCB returns 404 for ranges with no published data (e.g. weekends)
                    if ex.response.status_code == 404:
                        continue
                    raise
                rows += d or []
            if not rows:
                log.info("macro %s: no new rows", name)
                continue

            df = pd.DataFrame(rows)
            df["reference_date"] = pd.to_datetime(df["data"], dayfirst=True)
            df[name] = pd.to_numeric(df["valor"].astype(str).str.replace(",", "."), errors="coerce")
            df = df[["reference_date", name]].dropna()

            saved = _merge_save(df, path, "reference_date",
                                lambda x: validate.validate_macro(x, name), f"macro/{name}")
            if saved is not None:
                cp[name] = {"last_date": str(saved["reference_date"].max().date()), "rows": len(saved)}
                checkpoint.save("macro", mode, cp)
                log.info("macro %s: %d total rows", name, len(saved))
    finally:
        c.close()


# ---------------------------------------------------------------------------
# BolsAI prices
# ---------------------------------------------------------------------------

def _fetch_price_window(c, ticker, start, end):
    d = client.get_json(c, f"/stocks/{ticker}/history",
                        {"limit": config.PRICE_LIMIT, "start": start, "end": end})
    return d.get("prices", [])


def collect_prices(tickers: list[str], mode: str):
    c = client.make_client(config.BOLSAI_BASE, config.BOLSAI_API_KEY)
    cp = checkpoint.load("prices", mode)
    try:
        for ticker in tickers:
            path = config.PRICES_DIR / f"{ticker}.parquet"
            last = cp.get(ticker, {}).get("last_date")
            end = datetime.now().strftime("%Y-%m-%d")

            if last:  # incremental: one small window from day after last
                start = (pd.to_datetime(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                if start > end:
                    log.info("prices %s: up to date", ticker)
                    continue
                windows = [(start, end)]
            else:      # backfill: chunk to stay under the 5000-row cap
                windows = list(_chunk_dates(config.START_DATE, end, config.PRICE_CHUNK_YEARS))

            records = []
            for s, e in windows:
                records += _fetch_price_window(c, ticker, s, e)
            if not records:
                log.info("prices %s: no new rows", ticker)
                continue

            df = pd.DataFrame(records).rename(columns=PRICE_RENAME)
            df["ticker"] = ticker
            df = df[validate.PRICE_COLS]

            saved = _merge_save(df, path, "trade_date", validate.validate_prices, f"prices/{ticker}")
            if saved is not None:
                cp[ticker] = {"last_date": str(saved["trade_date"].max().date()), "rows": len(saved)}
                checkpoint.save("prices", mode, cp)
                log.info("prices %s: %d total rows", ticker, len(saved))
            sleep(config.RATE_LIMIT_SLEEP)
    finally:
        c.close()


# ---------------------------------------------------------------------------
# BolsAI fundamentals
# ---------------------------------------------------------------------------

def collect_fundamentals(tickers: list[str], mode: str):
    c = client.make_client(config.BOLSAI_BASE, config.BOLSAI_API_KEY)
    cp = checkpoint.load("fundamentals", mode)
    try:
        for ticker in tickers:
            path = config.FUND_DIR / f"{ticker}.parquet"
            d = client.get_json(c, f"/fundamentals/{ticker}/history", {"limit": config.FUND_LIMIT})
            hist = d.get("history", [])
            if not hist:
                log.warning("fundamentals %s: no data", ticker)
                continue

            df = pd.DataFrame(hist)
            df["ticker"] = ticker

            saved = _merge_save(df, path, "reference_date",
                                validate.validate_fundamentals, f"fundamentals/{ticker}")
            if saved is not None:
                cp[ticker] = {"last_quarter": str(saved["reference_date"].max().date()), "rows": len(saved)}
                checkpoint.save("fundamentals", mode, cp)
                log.info("fundamentals %s: %d quarters", ticker, len(saved))
            sleep(config.RATE_LIMIT_SLEEP)
    finally:
        c.close()


# ---------------------------------------------------------------------------
# BolsAI company info
# ---------------------------------------------------------------------------

COMPANY_FIELDS = ["ticker", "ticker_primary", "corporate_name", "trade_name",
                  "cvm_code", "cnpj", "sector", "status"]


def _find_company(c, ticker):
    """Fetch company info directly from /companies/{ticker} endpoint."""
    try:
        co = client.get_json(c, f"/companies/{ticker}", {})
        if co and co.get("ticker_primary"):
            return {**{f: co.get(f) for f in COMPANY_FIELDS}, "ticker": ticker}
    except Exception as e:
        log.debug("company %s: direct lookup failed, trying search as fallback", ticker, exc_info=True)

    # Fallback: fuzzy search if direct lookup fails
    base = ticker.rstrip("0123456789")
    for term in dict.fromkeys([base.lower(), base[:3].lower(), base[:2].lower()]):
        if not term:
            continue
        d = client.get_json(c, "/companies/", {"search": term, "limit": 20})
        for co in d.get("data", []):
            if str(co.get("ticker_primary", "")).strip().upper() == ticker:
                return {**{f: co.get(f) for f in COMPANY_FIELDS}, "ticker": ticker}
        sleep(0.2)
    return None


def collect_company_info(tickers: list[str], mode: str):
    c = client.make_client(config.BOLSAI_BASE, config.BOLSAI_API_KEY)
    cp = checkpoint.load("company_info", mode)
    done = set(cp.get("done", []))
    path = config.COMPANY_DIR / "company_info.parquet"
    try:
        rows = []
        for ticker in tickers:
            if ticker in done:
                log.info("company %s: already collected", ticker)
                continue
            row = _find_company(c, ticker)
            if row:
                rows.append(row)
                log.info("company %s: matched", ticker)
            else:
                log.warning("company %s: no exact match (skipped)", ticker)
            done.add(ticker)
            checkpoint.save("company_info", mode, {"done": sorted(done)})
            sleep(config.RATE_LIMIT_SLEEP)

        if not rows:
            log.info("company_info: no new companies collected")
            return
        df_new = pd.DataFrame(rows, columns=COMPANY_FIELDS)
        if path.exists():
            df_new = pd.concat([pd.read_parquet(path), df_new], ignore_index=True)
        df_new = df_new.drop_duplicates("ticker", keep="last").sort_values("ticker").reset_index(drop=True)
        vr = validate.validate_company_info(df_new)
        if not vr.passed:
            log.error("company_info validation FAILED: %s", vr.errors)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        df_new.to_parquet(path, index=False)
        log.info("company_info: %d total companies", len(df_new))
    finally:
        c.close()


# ---------------------------------------------------------------------------
# BolsAI dividends
# ---------------------------------------------------------------------------

def collect_dividends(tickers: list[str], mode: str):
    c = client.make_client(config.BOLSAI_BASE, config.BOLSAI_API_KEY)
    try:
        for ticker in tickers:
            path = config.DIVIDENDS_DIR / f"{ticker}.parquet"
            d = client.get_json(c, f"/dividends/{ticker}", {"years": config.DIVIDENDS_YEARS})
            payments = d.get("payments", [])
            if not payments:
                log.warning("dividends %s: no data", ticker)
                continue
            df = pd.DataFrame(payments)
            df["ticker"] = ticker
            saved = _merge_save(df, path, "ex_date", validate.validate_dividends, f"dividends/{ticker}")
            if saved is not None:
                log.info("dividends %s: %d payments", ticker, len(saved))
            sleep(config.RATE_LIMIT_SLEEP)
    finally:
        c.close()
