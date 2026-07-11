"""
collectors.py — one function per data source.

Each collector: fetch (resilient, resumable) → validate → idempotent save
(append + dedup) → checkpoint. No abstract base class: the four sources have
genuinely different fetch logic and share nothing worth abstracting.

Sources:
  - collect_macro             BCB SGS (SELIC/CDI/IPCA), keyless, date-chunked
  - collect_prices            BolsAI daily OHLCV, date-window paginated
  - collect_fundamentals      BolsAI quarterly, single call
  - collect_company_info      BolsAI metadata, fuzzy search
  - collect_dividends         BolsAI dividend/JCP payment history
  - collect_corporate_events  BolsAI splits/reverse-splits, market-wide, year-chunked
  - collect_sectors           BolsAI sector reference table, single call, no history
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
    import re
    c = client.make_client(config.BOLSAI_BASE, config.BOLSAI_API_KEY)
    _standard = re.compile(r"^[A-Z0-9]{4}[3-8]$")
    try:
        tickers, offset = [], 0
        while True:
            d = client.get_json(c, "/stocks/", {"limit": 500, "offset": offset})
            batch = d.get("tickers", [])
            if not batch:
                break
            tickers += batch
            offset += len(batch)
            if len(batch) < 500:
                break
        # exclude BDRs (34/35), FIIs/ETFs (11), and non-standard suffixes
        # But explicitly include BOVA11 (iShares Bovespa ETF, used as IBOV benchmark)
        result = sorted(t for t in tickers if _standard.match(t))
        if "BOVA11" not in result:
            result.append("BOVA11")
        return sorted(result)
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
            try:
                path = config.PRICES_DIR / f"{ticker}.parquet"
                if path.exists():
                    log.info("prices %s: already collected, skipping", ticker)
                    continue
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
            except Exception as e:
                log.warning("prices %s: skipping after error: %s", ticker, e)
            finally:
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
            try:
                path = config.FUND_DIR / f"{ticker}.parquet"
                if path.exists():
                    log.info("fundamentals %s: already collected, skipping", ticker)
                    continue
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
            except Exception as e:
                log.warning("fundamentals %s: skipping after error: %s", ticker, e)
            finally:
                sleep(config.RATE_LIMIT_SLEEP)
    finally:
        c.close()


# ---------------------------------------------------------------------------
# BolsAI company info
# ---------------------------------------------------------------------------

COMPANY_FIELDS = ["ticker", "ticker_primary", "corporate_name", "trade_name",
                  "cvm_code", "cnpj", "sector", "status"]


def _fetch_all_companies(c):
    """Paginate through all companies once. Returns dict: ticker_primary -> company_info."""
    all_companies = {}
    offset = 0
    while True:
        d = client.get_json(c, "/companies/", {"offset": offset, "limit": 500})
        batch = d.get("data", [])
        if not batch:
            break
        for co in batch:
            ticker = str(co.get("ticker_primary", "")).strip().upper()
            if ticker:  # only companies with a ticker
                all_companies[ticker] = co
        offset += len(batch)
        if len(batch) < 500:  # last page
            break
    return all_companies


def collect_company_info(tickers: list[str], mode: str):
    c = client.make_client(config.BOLSAI_BASE, config.BOLSAI_API_KEY)
    cp = checkpoint.load("company_info", mode)
    done = set(cp.get("done", []))
    path = config.COMPANY_DIR / "company_info.parquet"
    # Also skip tickers already in the parquet file (checkpoint-resilient)
    if path.exists():
        existing = set(pd.read_parquet(path)["ticker"].dropna().unique())
        done.update(existing)
    try:
        # Single paginated fetch of all companies (1-2 API calls vs 500+)
        all_companies = _fetch_all_companies(c)

        rows = []
        for ticker in tickers:
            if ticker in done:
                log.info("company %s: already collected", ticker)
                continue
            co = all_companies.get(ticker)
            if co:
                row = {**{f: co.get(f) for f in COMPANY_FIELDS}, "ticker": ticker}
                rows.append(row)
                log.info("company %s: matched", ticker)
            else:
                log.warning("company %s: not found on B3", ticker)
            done.add(ticker)
            checkpoint.save("company_info", mode, {"done": sorted(done)})

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
            try:
                path = config.DIVIDENDS_DIR / f"{ticker}.parquet"
                if path.exists():
                    log.info("dividends %s: already collected, skipping", ticker)
                    continue
                d = client.get_json(c, f"/dividends/{ticker}", {"years": config.DIVIDENDS_YEARS})
                payments = d.get("payments", [])
                if not payments:
                    log.warning("dividends %s: no data", ticker)
                    continue
                df = pd.DataFrame(payments)
                df["ticker"] = ticker
                # Announced-but-not-yet-effective dividends have a future ex_date; that's
                # normal for this endpoint (unlike prices/fundamentals), but validate_dividends'
                # shared future-date guard would reject the whole batch, so drop just those rows.
                df["ex_date"] = pd.to_datetime(df["ex_date"])
                future = df["ex_date"] > pd.Timestamp.now() + pd.Timedelta(days=2)
                if future.any():
                    log.info("dividends %s: dropping %d rows with future ex_date", ticker, future.sum())
                    df = df[~future]
                if df.empty:
                    log.warning("dividends %s: no data after dropping future rows", ticker)
                    continue
                saved = _merge_save(df, path, "ex_date", validate.validate_dividends, f"dividends/{ticker}")
                if saved is not None:
                    log.info("dividends %s: %d payments", ticker, len(saved))
            except Exception as e:
                log.warning("dividends %s: skipping after error: %s", ticker, e)
            finally:
                sleep(config.RATE_LIMIT_SLEEP)
    finally:
        c.close()


# ---------------------------------------------------------------------------
# BolsAI corporate events (splits/reverse-splits, market-wide)
# ---------------------------------------------------------------------------

def collect_corporate_events(mode: str):
    """All confirmed splits/reverse-splits for all tickers. Endpoint has no
    offset param, so pagination = one call per calendar year."""
    c = client.make_client(config.BOLSAI_BASE, config.BOLSAI_API_KEY)
    cp = checkpoint.load("corporate_events", mode)
    path = config.CORP_EVENTS_DIR / "corporate_events.parquet"
    try:
        start_year = cp.get("last_year", int(config.START_DATE[:4]) - 1) + 1
        end_year = datetime.now().year
        start_year = min(start_year, end_year)

        rows = []
        for year in range(start_year, end_year + 1):
            d = client.get_json(c, "/stocks/corporate-events", {"year": year, "limit": 1000})
            rows += d.get("events", [])
        if not rows:
            log.info("corporate_events: no new rows")
            return

        df = pd.DataFrame(rows)
        saved = _merge_save(df, path, "date", validate.validate_corporate_events, "corporate_events")
        if saved is not None:
            # leave end_year unlocked: a same-year split can be announced after this run
            cp["last_year"] = end_year - 1
            checkpoint.save("corporate_events", mode, cp)
            log.info("corporate_events: %d total rows", len(saved))
    finally:
        c.close()


# ---------------------------------------------------------------------------
# BolsAI sectors (reference table, no history)
# ---------------------------------------------------------------------------

def collect_sectors():
    """Canonical sector names + active company counts. Single call, full overwrite."""
    c = client.make_client(config.BOLSAI_BASE, config.BOLSAI_API_KEY)
    path = config.COMPANY_DIR / "sectors.parquet"
    try:
        d = client.get_json(c, "/companies/sectors")
        df = pd.DataFrame(d.get("sectors", []))
        vr = validate.validate_sectors(df)
        if not vr.passed:
            log.error("sectors validation FAILED: %s", vr.errors)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        log.info("sectors: %d sectors total", len(df))
    finally:
        c.close()
