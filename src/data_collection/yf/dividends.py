"""
yf/dividends.py — yfinance dividend/JCP + corporate-events (splits) collectors.

Split out of yf_collectors.py (docs/DATA_LAYER_ORGANIZATION_PLAN.md §O3).
collect_splits_yf stays here rather than its own module: one ~50-line
function, same free `Ticker.history`/`Ticker.splits` endpoint family as the
rest of this file, not enough of a distinct concern to justify a 5th
submodule.
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

from .. import checkpoint, config, validate
from ..storage import _merge_save
from ._common import _extract_dividends, _last_completed_trading_day, _retry, _seed_last_date, _yf_symbol

log = logging.getLogger(__name__)


def collect_dividends_yf(tickers: list[str], mode: str, dividend_dir=None,
                          suffix: str | None = None, floor: str | None = None,
                          workers: int = 4) -> set[str]:
    """Returns the set of tickers whose fetch window had a new dividend or split --
    feed straight into collect_prices_yf's `full_refetch` (those are the only
    tickers whose stored adj_close can actually be stale; everyone else needs
    only a tail-only price fetch). `dividend_dir`/`suffix`/`floor` default to the BR globals (config.DIVIDENDS_DIR /
    config.YF_SUFFIX / config.START_DATE) -- mirrors collect_prices_yf's override pattern
    for the pure-yfinance US path (price_dir=config.US_DIVIDENDS_DIR, suffix="",
    floor="1900-01-01"). Verified back to 1994 against 5 real long-history payers
    (KO/GE/IBM/PG/XOM, 2026-07-28): plausible quarterly values, and KO's reconciles
    exactly to its known real 1994 dividend once un-split (0.04875 x 4 = $0.195/share,
    matching its real 1996 + 2012 2:1 splits).

    A never-payer gets no dividends file (there's nothing to write), so before
    2026-08-13 it also got no checkpoint entry -- every run re-walked its FULL
    history from `floor` just to reconfirm "still nothing" (5,376 of 9,593 US
    priced tickers, measured). Both no-data early-return paths now persist
    `cp[ticker]["checked_through"]` via `_mark_checked` so the next run starts
    from there instead (see _seed_last_date's checked_through fallback).

    `workers` runs multiple tickers concurrently -- real speedup found needed
    2026-07-31: a strictly sequential pass over the full 10,432-ticker US
    universe (each request already throttled to config.YF_RATE_LIMIT_SLEEP,
    same deliberate pace as collect_prices_yf) projected ~13 more hours for the
    remaining long tail. Each thread keeps its OWN YF_RATE_LIMIT_SLEEP pace --
    this doesn't remove the per-request throttle that pace exists for (the
    2026-07-31 Yahoo rate-limit incident earlier today), it runs a modest
    number of independently-throttled lanes in parallel instead of one,
    default kept low (4, not fundamentals.py's 8) since that incident was
    about cumulative volume over hours, not pure concurrency, and this is the
    same vendor prices' own throttle exists to protect. `cp` (the shared
    checkpoint dict) is mutated and persisted from multiple threads -- real
    race found while designing this: checkpoint.save()'s own lock only
    protects its file write, not a caller mutating the same dict object
    concurrently from another thread while save() is mid-snapshot
    (`{**data}` can raise "dictionary changed size during iteration" if
    another thread inserts a new key at that exact moment) -- _cp_lock
    serializes the whole "mutate cp, then persist it" step per ticker.
    """
    dividend_dir = dividend_dir or config.DIVIDENDS_DIR
    cp = checkpoint.load("yf_dividends", mode)
    cp_lock = Lock()
    changed: set[str] = set()
    checked_through = str(_last_completed_trading_day().date())

    def _mark_checked(ticker: str) -> None:
        # Persists "we looked through this date, nothing new" for a ticker
        # that gets no file this run (no dividend rows at all -- see
        # _seed_last_date's checked_through fallback). A payer's checkpoint
        # entry is fully overwritten by the success branch below on its next
        # real dividend, so this never lingers stale next to a real last_date.
        with cp_lock:
            cp[ticker] = {**cp.get(ticker, {}), "checked_through": checked_through}
            checkpoint.save("yf_dividends", mode, cp)

    def _one(ticker: str) -> None:
        try:
            path = dividend_dir / f"{ticker}.parquet"
            start = _seed_last_date(cp, ticker, path, "ex_date")
            fetch_start = (pd.to_datetime(start) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") \
                if start else (floor or config.START_DATE)

            t = yf.Ticker(_yf_symbol(ticker, suffix))
            hist = _retry(lambda: t.history(start=fetch_start, actions=True), f"dividends/{ticker}")
            if hist.empty:
                log.info("dividends %s: no new rows", ticker)
                _mark_checked(ticker)
                return

            # A split alone (no dividend) still staleifies stored adj_close --
            # checked independently of the dividends-empty return below, or a
            # split-only quarter would silently vanish from the changed set.
            if "Stock Splits" in hist.columns and (hist["Stock Splits"] > 0).any():
                changed.add(ticker)

            df = _extract_dividends(ticker, hist)
            if df.empty:
                log.info("dividends %s: no new dividend rows", ticker)
                _mark_checked(ticker)
                return
            changed.add(ticker)

            saved = _merge_save(df, path, "ex_date", validate.validate_dividends, f"dividends/{ticker}")
            if saved is not None:
                with cp_lock:
                    cp[ticker] = {"last_date": str(saved["ex_date"].max().date()), "rows": len(saved)}
                    checkpoint.save("yf_dividends", mode, cp)
                log.info("dividends %s: %d total payments", ticker, len(saved))
        except Exception as e:
            log.warning("dividends %s: skipping after error: %s", ticker, e)
        finally:
            sleep(config.YF_RATE_LIMIT_SLEEP)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_one, tickers))
    return changed


def collect_splits_yf(tickers: list[str], mode: str) -> None:
    """Free replacement for BolsAI's /stocks/corporate-events -- writes the same
    on-disk schema (ticker/date/type/ratio_from/ratio_to/factor) `repair.py`'s
    split-repair logic already consumes as its audit log. Dedicated collector
    (not folded into collect_prices_yf/collect_dividends_yf's changes_out side
    channel) because corporate_events is its own pipeline stage, run
    independently of whether prices are being re-fetched this pass -- reusing
    the price-fetch's `raw` would couple two stages that don't otherwise share
    state. `Ticker.splits` is a light, dedicated endpoint (not a full OHLCV
    history() call), so a full-history re-fetch every run is cheap; `_merge_save`
    dedups idempotently, same "small dataset, full overwrite" precedent as
    collect_sectors(). `mode` unused (kept for stage-map signature parity with
    every other collect_X(tickers, mode) in this module).

    factor convention: yfinance's split value is already "new shares per old
    share" (2.0 for a 2:1 split, 0.5 for a 1:2 reverse split) -- BolsAI's own
    audit log direction is inconsistent (documented in repair.py, which
    already tries both `factor` and `1/factor` when matching a price jump),
    so no inversion is needed here either way.
    """
    rows = []
    for ticker in tickers:
        try:
            splits = _retry(lambda: yf.Ticker(_yf_symbol(ticker)).splits, f"splits/{ticker}")
        except Exception as e:
            log.warning("splits %s: skipping after error: %s", ticker, e)
            continue
        finally:
            sleep(config.YF_RATE_LIMIT_SLEEP)
        if splits is None or splits.empty:
            continue
        ratio = splits.to_numpy(dtype=float)
        rows.append(pd.DataFrame({
            "ticker": ticker,
            "date": splits.index.tz_localize(None),
            "type": np.where(ratio >= 1, "SPLIT", "INPLIT"),
            "ratio_from": 1.0,
            "ratio_to": ratio,
            "description": [f"1:{r:g}" for r in ratio],
            "factor": ratio,
        }))
    if not rows:
        log.info("splits: no new corporate events")
        return
    df = pd.concat(rows, ignore_index=True)
    saved = _merge_save(df, config.CORP_EVENTS_PATH, "date",
                         validate.validate_corporate_events, "corporate_events")
    if saved is not None:
        log.info("splits: %d total rows", len(saved))
