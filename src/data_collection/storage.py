"""
storage.py — shared parquet append/dedup/validate/write + date-window chunking.

Generic to any collector (BolsAI, FRED, CVM, yfinance alike) -- extracted out
of collectors.py (2026-08-05) so BR-specific code isn't a dependency of
US-specific code just to save a file. See
docs/DATA_COLLECTION_REORGANIZATION_PLAN.md S1.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd

log = logging.getLogger(__name__)


def _chunk_dates(start: str, end: str, years: int):
    """Yield (start, end) ISO windows of <= `years` each, to stay under API caps.

    Uses pd.DateOffset (not raw datetime(s.year+years, s.month, s.day)) to
    step forward: a plain datetime() construction raises ValueError whenever
    `s` is Feb 29 and `s.year + years` isn't a leap year -- which, for
    years=10, is EVERY time (adding 10 always shifts year%4 by 2). Reachable
    in practice via collect_macro's incremental path: BCB's daily selic/cdi
    series can have a checkpoint last_date of Feb 28 in a leap year, making
    the next start date Feb 29. DateOffset clamps to Feb 28 on a non-leap
    target year instead of raising.
    """
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    while s <= e:
        chunk_end = min(pd.Timestamp(s) + pd.DateOffset(years=years), pd.Timestamp(e))
        yield s.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        s = chunk_end.to_pydatetime() + timedelta(days=1)


def _merge_save(df_new, path, date_col, validator, ticker_label=""):
    """Append to existing parquet, dedup on date_col, validate, write. Idempotent.

    Validates only the newly-fetched batch, not the full merged history: a row
    already accepted onto disk in a previous run must not block ingestion of new,
    valid rows forever (e.g. a known vendor data glitch from years ago).
    """
    df_new = df_new.copy()
    df_new[date_col] = pd.to_datetime(df_new[date_col])
    dedup_cols = ["ticker", date_col] if "ticker" in df_new.columns else [date_col]
    df_new = df_new.drop_duplicates(subset=dedup_cols, keep="last")

    vr = validator(df_new)
    if not vr.passed:
        log.error("%s validation FAILED: %s", ticker_label, vr.errors)
        return None
    for w in vr.warnings:
        log.warning("%s: %s", ticker_label, w)

    if path.exists():
        df_old = pd.read_parquet(path)
        df_old[date_col] = pd.to_datetime(df_old[date_col])
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df = (df.drop_duplicates(subset=dedup_cols, keep="last")
            .sort_values(date_col)
            .reset_index(drop=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
