"""
fred_collectors.py — FRED (St. Louis Fed) macro series, for the US pipeline.

Keyless: fredgraph.csv?id={series} returns a series' full history in one
request (verified 2026-07-28: CPIAUCNS -> 1913-01-01..present, 1,363 rows).
Unlike BCB's collect_macro, no date-chunking or incremental fetch is needed —
each run refetches the whole series (cheap: ~1-2k rows, one request) and
_merge_save's dedup keeps it idempotent, same contract as every other
collector. See config.py's FRED_SERIES for per-series frequency/units.
"""

import io
import logging

import pandas as pd

from . import checkpoint, client, config, validate
from .collectors import _merge_save

log = logging.getLogger(__name__)


def _fetch_series(c, series_id: str) -> pd.DataFrame:
    text = client.get_text(c, "fredgraph.csv", {"id": series_id})
    df = pd.read_csv(io.StringIO(text))
    df.columns = ["reference_date", series_id]
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    # FRED marks missing observations "." in the CSV, not blank -> coerce, then drop.
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df.dropna()


def collect_macro_us(mode: str = "full_scale"):
    c = client.make_client(config.FRED_BASE)
    cp = checkpoint.load("macro_us", mode)
    try:
        for name, series_id in config.FRED_SERIES.items():
            path = config.US_MACRO_DIR / f"{name}.parquet"
            try:
                df = _fetch_series(c, series_id)
            except Exception as e:
                log.error("fred %s (%s): fetch failed: %s", name, series_id, e)
                continue
            df = df.rename(columns={series_id: name})

            saved = _merge_save(df, path, "reference_date",
                                 lambda x: validate.validate_macro(x, name), f"macro_us/{name}")
            if saved is not None:
                cp[name] = {"last_date": str(saved["reference_date"].max().date()), "rows": len(saved)}
                checkpoint.save("macro_us", mode, cp)
                log.info("fred %s: %d total rows (%s -> %s)", name, len(saved),
                          saved["reference_date"].min().date(), saved["reference_date"].max().date())
    finally:
        c.close()


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL)
    collect_macro_us()
