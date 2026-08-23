"""
br/macro.py — BCB SGS collector (SELIC/CDI/IPCA), keyless.

Split out of collectors.py (docs/DATA_LAYER_ORGANIZATION_PLAN.md §O3): the
only free, keyless, live-in-every-mode collector in a module whose docstring
and every other function assume a paid BolsAI key.
"""

import logging
from datetime import datetime

import httpx
import pandas as pd

from .. import checkpoint, client, config, validate
from ..storage import _chunk_dates, _merge_save

log = logging.getLogger(__name__)


def collect_macro(mode: str):
    # BCB needs "bcdata.sgs.{id}/dados" (dot, not slash) — base_url joining would
    # mangle it, so use a baseless client and pass the full URL.
    c = client.make_client("")
    cp = checkpoint.load("macro", mode)
    try:
        for name, sid in config.BCB_SERIES.items():
            path = config.MACRO_DIR / f"{name}.parquet"
            # Checkpoint only trusted if the file it describes still exists -- same
            # wiped-file/stale-checkpoint bug yf/_common.py's _seed_last_date guards
            # against (its docstring has the full incident writeup).
            start = cp.get(name, {}).get("last_date") if path.exists() else None
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
                # BCB returns a bare object (not a list) when the range has exactly
                # one data point — normalize before extending, or dict iteration
                # silently corrupts rows with its keys instead of the record.
                if isinstance(d, dict):
                    d = [d]
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
