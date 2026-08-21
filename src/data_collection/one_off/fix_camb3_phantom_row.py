"""
fix_camb3_phantom_row.py — one-off repair for CAMB3's single phantom
non-trading row in raw price data.

Background (docs/DATA_LAYER_FOLLOWUP_FINDINGS.md, found via
tests/data_collection/test_br_data_quality.py): data/raw/br/prices/CAMB3.parquet
has exactly one row (2019-08-15) where open/high/low/close/adj_* are all NaN
and volume is 0 -- a phantom non-trading day from the raw source, not a real
quote and not a merge artifact. Stage 2 (build_dataset.loaders.load_prices)
already drops any such all-NaN-OHLC row generically going forward; this
one-off just cleans the already-collected raw file so the raw-data quality
sweep (test_br_data_quality.py) is clean at the source too.

Run from project root: python -m src.data_collection.one_off.fix_camb3_phantom_row
"""

import logging

import pandas as pd

from .. import config

log = logging.getLogger(__name__)

TICKER = "CAMB3"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = config.PRICES_DIR / f"{TICKER}.parquet"
    df = pd.read_parquet(path)

    ohlc_cols = ["open", "high", "low", "close"]
    all_nan = df[ohlc_cols].isna().all(axis=1)
    if not all_nan.any():
        log.info("%s: no phantom all-NaN OHLC rows found, nothing to do", TICKER)
        return

    log.info("%s: dropping %d phantom row(s): %s", TICKER, int(all_nan.sum()),
              df.loc[all_nan, "trade_date"].dt.date.tolist())
    df = df[~all_nan].reset_index(drop=True)
    df.to_parquet(path, index=False)
    log.info("%s: written to %s (%d rows)", TICKER, path, len(df))


if __name__ == "__main__":
    main()
