"""clean.py — final pass: dedupe, inf->NaN, sort."""

import numpy as np


def clean_dataset(df):

    print()
    print("=" * 80)
    print("CLEANING DATASET")
    print("=" * 80)

    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed duplicates: {before - len(df)}")

    # Growth rates (pct_change from a zero base) and ratios (zero denominator,
    # e.g. hl_ratio/adj_close) can produce literal inf — clean to NaN so it
    # never reaches training/inference.
    numeric_cols = df.select_dtypes(include="number").columns
    n_inf = np.isinf(df[numeric_cols]).sum().sum()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    print(f"Replaced inf/-inf with NaN: {n_inf}")

    df = df.sort_values(["ticker", "trade_date"]).reset_index(drop=True)

    return df
