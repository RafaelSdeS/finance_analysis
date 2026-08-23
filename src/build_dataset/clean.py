"""clean.py — final pass: dedupe, inf->NaN, sort."""

import numpy as np


def clean_dataset(df):

    print()
    print("=" * 80)
    print("CLEANING DATASET")
    print("=" * 80)

    before = len(df)
    # ignore_index=True rather than a trailing `.copy()`: it returns a frame
    # with a fresh RangeIndex, which is unambiguously a new object, so the
    # in-place inf fix below can't trip SettingWithCopy. The old
    # `.drop_duplicates().copy()` bought the same guarantee by copying the
    # whole frame a second time -- ~1GB per row group at US scale, on a stage
    # that already churns several copies of it (this runs once per Pass-3 row
    # group, so every copy here is paid ~20 times over a full build).
    df = df.drop_duplicates(ignore_index=True)
    print(f"Removed duplicates: {before - len(df)}")

    # Growth rates (pct_change from a zero base) and ratios (zero denominator,
    # e.g. hl_ratio/adj_close) can produce literal inf — clean to NaN so it
    # never reaches training/inference.
    #
    # One column at a time, not `df[numeric_cols].replace([inf, -inf], nan)`:
    # that built a copy of every numeric column at once (~150 columns, ~1GB per
    # row group), produced a second copy from replace(), then assigned the
    # result back over the originals — three whole-frame-sized allocations to
    # fix a handful of cells. Per column the transient is one bool mask
    # (~1 byte/row) and the write only happens for columns that actually carry
    # an inf. Only floats are scanned: an integer column cannot hold inf.
    n_inf = 0
    for col in df.select_dtypes(include="floating").columns:
        mask = np.isinf(df[col].to_numpy())
        if mask.any():
            n_inf += int(mask.sum())
            df.loc[mask, col] = np.nan
    print(f"Replaced inf/-inf with NaN: {n_inf}")

    df = df.sort_values(["ticker", "trade_date"], ignore_index=True)

    return df
