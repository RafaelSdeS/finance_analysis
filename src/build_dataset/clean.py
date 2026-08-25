"""clean.py — final pass: dedupe, inf->NaN, sort."""

import numpy as np


def clean_dataset(df):

    print()
    print("=" * 80)
    print("CLEANING DATASET")
    print("=" * 80)

    before = len(df)
    # Everything here is in place. This runs once per Pass-3 row group (~20
    # times over a full build) on a frame that is ~400MB at BR scale and ~1GB
    # at US scale, and the caller keeps its own reference to it
    # (`batch = clean_dataset(batch)`) -- so a returned copy leaves BOTH frames
    # resident, and each further step stacks another. In place, a copy is
    # transient rather than cumulative.
    #
    # ponytail: and don't take the full-width copy at all when there's nothing
    # to drop. DataFrame.drop_duplicates() factorizes every one of the ~159
    # columns (159 int64 label arrays = ~360MB per BR row group) and then
    # rebuilds the whole frame via `self[~dup]` even when the mask is all-False.
    # Identical rows are a strict subset of rows sharing (ticker, trade_date),
    # so screen on those two columns first -- ~10MB -- and only pay the
    # full-width comparison on whatever that turns up. This is where Pass 3 ran
    # out of address space on 2026-08-23 (a 2MB allocation failed).
    key = ["ticker", "trade_date"]
    dupes = df.duplicated(subset=key, keep=False)
    if dupes.any():
        # keep="first" within the candidate rows gives the same verdict as
        # across the whole frame: any row identical to an earlier one shares
        # that row's key, so both are already in this subset, in the same
        # relative order.
        dupes.loc[dupes] = df.loc[dupes].duplicated().to_numpy()
        df.drop(index=df.index[dupes], inplace=True)
        df.reset_index(drop=True, inplace=True)
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

    # ponytail: Pass 1 already leaves every row group sorted by exactly this
    # key (compute_history_relative_features's own sort is the last thing that
    # touches the batch), so the unconditional sort was a full-frame copy per
    # row group to produce the order the frame was already in. The check is two
    # monotonicity scans and no allocation of consequence.
    if not (df["ticker"].is_monotonic_increasing
            and df.groupby("ticker", sort=False)["trade_date"].is_monotonic_increasing.all()):
        df.sort_values(["ticker", "trade_date"], ignore_index=True, inplace=True)

    return df
