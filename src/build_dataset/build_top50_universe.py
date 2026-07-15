"""
build_top50_universe.py — point-in-time top-50-by-volume universe filter.

Construction per TOP50_UNIVERSE_VALIDATION.md §1: at each quarterly rebalance
date, rank tickers by *trailing* 252-trading-day traded_amount (only data up
to and including that date), take the top 50, lock membership until the next
rebalance. Union across all rebalance periods recovers delisted names without
letting any single day's membership see future volume.

Run: python -m src.build_dataset.build_top50_universe
"""

import pandas as pd

from src.build_dataset.paths import OUTPUT_PATH, TOP50_MEMBERSHIP_PATH, TOP50_UNIVERSE_PATH

TOP_N = 50
TRAILING_DAYS = 252
REBALANCE_FREQ = "Q"  # ponytail: quarterly; pass "A" for annual
TOLERANCE_DAYS = 30  # a ticker with no trade in the last N days isn't "currently trading" at rebalance


def build_top50_membership(df: pd.DataFrame, top_n=TOP_N, trailing_days=TRAILING_DAYS,
                            rebalance_freq=REBALANCE_FREQ, tolerance_days=TOLERANCE_DAYS) -> pd.DataFrame:
    """Return the (ticker, period_id, start, end) membership table — one row
    per ticker per locked rebalance period it qualified for."""
    df = df[["ticker", "trade_date", "traded_amount"]].sort_values(["ticker", "trade_date"])
    trail_amt = df.groupby("ticker")["traded_amount"].transform(
        lambda s: s.rolling(trailing_days, min_periods=trailing_days).sum()
    )

    all_dates = df["trade_date"].drop_duplicates().sort_values()
    rebalance_dates = all_dates.groupby(all_dates.dt.to_period(rebalance_freq)).max().reset_index(drop=True)
    periods = pd.DataFrame({"start": rebalance_dates}).reset_index(drop=True)
    periods["period_id"] = periods.index
    periods["end"] = periods["start"].shift(-1).fillna(df["trade_date"].max() + pd.Timedelta(days=1))

    snap_source = df.assign(trail_amt=trail_amt).dropna(subset=["trail_amt"]).sort_values("trade_date")
    cross = periods[["start", "period_id"]].merge(
        pd.DataFrame({"ticker": df["ticker"].unique()}), how="cross"
    ).rename(columns={"start": "trade_date"}).sort_values("trade_date")

    snaps = pd.merge_asof(
        cross, snap_source[["ticker", "trade_date", "trail_amt"]],
        on="trade_date", by="ticker", direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days),
    )

    membership = (
        snaps.dropna(subset=["trail_amt"])
        .sort_values("trail_amt", ascending=False)
        .groupby("period_id")
        .head(top_n)[["period_id", "ticker"]]
        .merge(periods[["period_id", "start", "end"]], on="period_id")
    )
    return membership.sort_values(["ticker", "start"]).reset_index(drop=True)


def filter_to_top50_universe(df: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Restrict df to rows whose (ticker, trade_date) falls in a period the
    ticker qualified for. All original columns preserved, no row mutation."""
    df = df.sort_values("trade_date")
    periods = membership[["period_id", "start"]].drop_duplicates().sort_values("start")
    tagged = pd.merge_asof(df, periods, left_on="trade_date", right_on="start", direction="backward")
    member_idx = pd.MultiIndex.from_frame(membership[["period_id", "ticker"]])
    in_universe = pd.MultiIndex.from_frame(tagged[["period_id", "ticker"]]).isin(member_idx)
    return df.loc[in_universe].reset_index(drop=True)


def main():
    df = pd.read_parquet(OUTPUT_PATH)
    membership = build_top50_membership(df)
    universe_df = filter_to_top50_universe(df, membership)

    universe_df.to_parquet(TOP50_UNIVERSE_PATH, index=False)
    membership.to_parquet(TOP50_MEMBERSHIP_PATH, index=False)

    print(f"tickers ever in top-{TOP_N}: {membership['ticker'].nunique()}")
    print(f"rows: {len(universe_df)} / {len(df)} ({len(universe_df) / len(df):.1%})")
    print(f"wrote {TOP50_UNIVERSE_PATH}")
    print(f"wrote {TOP50_MEMBERSHIP_PATH}")


if __name__ == "__main__":
    main()
