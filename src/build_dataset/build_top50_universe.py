"""
build_top50_universe.py — point-in-time top-50-by-volume universe filter.

Construction per TOP50_UNIVERSE_VALIDATION.md §1: at each quarterly rebalance
date, rank tickers by *trailing* 252-trading-day traded_amount (only data up
to and including that date), take the top 50, lock membership until the next
rebalance. Union across all rebalance periods recovers delisted names without
letting any single day's membership see future volume.

Run: python -m src.build_dataset.build_top50_universe
       python -m src.build_dataset.build_top50_universe --top-n 150 --membership-only
"""

import argparse

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
    ticker qualified for. All original columns preserved, no row mutation.

    ponytail: merge_asof only against the (ticker, trade_date) key columns,
    not the full ~165-column df — tagging the whole wide frame doubled peak
    memory (df + tagged copy) and OOM-killed this step on a 1.3M-row dataset.
    """
    key = df[["ticker", "trade_date"]].sort_values("trade_date").reset_index()
    periods = membership[["period_id", "start"]].drop_duplicates().sort_values("start")
    tagged = pd.merge_asof(key, periods, left_on="trade_date", right_on="start", direction="backward")
    member_idx = pd.MultiIndex.from_frame(membership[["period_id", "ticker"]])
    in_universe = pd.MultiIndex.from_frame(tagged[["period_id", "ticker"]]).isin(member_idx)
    keep_idx = tagged.loc[in_universe, "index"]
    return df.loc[keep_idx].sort_values("trade_date").reset_index(drop=True)


def zero_fill_missing_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """ponytail: zero-fill fundamental columns where has_fundamentals=0. Deliberate choice:
    when a ticker hasn't reported yet, its fundamentals are unknown (not zero earnings).
    Replace NaN with 0 to signal this to the agent explicitly — the has_fundamentals
    flag provides context ("coverage started here"). Alternative: drop rows/tickers
    lacking coverage (simpler but introduces survivorship bias, defeats the point of
    keeping delisted names). This preserves the full time series while making it clear
    to the agent: zero fundamentals + has_fundamentals=0 means "not yet reported",
    not "confirmed zero earnings". In-place mutation to avoid OOM."""
    fundamental_cols = [
        'pl', 'pvp', 'ev_ebitda', 'ev_ebit', 'p_ebitda', 'p_ebit', 'p_sr',
        'lpa', 'vpa', 'gross_margin', 'net_margin', 'ebitda_margin', 'ebit_margin',
        'roe', 'roa', 'roic', 'ebit_over_assets', 'asset_turnover', 'p_assets',
        'current_ratio', 'debt_equity', 'net_debt_equity', 'net_debt_ebitda', 'net_debt_ebit',
        'cagr_revenue_5y', 'cagr_earnings_5y', 'net_income', 'equity', 'net_revenue',
        'total_debt', 'ebitda', 'ebit', 'net_debt', 'cash', 'total_assets',
        'current_assets', 'current_liabilities', 'book_to_market', 'cash_ratio',
        'net_debt_to_assets', 'working_capital_ratio', 'revenue_growth_yoy',
        'earnings_growth_yoy', 'ebitda_growth_yoy', 'total_assets_growth_yoy',
        'total_debt_growth_yoy', 'gross_margin_qoq', 'net_margin_qoq', 'roe_qoq',
        'debt_equity_qoq', 'current_ratio_qoq', 'cagr_earnings_5y_final',
        'cagr_revenue_5y_final', 'payout_ratio', 'dividend_coverage_ratio',
        'revenue_per_earning', 'revenue_vs_earnings_growth_delta',
        'peg_ratio', 'pvp_to_roe_ratio', 'earnings_yield', 'earnings_yield_vs_selic',
        # Sibling fundamental-derived columns missed by the original list --
        # left NaN on has_fundamentals==0 rows while e.g. 'roe' above got
        # zero-filled for the same rows, an inconsistent missing-data signal
        # (docs/TOP50_UNIVERSE_ML_READINESS_AUDIT.md §1.3).
        'market_cap', 'filing_lag_days', 'days_since_fundamental',
        'pl_zscore_sector', 'pvp_zscore_sector', 'roe_zscore_sector', 'debt_equity_zscore_sector',
        'pl_percentile_5y',
        'f_roa_positive', 'f_roa_improving', 'f_margin_improving',
        'f_leverage_decreasing', 'f_liquidity_improving', 'f_score',
        'had_negative_earnings_5y',
        'roe_trend_4q', 'margin_trend_4q', 'debt_trend_4q', 'roa_trend_4q',
    ]
    mask = df['has_fundamentals'] == 0
    for col in fundamental_cols:
        if col in df.columns:
            df.loc[mask, col] = 0
    return df


def main(top_n=TOP_N, membership_only=False):
    print("Loading data...")
    df = pd.read_parquet(OUTPUT_PATH)
    total_rows = len(df)

    print(f"Building membership (top_n={top_n})...")
    membership = build_top50_membership(df[["ticker", "trade_date", "traded_amount"]], top_n=top_n)

    membership_path = (TOP50_MEMBERSHIP_PATH if top_n == TOP_N
                        else TOP50_MEMBERSHIP_PATH.with_name(f"top{top_n}_universe_membership.parquet"))
    if membership_only:
        membership.to_parquet(membership_path, index=False)
        print(f"\n✓ tickers ever in top-{top_n}: {membership['ticker'].nunique()}")
        print(f"✓ wrote {membership_path}")
        return

    print("Filtering to top-50 universe...")
    universe_df = filter_to_top50_universe(df, membership)
    del df  # Free memory immediately after filtering

    print("Zero-filling missing fundamentals...")
    universe_df = zero_fill_missing_fundamentals(universe_df)

    # ponytail: trim to earliest date with any fundamentals (no pre-fundamental noise)
    earliest_fund = universe_df[universe_df['has_fundamentals'] == 1]['trade_date'].min()
    rows_before = len(universe_df[universe_df['trade_date'] < earliest_fund])
    universe_df = universe_df[universe_df['trade_date'] >= earliest_fund].reset_index(drop=True)

    print("Writing outputs...")
    universe_df.to_parquet(TOP50_UNIVERSE_PATH, index=False)
    membership.to_parquet(membership_path, index=False)

    print(f"\n✓ tickers ever in top-{top_n}: {membership['ticker'].nunique()}")
    print(f"✓ rows after trim to earliest fundamental: {len(universe_df)} (removed {rows_before} pre-fundamental rows)")
    print(f"✓ coverage vs original full dataset: {len(universe_df) / total_rows:.1%}")
    print(f"✓ wrote {TOP50_UNIVERSE_PATH}")
    print(f"✓ wrote {membership_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--membership-only", action="store_true",
                         help="skip the filtered wide-universe parquet, write only the membership table")
    args = parser.parse_args()
    main(top_n=args.top_n, membership_only=args.membership_only)
