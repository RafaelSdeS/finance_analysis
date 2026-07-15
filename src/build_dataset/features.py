"""features.py — per-ticker feature engineering (Pass 1 of compute_features_chunked):
CAGR backfill, dividend yield, price technicals, fundamental ratios/trends,
macro-adjusted returns, daily valuation re-anchoring, and the "advanced"
contextual features. Everything here operates on one ticker's rows at a time,
so it's safe to run on a ticker-batch without seeing the full universe
(contrast with cross_sectional.py, which needs the full universe at once).
"""

import numpy as np
import pandas as pd

from .cagr_handler import fill_cagr_columns


# =============================================================================
# FILL MISSING CAGR VALUES
# =============================================================================

def fill_missing_cagr(fundamentals):

    print()
    print("=" * 80)
    print("FILLING MISSING CAGR VALUES")
    print("=" * 80)

    # Group by ticker and apply CAGR filling
    # ponytail: split by ticker once via groupby instead of re-filtering the
    # full table on every loop iteration
    fundamentals_by_ticker = dict(tuple(fundamentals.groupby("ticker", sort=False)))
    dfs = []
    for ticker in sorted(fundamentals["ticker"].unique()):
        ticker_df = fundamentals_by_ticker[ticker].copy()

        # Track coverage before
        earnings_before = ticker_df["cagr_earnings_5y"].isna().sum() if "cagr_earnings_5y" in ticker_df.columns else 0
        revenue_before = ticker_df["cagr_revenue_5y"].isna().sum() if "cagr_revenue_5y" in ticker_df.columns else 0

        # Fill CAGR
        ticker_df = fill_cagr_columns(ticker_df)

        # Track coverage after
        earnings_after = ticker_df["cagr_earnings_5y_final"].isna().sum()
        revenue_after = ticker_df["cagr_revenue_5y_final"].isna().sum()

        dfs.append(ticker_df)

        print(f"{ticker}: earnings nulls {earnings_before} → {earnings_after}, revenue nulls {revenue_before} → {revenue_after}")

    fundamentals = pd.concat(dfs, ignore_index=True)
    fundamentals = fundamentals.sort_values(["ticker", "reference_date"])

    print(f"CAGR filling complete: {len(fundamentals)} total rows")

    return fundamentals


# =============================================================================
# COMPUTE DIVIDEND FEATURES
# =============================================================================

def compute_dividend_features(dataset, dividends):
    """Compute rolling dividend yield and frequency after dividends are loaded."""

    print()
    print("=" * 80)
    print("COMPUTING DIVIDEND FEATURES")
    print("=" * 80)

    # Calendar days, not trading-day rows (unlike return_12m's .rolling(252) —
    # this window is a date-based searchsorted over ex_date, so it needs a
    # true calendar year (365d), not the 252-trading-day row count used
    # elsewhere. Using 252 here previously dropped ~113 days (2.7-3.7 months)
    # of real trailing dividends -- e.g. an annual-payer would read a false
    # div_yield_12m=0 for that whole window every year (confirmed 2026-07-14).
    window = np.timedelta64(365, "D")

    # ponytail: split dividends by ticker once via groupby instead of
    # re-filtering the full table on every loop iteration
    dividends_by_ticker = dict(tuple(dividends.groupby("ticker", sort=False)))

    result = []
    for ticker, g in dataset.groupby("ticker", sort=False):
        g = g.sort_values("trade_date").copy()

        div = dividends_by_ticker.get(ticker, dividends.iloc[0:0]).sort_values("ex_date")

        if len(div) == 0:
            g["div_yield_12m"] = 0.0
            g["div_count_12m"] = 0
            result.append(g)
            continue

        # Trailing-252-day dividend sum/count at each trade_date, vectorized.
        # Window is (trade_date - 252d, trade_date]; searchsorted over sorted ex_dates
        # gives the count in O(log n), and cumulative sums give the value in the window.
        ex = div["ex_date"].to_numpy()
        cum_val = np.concatenate([[0.0], np.cumsum(div["value_per_share"].to_numpy())])
        td = g["trade_date"].to_numpy()

        hi = np.searchsorted(ex, td, side="right")           # ex_date <= trade_date
        lo = np.searchsorted(ex, td - window, side="right")  # ex_date <= trade_date - 252d
        count = hi - lo
        summed = cum_val[hi] - cum_val[lo]

        price = g["adj_close"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            g["div_yield_12m"] = np.where(price > 0, summed / price, 0.0)
        g["div_count_12m"] = count

        result.append(g)

    print(f"Dividend features added for {len(result)} tickers")
    return pd.concat(result, ignore_index=True)


# =============================================================================
# PRICE FEATURES
# =============================================================================

# log_return is computed by ROW adjacency (.shift(1)), not calendar-date
# adjacency. A ticker with a genuine raw-data hole (delisted/relisted under a
# recycled code, or an unfillable vendor collection gap -- e.g. UGPA3's
# confirmed 2010-2011 gap, in yf_collectors.FLAT_RUN_PADDING because yfinance
# has no real data for it either) would otherwise silently produce a fake
# multi-day/multi-year "single-day" return (confirmed: VBBR3/BRDT3 read
# -95.6% across a 14.9-year hole, UGPA3 +46.8% across a 499-day one,
# 2026-07-15 audit). Anything below this threshold is real trading-calendar
# noise already confirmed legitimate elsewhere (BHIA3/CCRO3 gaps up to
# 47-53 days for illiquid micro-caps) and must not be touched.
MAX_RETURN_GAP_DAYS = 120


def _rsi(series, n=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    # zero down-days in the window: RS is undefined by division (loss replaced with
    # NaN above to dodge the divide-by-zero), but RSI itself is well-defined -- 100
    # if there was any gain, 50 (neutral) if the window was perfectly flat. Without
    # this, both cases silently fall to NaN instead of their true value.
    zero_loss = loss == 0
    return rsi.where(~zero_loss, np.where(gain > 0, 100.0, 50.0))


def compute_price_features(df):

    print()
    print("=" * 80)
    print("COMPUTING PRICE FEATURES")
    print("=" * 80)

    result = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("trade_date")
        # Mask non-positive prices to NaN before log to avoid divide-by-zero warnings
        adj = g["adj_close"].where(g["adj_close"] > 0)
        prev_adj_close = adj.shift(1)
        g["log_return"]     = np.log(adj / prev_adj_close)
        gap_days = g["trade_date"].diff().dt.days
        large_gap = gap_days > MAX_RETURN_GAP_DAYS
        g.loc[large_gap, "log_return"] = np.nan
        # Vendor (BolsAI) stores adj_close at 2-decimal precision. For
        # deep-history microcaps with a large cumulative split/dividend
        # adjustment factor, the true adjusted price underflows toward that
        # precision floor: adj_close either rounds to exactly 0.00 (already
        # masked to NaN above) or gets pinned at a tiny nonzero constant
        # (e.g. 0.03) across several consecutive days while the real,
        # unrounded price keeps moving -- producing a spurious log_return of
        # exactly 0.0 for a real move (confirmed on BIOM3 and 27 others,
        # 2026-07-14 anomaly investigation). Flag rather than fix: there's no
        # way to recover the lost precision, and CLAUDE.md already documents
        # why adj_close must not be reconstructed from other sources.
        #
        # Must require exact 2dp quantization, not just magnitude < 0.05:
        # some tickers (e.g. TIMS3, down to 0.000568) are genuinely
        # low-priced with full float precision preserved -- a real number,
        # not a rounding artifact -- and must NOT be flagged.
        near_floor = (g["adj_close"] > 0) & (g["adj_close"] < 0.05)
        quantized = np.isclose(g["adj_close"], g["adj_close"].round(2))
        g["adj_close_precision_degraded"] = (near_floor & quantized).astype(int)

        # Overnight gap / intraday return: log_return decomposed into the
        # portion that accrued outside trading hours (today's open vs. prior
        # close) and the portion that accrued during the session (today's
        # close vs. today's open) -- overnight_gap + intraday_return ==
        # log_return by construction. adj_open (not raw open), same reasoning
        # as hl_ratio using adj_high/adj_low. overnight_gap spans the same
        # prior-close reference as log_return, so it needs the identical
        # MAX_RETURN_GAP_DAYS guard; intraday_return is same-day open->close
        # and can't straddle a collection gap, so it doesn't.
        adj_open = g["adj_open"].where(g["adj_open"] > 0)
        g["overnight_gap"]    = np.log(adj_open / prev_adj_close)
        g["intraday_return"]  = np.log(adj / adj_open)
        g.loc[large_gap, "overnight_gap"] = np.nan

        g["volatility_20d"] = g["log_return"].rolling(20).std()
        g["volatility_60d"] = g["log_return"].rolling(60).std()
        # Short-vs-long vol regime ratio: expanding (>1) or contracting (<1)
        # volatility, independent of any ticker's absolute vol level.
        g["volatility_ratio_20_60"] = g["volatility_20d"] / g["volatility_60d"]
        g["ma_20"]          = g["adj_close"].rolling(20).mean()
        g["ma_60"]          = g["adj_close"].rolling(60).mean()
        # Price relative to its own trend, not the raw MA level (which is an
        # absolute price and not comparable across tickers) -- same pattern
        # as hl_ratio below.
        g["price_vs_ma20"]  = g["adj_close"] / g["ma_20"]
        g["price_vs_ma60"]  = g["adj_close"] / g["ma_60"]
        # adj_high/adj_low, not raw high/low: raw and adjusted prices live on
        # different scales whenever the cumulative split adjustment != 1, and
        # mixing them made hl_ratio meaningless around splits.
        g["hl_ratio"]       = (g["adj_high"] - g["adj_low"]) / g["adj_close"]
        # True range: hl_ratio's blind spot is a gap day (price gaps overnight
        # then trades in a tight intraday range) -- true range also counts the
        # distance from the prior close, same guard as overnight_gap since it
        # shares the same prior-close reference.
        true_range = pd.concat([
            g["adj_high"] - g["adj_low"],
            (g["adj_high"] - prev_adj_close).abs(),
            (g["adj_low"] - prev_adj_close).abs(),
        ], axis=1).max(axis=1)
        g["true_range_ratio"] = true_range / g["adj_close"]
        g.loc[large_gap, "true_range_ratio"] = np.nan
        g["drawdown"]       = (g["adj_close"] - g["adj_close"].cummax()) / g["adj_close"].cummax()
        g["rsi_14"]         = _rsi(g["adj_close"], 14)
        # Volume relative to its own trailing average -- raw volume spans
        # orders of magnitude across the universe (blue chip vs. microcap)
        # and isn't comparable across tickers; this ratio is.
        g["volume_ratio_20d"] = g["volume"] / g["volume"].rolling(20).mean()
        # Amihud illiquidity: price impact per unit of currency traded -- a
        # liquidity measure, distinct from volume_ratio_20d (which only asks
        # whether today's volume is unusual for this ticker, not how much
        # price moves per unit traded).
        g["amihud_illiquidity"] = g["log_return"].abs() / (g["volume"] * g["adj_close"])
        g["return_1m"]      = g["log_return"].rolling(21).sum()
        g["return_3m"]      = g["log_return"].rolling(63).sum()
        g["return_6m"]      = g["log_return"].rolling(126).sum()
        g["return_12m"]     = g["log_return"].rolling(252).sum()
        result.append(g)

    print(f"Price features added for {len(result)} tickers")
    return pd.concat(result, ignore_index=True)


# =============================================================================
# FUNDAMENTAL FEATURES
# =============================================================================

def compute_fundamental_features(df):
    """Called on the fundamentals DataFrame BEFORE the asof merge."""

    print()
    print("=" * 80)
    print("COMPUTING FUNDAMENTAL FEATURES")
    print("=" * 80)

    result = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("reference_date")

        # Value signals (inverse/normalized forms not pre-computed by API).
        # earnings_yield is NOT computed here: recompute_valuation_daily()
        # runs after this (re-anchoring market_cap to the daily close), so a
        # net_income/market_cap computed at this stage would be stale the
        # moment that re-anchoring happens. compute_advanced_features()
        # computes it once, correctly, as 1/pl (post-re-anchoring) -- this
        # function used to also define it as net_income/market_cap, silently
        # overwritten by the later definition every time; removed as dead
        # code (confirmed 2026-07-15, no code path ever read this value).
        g["book_to_market"]        = g["equity"] / g["market_cap"]
        g["cash_ratio"]            = g["cash"] / g["current_liabilities"]
        g["net_debt_to_assets"]    = g["net_debt"] / g["total_assets"]
        g["working_capital_ratio"] = (g["current_assets"] - g["current_liabilities"]) / g["total_assets"]

        # YoY growth (4 quarters back)
        g["revenue_growth_yoy"]       = g["net_revenue"].pct_change(4, fill_method=None)
        g["earnings_growth_yoy"]      = g["net_income"].pct_change(4, fill_method=None)
        g["ebitda_growth_yoy"]        = g["ebitda"].pct_change(4, fill_method=None)
        g["total_assets_growth_yoy"]  = g["total_assets"].pct_change(4, fill_method=None)
        g["total_debt_growth_yoy"]    = g["total_debt"].pct_change(4, fill_method=None)

        # QoQ trend (sequential quarter diff)
        g["gross_margin_qoq"]  = g["gross_margin"].diff(1)
        g["net_margin_qoq"]    = g["net_margin"].diff(1)
        g["roe_qoq"]           = g["roe"].diff(1)
        g["debt_equity_qoq"]   = g["debt_equity"].diff(1)
        g["current_ratio_qoq"] = g["current_ratio"].diff(1)

        # Partial Piotroski F-Score (5-point; omits cash-flow components we lack)
        g["f_roa_positive"]        = (g["roa"] > 0).astype(int)
        g["f_roa_improving"]       = (g["roa"] > g["roa"].shift(4)).astype(int)
        g["f_margin_improving"]    = (g["gross_margin"] > g["gross_margin"].shift(4)).astype(int)
        g["f_leverage_decreasing"] = (g["debt_equity"] < g["debt_equity"].shift(4)).astype(int)
        g["f_liquidity_improving"] = (g["current_ratio"] > g["current_ratio"].shift(4)).astype(int)
        f_cols = ["f_roa_positive", "f_roa_improving", "f_margin_improving",
                  "f_leverage_decreasing", "f_liquidity_improving"]
        g["f_score"] = g[f_cols].sum(axis=1)

        result.append(g)

    print(f"Fundamental features added for {len(result)} tickers")
    return pd.concat(result, ignore_index=True)


# =============================================================================
# MACRO FEATURES
# =============================================================================

def compute_macro_features(df):
    """Requires log_return (from compute_price_features) and selic/ipca already merged."""

    print()
    print("=" * 80)
    print("COMPUTING MACRO FEATURES")
    print("=" * 80)

    # ponytail: selic/ipca are annual %; divide by 252 trading days for daily equivalent
    df["excess_return"]    = df["log_return"] - df["selic"] / 252
    df["real_return"]      = df["log_return"] - df["ipca"] / 252
    df["selic_trend_20d"]  = df["selic"] - df["selic"].shift(20)

    return df


# =============================================================================
# DAILY VALUATION RE-ANCHORING
# =============================================================================

def recompute_valuation_daily(df):
    """Re-anchor BolsAI valuation ratios to the daily close.

    The API computes pl/pvp/market_cap/etc. with the price on the filing date
    (close_price) and they stay frozen until the next quarter. Rescaling by
    close/close_price is exact for any ratio with price in the numerator,
    whatever denominator definition the API used (TTM vs quarterly).
    """

    print()
    print("=" * 80)
    print("RE-ANCHORING VALUATION RATIOS TO DAILY CLOSE")
    print("=" * 80)

    factor = (df["close"] / df["close_price"]).where(df["close_price"] > 0)

    # Split guard: right after a fundamental first becomes available (within 1 day),
    # an EXTREME jump (>200%) likely means an unrecorded split. Note: close_price
    # is from reference_date (quarter-end), while fundamentals_available_date is
    # 45-90 days later, so price can drift 20-50% in normal markets. Only flag
    # truly extreme jumps (3x) that are almost certainly splits, not price movement.
    # ponytail: threshold 1.5x → 3.0x to filter out legitimate bull markets
    near_filing = (df["trade_date"] - df["fundamentals_available_date"]).dt.days.between(0, 1)
    suspicious = near_filing & ((factor > 3.0) | (factor < 1 / 3.0))
    if suspicious.any():
        bad = sorted(df.loc[suspicious, "ticker"].unique())
        print(f"WARNING: close/close_price jump >200% within 1 day of filing date "
              f"for {len(bad)} tickers (likely unrecorded split): {bad[:20]}")

    # EV ratios first: only the market-cap leg of EV moves with price, so
    # recover the API's denominator from its own numbers before market_cap changes.
    ev_api = df["market_cap"] + df["net_debt"]
    for col in ("ev_ebit", "ev_ebitda"):
        if col in df.columns:
            denom = ev_api / df[col].where(df[col].abs() > 1e-12)
            df[col] = (df["market_cap"] * factor + df["net_debt"]) / denom

    # Ratios linear in price: scale by the price factor
    for col in ("pl", "pvp", "market_cap", "p_sr", "p_ebit", "p_ebitda", "p_assets"):
        if col in df.columns:
            df[col] = df[col] * factor

    # Inverse ratio (price in the denominator)
    if "book_to_market" in df.columns:
        df["book_to_market"] = df["book_to_market"] / factor

    # Availability flag: lets the model tell "no filing yet" (pre-2011 / pre-IPO)
    # apart from "average company" after the env's NaN→0 imputation
    df["has_fundamentals"] = df["reference_date"].notna().astype(float)

    # close_price (price at filing date) is now redundant and misleading.
    # fundamentals_available_date stays: the T31 validation gate needs it, and
    # "when did these numbers become public" is legitimate agent-visible state.
    df = df.drop(columns=["close_price"])

    print(f"Valuation ratios re-anchored for {len(df)} rows")
    return df


# =============================================================================
# ADVANCED CONTEXTUAL FEATURES (for conservative long-term allocation)
# =============================================================================

def compute_advanced_features(df):
    """
    Add context-aware, raw metrics (no thresholds or hardcoded rules).
    Model learns relationships from data, not from pre-baked heuristics.
    """

    print()
    print("=" * 80)
    print("COMPUTING ADVANCED CONTEXTUAL FEATURES")
    print("=" * 80)

    # --- DIVIDEND & PAYOUT ANALYSIS (raw, no thresholds) ---

    # Use LPA (lucro per ação = EPS) directly from API
    df["payout_ratio"] = df["div_value_recent"] / (df["lpa"] + 1e-8)

    # Dividend coverage: can EBITDA support annual dividend?
    # annual_dividend = div_value_recent * shares_outstanding
    #
    # NOT a "+1e-8 near-zero denominator" case like the other ratios below --
    # div_value_recent==0 (no dividend paid, ever) is the ORDINARY case for
    # ~27% of rows / 213 tickers (confirmed 2026-07-14), not rare distress.
    # A +1e-8 epsilon there computes ebitda/1e-8 -- a ~1e8x-inflated, finite
    # (so clean_dataset's inf->NaN pass never catches it) number as extreme as
    # 2.3e15, poisoning any scaler/model that touches this column. "Coverage"
    # is undefined, not infinite, when there's no dividend to cover -- NaN.
    annual_dividend = df["div_value_recent"] * df["shares_outstanding"]
    df["dividend_coverage_ratio"] = df["ebitda"] / annual_dividend.where(annual_dividend > 0)

    # --- EARNINGS QUALITY (raw signals, no classification) ---

    # Revenue-to-earnings trend: stable ratio suggests quality
    df["revenue_per_earning"] = df["net_revenue"] / (df["net_income"] + 1e-8)

    # YoY comparison: revenue growth aligned with earnings growth?
    df["revenue_vs_earnings_growth_delta"] = (
        df["revenue_growth_yoy"] - df["earnings_growth_yoy"]
    )

    # EBITDA margin as quality proxy (higher = better operational efficiency, but let model learn)
    df["ebitda_margin"] = df["ebitda"] / (df["net_revenue"] + 1e-8)

    # --- LIQUIDITY (raw volume vs. float -- lives here, not compute_price_features,
    # since shares_outstanding is a fundamentals column) ---

    # Turnover: % of the float traded today. Comparable across tickers of very
    # different sizes in a way raw volume never is -- distinct from
    # volume_ratio_20d (compute_price_features), which only asks whether
    # today's volume is unusual for THIS ticker's own recent norm.
    df["turnover_ratio"] = df["volume"] / df["shares_outstanding"]

    # --- FUNDAMENTAL FRESHNESS (raw days, model learns staleness impact) ---

    df["days_since_fundamental"] = (df["trade_date"] - df["reference_date"]).dt.days

    # --- WITHIN-TICKER HISTORICAL PERCENTILES (context for model) ---

    result = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("trade_date").reset_index(drop=True)

        # rolling.rank(method="max", pct=True) == share of window values <= current,
        # same as the old rolling.apply lambda but computed in cython (~1000x faster).
        # ponytail: NaNs are excluded from the window count here (old lambda counted them)
        window_252 = 252 * 5  # 5 years

        # Volatility percentile: where is current vol vs this stock's history?
        # Rolling (not a plain .rank()) so row i only sees rows <= i — a plain
        # rank() here would rank against the ticker's *future* volatility too.
        g["volatility_20d_percentile"] = g["volatility_20d"].rolling(
            window=window_252, min_periods=1
        ).rank(method="max", pct=True)
        g["volatility_60d_percentile"] = g["volatility_60d"].rolling(
            window=window_252, min_periods=1
        ).rank(method="max", pct=True)

        # Price percentile: is price high/low vs own history (last 5 years)?
        g["price_percentile_5y"] = g["adj_close"].rolling(
            window=window_252, min_periods=1
        ).rank(method="max", pct=True)

        # 1-year version: the standard "52-week high/low" framing, a distinct
        # signal from the 5y version for younger listings or a recent regime
        # change that 5 years of history would dilute. Same window as
        # drawdown_percentile below, for consistency.
        g["price_percentile_1y"] = g["adj_close"].rolling(
            window=252, min_periods=1
        ).rank(method="max", pct=True)

        # P/L (P/E) percentile within stock's history
        g["pl_percentile_5y"] = g["pl"].rolling(
            window=window_252, min_periods=1
        ).rank(method="max", pct=True)

        # Drawdown percentile: how deep is current drawdown vs historical?
        g["drawdown_percentile"] = g["drawdown"].rolling(
            window=252, min_periods=1
        ).rank(method="max", pct=True)

        result.append(g)

    df = pd.concat(result, ignore_index=True).reset_index(drop=True)

    # --- FUNDAMENTAL TREND SIGNALS (raw, no thresholds) ---

    # diff(4) must run over 4 real fiscal quarters, not 4 rows of this daily
    # panel — fundamentals are forward-filled for ~63 trading days between
    # filings, so diffing the daily panel directly is ~0 all quarter with a
    # 4-row blip right after each filing. Dedup to one row per (ticker,
    # reference_date), diff there, then map the quarterly trend back onto
    # every daily row.
    df = df.sort_values(["ticker", "reference_date"]).reset_index(drop=True)

    trend_cols = {
        "roe": "roe_trend_4q",
        "net_margin": "margin_trend_4q",
        "debt_equity": "debt_trend_4q",
        "roa": "roa_trend_4q",
    }
    result = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.copy()
        q = g.drop_duplicates("reference_date").set_index("reference_date").sort_index()
        for col, out in trend_cols.items():
            g[out] = g["reference_date"].map(q[col].diff(4))

        # Cumulative quarterly filing count per ticker: number of distinct
        # reference_date values seen so far (expanding count). Explains all
        # window-based NaNs (CAGR history, YoY, QoQ, trends). Non-decreasing by reference_date order.
        # Only assign where reference_date is not NaN (rows without fundamentals get NaN).
        q["n_quarters_cumulative"] = range(1, len(q) + 1)
        g["n_quarters_available"] = g["reference_date"].map(q["n_quarters_cumulative"])

        result.append(g)
    df = pd.concat(result, ignore_index=True)

    # --- VALUATION RELATIVE TO FUNDAMENTALS (raw relationships) ---

    # PEG ratio: P/L (P/E) relative to earnings growth
    df["peg_ratio"] = df["pl"] / (df["earnings_growth_yoy"] * 100 + 1e-8)

    # P/VP (P/B) relative to ROE (value signal: low P/VP + high ROE = cheap quality)
    df["pvp_to_roe_ratio"] = df["pvp"] / (df["roe"] + 1e-8)

    # Earnings yield (inverse P/L) vs macro rates
    df["earnings_yield"] = 1.0 / (df["pl"] + 1e-8)
    df["earnings_yield_vs_selic"] = df["earnings_yield"] - (df["selic"] / 100)

    # Flag columns: CAGR defined (only if *_final columns exist; they're populated
    # by fill_missing_cagr before this pipeline in production, but test fixtures may omit them)
    if "cagr_earnings_5y_final" in df.columns:
        df["cagr_earnings_defined"] = df["cagr_earnings_5y_final"].notna().astype(float)
    if "cagr_revenue_5y_final" in df.columns:
        df["cagr_revenue_defined"] = df["cagr_revenue_5y_final"].notna().astype(float)

    print(f"Advanced features computed for {len(df)} rows")
    return df
