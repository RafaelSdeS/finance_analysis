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
            g["div_value_12m"] = 0.0
            result.append(g)
            continue

        # Per-event yield at the dividend's OWN ex-date close, not today's
        # adj_close: adj_close is discounted backward by every dividend/split
        # since, so dividing a historical nominal payment by it overstates
        # yield the further back in time the row sits -- confirmed on BBAS3,
        # 2010 read a 6.8% yield vs the true 2.9% (2026-07-23 audit). "close"
        # (raw, unadjusted) shares value_per_share's own nominal basis at
        # that date, so this ratio is immune to any later adjustment and to
        # a split falling inside the trailing window (previously mixed
        # pre/post-split nominal payments over one post-split price).
        px = g[["trade_date", "close"]].rename(columns={"trade_date": "ex_date"})
        div = pd.merge_asof(div, px, on="ex_date", direction="backward")
        with np.errstate(divide="ignore", invalid="ignore"):
            event_yield = np.where(div["close"] > 0, div["value_per_share"] / div["close"], 0.0)

        # Trailing-252-day dividend sum/count at each trade_date, vectorized.
        # Window is (trade_date - 252d, trade_date]; searchsorted over sorted ex_dates
        # gives the count in O(log n), and cumulative sums give the value in the window.
        ex = div["ex_date"].to_numpy()
        cum_yield = np.concatenate([[0.0], np.cumsum(event_yield)])
        # Nominal (not yield) trailing sum -- true "annual dividend per share"
        # for payout_ratio/dividend_coverage_ratio (compute_advanced_features),
        # which need actual currency, not a price-normalized ratio.
        # div_value_recent (merge_dividends) is only the single MOST RECENT
        # event's nominal value; treating that one payment as if it were the
        # full year's dividend mislabeled a quarterly-or-less-frequent payout
        # as annual, and stair-stepped discontinuously at every ex-date
        # (2026-07-24 audit). value_per_share is already nominal at its own
        # ex-date, so summing it directly (unlike event_yield) needs no price
        # normalization and is immune to adjustment/split basis by construction.
        cum_value = np.concatenate([[0.0], np.cumsum(div["value_per_share"].to_numpy())])
        td = g["trade_date"].to_numpy()

        hi = np.searchsorted(ex, td, side="right")           # ex_date <= trade_date
        lo = np.searchsorted(ex, td - window, side="right")  # ex_date <= trade_date - 252d
        count = hi - lo
        g["div_yield_12m"] = cum_yield[hi] - cum_yield[lo]
        g["div_count_12m"] = count
        g["div_value_12m"] = cum_value[hi] - cum_value[lo]

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
        # price moves per unit traded). traded_amount (raw currency volume,
        # already on-disk), not volume*adj_close: adj_close is discounted
        # backward by every dividend/split since, understating true traded
        # currency the further back in history a row sits and inflating
        # Amihud with a secular drift unrelated to actual liquidity
        # (2026-07-23 audit). traded_amount needs no split/merger rescaling
        # either -- dollar volume is invariant to both (same currency changes
        # hands regardless of how many shares or what price it's split into).
        g["amihud_illiquidity"] = g["log_return"].abs() / g["traded_amount"]
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

# pct_change(4)/diff(1)/shift(4) below index by ROW POSITION, not calendar
# time -- correct only when fundamentals rows are perfectly contiguous
# quarters. A vendor-missing quarter (a real, if uncommon, gap -- see
# quality_filters.filter_excessive_filing_lag's own rows-dropped note) leaves
# the row count unchanged but silently stretches "4 quarters back" to 15+
# months, mislabeling a much longer window as YoY/QoQ (2026-07-23 audit).
# These bounds bracket normal quarterly spacing (real quarter-ends run
# 89-92 days apart) with margin, while a single skipped quarter (~180d+ for
# QoQ, ~456d+ for a 4-quarter YoY window) falls clearly outside them.
QOQ_GAP_DAYS = (60, 120)     # ~1 real quarter apart
YOY_GAP_DAYS = (300, 400)    # ~4 real contiguous quarters (1 year) apart


def _within_calendar_gap(dates: pd.Series, lookback: int, lo_days: int, hi_days: int) -> pd.Series:
    """True where the row `lookback` ROWS back is also `lookback` real
    quarters back in CALENDAR time (between lo_days and hi_days apart) --
    i.e. genuinely contiguous quarters, not `lookback` rows with one or more
    quarters missing between them."""
    gap = (dates - dates.shift(lookback)).dt.days
    return gap.between(lo_days, hi_days)


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
        yoy_ok = _within_calendar_gap(g["reference_date"], 4, *YOY_GAP_DAYS)
        g["revenue_growth_yoy"]       = g["net_revenue"].pct_change(4, fill_method=None).where(yoy_ok)
        g["earnings_growth_yoy"]      = g["net_income"].pct_change(4, fill_method=None).where(yoy_ok)
        g["ebitda_growth_yoy"]        = g["ebitda"].pct_change(4, fill_method=None).where(yoy_ok)
        g["total_assets_growth_yoy"]  = g["total_assets"].pct_change(4, fill_method=None).where(yoy_ok)
        g["total_debt_growth_yoy"]    = g["total_debt"].pct_change(4, fill_method=None).where(yoy_ok)

        # QoQ trend (sequential quarter diff)
        qoq_ok = _within_calendar_gap(g["reference_date"], 1, *QOQ_GAP_DAYS)
        g["gross_margin_qoq"]  = g["gross_margin"].diff(1).where(qoq_ok)
        g["net_margin_qoq"]    = g["net_margin"].diff(1).where(qoq_ok)
        g["roe_qoq"]           = g["roe"].diff(1).where(qoq_ok)
        g["debt_equity_qoq"]   = g["debt_equity"].diff(1).where(qoq_ok)
        g["current_ratio_qoq"] = g["current_ratio"].diff(1).where(qoq_ok)

        # Partial Piotroski F-Score (5-point; omits cash-flow components we lack).
        # The 4 *_improving/*_decreasing components compare against shift(4)
        # (same window as YoY above) -- guarded the same way, but as a
        # boolean comparison rather than a diff, so the guard NaNs the
        # comparison result directly (float, not the plain int the unguarded
        # version used, to hold NaN) rather than masking an intermediate series.
        g["f_roa_positive"]        = (g["roa"] > 0).astype(float)
        g["f_roa_improving"]       = (g["roa"] > g["roa"].shift(4)).where(yoy_ok).astype(float)
        g["f_margin_improving"]    = (g["gross_margin"] > g["gross_margin"].shift(4)).where(yoy_ok).astype(float)
        g["f_leverage_decreasing"] = (g["debt_equity"] < g["debt_equity"].shift(4)).where(yoy_ok).astype(float)
        g["f_liquidity_improving"] = (g["current_ratio"] > g["current_ratio"].shift(4)).where(yoy_ok).astype(float)
        f_cols = ["f_roa_positive", "f_roa_improving", "f_margin_improving",
                  "f_leverage_decreasing", "f_liquidity_improving"]
        # skipna=False: a Piotroski-style composite score with an undefined
        # component (post-gap-guard NaN) is itself undefined, not a partial
        # score silently missing a fifth of its weight.
        g["f_score"] = g[f_cols].sum(axis=1, skipna=False)

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

    # selic (SGS 11) is already a DAILY rate in percent (e.g. 0.0692 means
    # 0.0692%/day) -- NOT annual. ipca (SGS 433) is a MONTHLY rate in percent.
    # The previous version treated both as annual and divided by 252, which
    # under-subtracted the daily risk-free rate by ~2.5x (excess_return) and
    # over-subtracted daily inflation by ~8.3x (real_return) -- confirmed via
    # 2026-07-23 audit against the built dataset (mean real_return read
    # -0.165%/day, an economically impossible sustained real loss).
    df["excess_return"]    = df["log_return"] - np.log1p(df["selic"] / 100)
    df["real_return"]      = df["log_return"] - np.log1p(df["ipca"] / 100) / 21

    # selic_trend_20d is NOT computed here: it's merged in by merge_macro()
    # (merge.py) directly off the raw daily selic series, before any
    # ticker-batching exists. Computing it here on a ticker-blocked batch
    # (all of ticker A's rows, then all of ticker B's) was found to leak
    # across the batch boundary regardless of how it was windowed -- a plain
    # df["selic"].shift(20) mixed unrelated tickers' dates outright, and even
    # a per-batch date-dedup approach still mixed dates that happen to be
    # decades apart whenever a batch's own ticker composition left gaps in
    # its observed calendar (confirmed 2026-07-23 audit + regression check).
    # The raw macro series has one row per real trading day independent of
    # any ticker, which is the only grid this can be computed correctly on.

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

# Floor for the rolling-percentile features below (~1 trading quarter) --
# matches cross_sectional.BETA_MIN_PERIODS's convention for "enough history
# to not be a noisy/degenerate estimate."
PERCENTILE_MIN_PERIODS = 60


def _safe_ratio(numerator, denominator, min_abs=1e-6):
    """numerator / denominator, NaN where |denominator| isn't meaningfully
    away from zero. Replaces the `x / (y + 1e-8)` pattern: that guard avoids
    a literal division-by-zero crash, but when y==0 is the ORDINARY case
    (not rare distress) it produces a finite-but-astronomical value instead
    (e.g. ebitda/1e-8) -- finite means clean_dataset's inf->NaN pass never
    catches it. Confirmed up to 1e15-1e16 in the real dataset across several
    ratios below (see docs/TOP50_UNIVERSE_ML_READINESS_AUDIT.md §1.1)."""
    return numerator / denominator.where(denominator.abs() > min_abs)


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

    # div_value_12m (compute_dividend_features: trailing-12m nominal sum of
    # per-event dividends), not div_value_recent -- div_value_recent is only
    # the single MOST RECENT payment, which mislabels a quarterly-or-less-
    # frequent payout as if it were the whole year's dividend and stair-steps
    # discontinuously at every ex-date (2026-07-24 audit). div_value_12m is
    # already on the same trailing-window convention as div_yield_12m/
    # div_count_12m, so payout_ratio/dividend_coverage_ratio are now
    # consistent with the rest of this pipeline's dividend features.

    # Use LPA (lucro per ação = EPS) directly from API
    df["payout_ratio"] = _safe_ratio(df["div_value_12m"], df["lpa"])

    # Dividend coverage: can EBITDA support the trailing-12m dividend?
    # annual_dividend = div_value_12m * shares_outstanding
    #
    # div_value_12m==0 (no dividend paid in the trailing window) is the
    # ORDINARY case for a large share of rows, not rare distress, so this
    # uses a >0 guard (annual_dividend is never legitimately negative) rather
    # than _safe_ratio's abs()-near-zero guard.
    annual_dividend = df["div_value_12m"] * df["shares_outstanding"]
    df["dividend_coverage_ratio"] = df["ebitda"] / annual_dividend.where(annual_dividend > 0)

    # --- EARNINGS QUALITY (raw signals, no classification) ---

    # Revenue-to-earnings trend: stable ratio suggests quality
    df["revenue_per_earning"] = _safe_ratio(df["net_revenue"], df["net_income"])

    # YoY comparison: revenue growth aligned with earnings growth?
    df["revenue_vs_earnings_growth_delta"] = (
        df["revenue_growth_yoy"] - df["earnings_growth_yoy"]
    )

    # EBITDA margin as quality proxy (higher = better operational efficiency, but let model learn)
    df["ebitda_margin"] = _safe_ratio(df["ebitda"], df["net_revenue"])

    # --- LIQUIDITY (raw volume vs. float -- lives here, not compute_price_features,
    # since shares_outstanding is a fundamentals column) ---

    # Turnover: % of the float traded today. Comparable across tickers of very
    # different sizes in a way raw volume never is -- distinct from
    # volume_ratio_20d (compute_price_features), which only asks whether
    # today's volume is unusual for THIS ticker's own recent norm.
    df["turnover_ratio"] = df["volume"] / df["shares_outstanding"]

    # --- FUNDAMENTAL FRESHNESS (raw days, model learns staleness impact) ---

    # Staleness must be measured from when the market actually SAW these
    # numbers (fundamentals_available_date, the real CVM filing/statutory
    # date merge_prices_and_fundamentals attached), not from reference_date
    # (the fiscal quarter-end the numbers describe) -- the two differ by the
    # 45-90+ day filing lag itself, so measuring from reference_date silently
    # overstated every row's true information age by that same lag
    # (2026-07-24 audit).
    df["days_since_fundamental"] = (df["trade_date"] - df["fundamentals_available_date"]).dt.days

    # --- WITHIN-TICKER HISTORICAL PERCENTILES (context for model) ---

    result = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("trade_date").reset_index(drop=True)

        # rolling.rank(method="max", pct=True) == share of window values <= current,
        # same as the old rolling.apply lambda but computed in cython (~1000x faster).
        # ponytail: NaNs are excluded from the window count here (old lambda counted them)
        window_252 = 252 * 5  # 5 years

        # min_periods floor (not 1): a percentile ranked against a 1-3 row
        # window is trivially 100th-percentile and carries no real
        # information -- every OTHER rolling feature in this pipeline (beta,
        # volatility, zhist) leaves a NaN warm-up until its window has enough
        # history to mean something; these percentiles were the one
        # exception, silently emitting degenerate 1.0s for young listings'
        # earliest rows (2026-07-23 audit). PERCENTILE_MIN_PERIODS matches
        # cross_sectional.BETA_MIN_PERIODS's ~1-quarter convention.

        # Volatility percentile: where is current vol vs this stock's history?
        # Rolling (not a plain .rank()) so row i only sees rows <= i — a plain
        # rank() here would rank against the ticker's *future* volatility too.
        g["volatility_20d_percentile"] = g["volatility_20d"].rolling(
            window=window_252, min_periods=PERCENTILE_MIN_PERIODS
        ).rank(method="max", pct=True)
        g["volatility_60d_percentile"] = g["volatility_60d"].rolling(
            window=window_252, min_periods=PERCENTILE_MIN_PERIODS
        ).rank(method="max", pct=True)

        # Price percentile: is price high/low vs own history (last 5 years)?
        g["price_percentile_5y"] = g["adj_close"].rolling(
            window=window_252, min_periods=PERCENTILE_MIN_PERIODS
        ).rank(method="max", pct=True)

        # 1-year version: the standard "52-week high/low" framing, a distinct
        # signal from the 5y version for younger listings or a recent regime
        # change that 5 years of history would dilute. Same window as
        # drawdown_percentile below, for consistency.
        g["price_percentile_1y"] = g["adj_close"].rolling(
            window=252, min_periods=PERCENTILE_MIN_PERIODS
        ).rank(method="max", pct=True)

        # P/L (P/E) percentile within stock's history
        g["pl_percentile_5y"] = g["pl"].rolling(
            window=window_252, min_periods=PERCENTILE_MIN_PERIODS
        ).rank(method="max", pct=True)

        # Drawdown percentile: how deep is current drawdown vs historical?
        g["drawdown_percentile"] = g["drawdown"].rolling(
            window=252, min_periods=PERCENTILE_MIN_PERIODS
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
    df["peg_ratio"] = _safe_ratio(df["pl"], df["earnings_growth_yoy"] * 100)

    # P/VP (P/B) relative to ROE (value signal: low P/VP + high ROE = cheap quality)
    df["pvp_to_roe_ratio"] = _safe_ratio(df["pvp"], df["roe"])

    # Earnings yield (inverse P/L) vs macro rates. selic is a daily rate
    # (see compute_macro_features) -- annualize it before comparing to the
    # already-annual earnings_yield, or the macro term is ~250x too small
    # to mean anything.
    df["earnings_yield"] = _safe_ratio(1.0, df["pl"])
    selic_annualized = (1 + df["selic"] / 100) ** 252 - 1
    df["earnings_yield_vs_selic"] = df["earnings_yield"] - selic_annualized

    # Flag columns: CAGR defined (only if *_final columns exist; they're populated
    # by fill_missing_cagr before this pipeline in production, but test fixtures may omit them)
    if "cagr_earnings_5y_final" in df.columns:
        df["cagr_earnings_defined"] = df["cagr_earnings_5y_final"].notna().astype(float)
    if "cagr_revenue_5y_final" in df.columns:
        df["cagr_revenue_defined"] = df["cagr_revenue_5y_final"].notna().astype(float)

    print(f"Advanced features computed for {len(df)} rows")
    return df


# =============================================================================
# HISTORY-RELATIVE (PER-TICKER OWN-HISTORY) FEATURES
# =============================================================================

# "How unusual is this value relative to THIS company's own trailing
# distribution" (docs/PER_TICKER_SCALING_PLAN.md, R1) -- distinct from the
# global RobustScaler's cross-sectional level view and from
# cross_sectional.py's peer-relative view. A plain trailing rolling stat, not
# a fitted transform: no train/test split to manage, and valid unchanged
# under any evaluation methodology (re-cutting the split never requires
# recomputing these columns).
FUND_ZHIST_COLS = [
    "pl", "pvp", "roe", "net_margin", "ebitda_margin", "debt_equity",
    "net_debt_ebitda", "earnings_yield", "book_to_market", "current_ratio",
    "asset_turnover",
]
FUND_ZHIST_WINDOW_QUARTERS = 20   # 5y of quarterly filings
FUND_ZHIST_MIN_QUARTERS = 8       # below this a per-ticker median/IQR is too noisy -> NaN

DAILY_ZHIST_COLS = ["amihud_illiquidity", "turnover_ratio"]
DAILY_ZHIST_WINDOW_DAYS = 1260    # 5y of trading days
DAILY_ZHIST_MIN_DAYS = 252        # 1y warm-up


def _rolling_robust_zscore(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """(x - rolling_median) / rolling_IQR over a trailing window ending at
    each row (inclusive) -- causal by construction, can never see future
    rows. A perfectly constant window (IQR == 0) yields NaN, not +-inf.
    """
    roll = s.rolling(window=window, min_periods=min_periods)
    median = roll.median()
    iqr = roll.quantile(0.75) - roll.quantile(0.25)
    z = (s - median) / iqr
    return z.where(iqr != 0)


def compute_history_relative_features(df):
    """Per-ticker own-history z-scores (R1, docs/PER_TICKER_SCALING_PLAN.md).

    Fundamental columns are quarterly step functions forward-filled across
    ~63 daily rows between filings -- rolling directly on the daily panel
    would be ~63x redundant and give degenerate windows for short histories.
    Dedup to one row per (ticker, reference_date), roll there, then map each
    quarter's own z-score back onto every daily row of that quarter -- same
    pattern as roe_trend_4q/n_quarters_available above.

    Daily (liquidity) columns roll directly over daily rows -- no such
    redundancy.

    Must run after compute_advanced_features: that's what computes
    amihud_illiquidity/turnover_ratio/earnings_yield, and re-anchors
    pl/pvp/book_to_market to their final, correct values.
    """

    print()
    print("=" * 80)
    print("COMPUTING HISTORY-RELATIVE (PER-TICKER) FEATURES")
    print("=" * 80)

    fund_cols = [c for c in FUND_ZHIST_COLS if c in df.columns]
    daily_cols = [c for c in DAILY_ZHIST_COLS if c in df.columns]

    result = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("trade_date").copy()

        for col in daily_cols:
            g[f"{col}_zhist_5y"] = _rolling_robust_zscore(
                g[col], DAILY_ZHIST_WINDOW_DAYS, DAILY_ZHIST_MIN_DAYS
            )

        q = g.drop_duplicates("reference_date").set_index("reference_date").sort_index()
        for col in fund_cols:
            q_zhist = _rolling_robust_zscore(q[col], FUND_ZHIST_WINDOW_QUARTERS, FUND_ZHIST_MIN_QUARTERS)
            g[f"{col}_zhist_5y"] = g["reference_date"].map(q_zhist)

        result.append(g)

    df = pd.concat(result, ignore_index=True)
    print(f"History-relative features added for {df['ticker'].nunique()} tickers")
    return df
