"""
build_ml_dataset.py
===================

Constrói um dataset final para Machine Learning unindo:

1. Prices (daily)
2. Fundamentals (quarterly)
3. Company info (static)

Resultado:
    Uma linha por:
        (ticker, trade_date)

Com:
    - preços diários
    - fundamentos mais recentes disponíveis
    - informações da empresa

Saída:
    data/processed/ml_dataset.parquet

Uso:
    python build_ml_dataset.py
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from .cagr_handler import fill_cagr_columns


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]
PRICES_DIR = ROOT / "data/raw/prices"
FUNDAMENTALS_DIR = ROOT / "data/raw/fundamentals"
COMPANY_INFO_PATH = ROOT / "data/raw/company_info/company_info.parquet"
MACRO_DIR = ROOT / "data/raw/macro"
DIVIDENDS_DIR = ROOT / "data/raw/dividends"
CORPORATE_EVENTS_PATH = ROOT / "data/raw/corporate_events/corporate_events.parquet"
OUTPUT_PATH = ROOT / "data/processed/ml_dataset.parquet"

# Tickers with fewer price rows than this carry no usable history (e.g. EGGY3 has 1 row)
MIN_PRICE_ROWS = 10

# Tickers whose raw price feed is broken beyond programmatic repair.
# Quarantined deliberately — document the reason, don't silently drop.
QUARANTINED_TICKERS = {
    "WDCN3": "raw close alternates between two price bases (~6x apart) "
             "hundreds of times 2021-2025; not a split, no factor to repair with",
}

# Columns the fundamentals API doesn't actually populate
FUNDAMENTALS_NULL_COLS = [
    "sector",
    "subsector",
    "segment",
    "listing_segment",
    "stock_type",
]


# =============================================================================
# LOAD ALL PRICE FILES
# =============================================================================

def load_prices():

    dfs = []
    files = sorted(PRICES_DIR.glob("*.parquet"))

    print()
    print("=" * 80)
    print("LOADING PRICES")
    print("=" * 80)

    for file in files:
        print(f"Loading: {file.name}")
        df = pd.read_parquet(file)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        dfs.append(df)

    prices = pd.concat(dfs, ignore_index=True)
    prices = prices.sort_values(["ticker", "trade_date"])

    print(f"Total price rows: {len(prices)}")

    return prices


# =============================================================================
# REPAIR UNADJUSTED SPLITS
# =============================================================================

ADJ_PRICE_COLS = ["adj_open", "adj_high", "adj_low", "adj_close"]

# An event is only detectable when its raw jump ln(1/factor) stands out from
# normal market moves (0.3 ≈ ±35%); the observed return must match it within
# JUMP_MATCH_TOL. The window is wide because corporate_events dates are
# month-granular (most are recorded as the 1st of the month).
MIN_DETECTABLE_JUMP = 0.3
JUMP_MATCH_TOL = 0.15
EVENT_WINDOW_DAYS = (-10, 35)


def repair_unadjusted_splits(prices):
    """Rescale adj_* history where the source left a split/inplit unadjusted.

    corporate_events.parquet is the audit log of all splits. Most are already
    baked into adj_close upstream, but ~45 events are not: the raw jump
    ln(1/factor) shows up verbatim in the daily return (a fake ±90-99.99%
    move that poisons returns, volatility, drawdown and any reward built on
    them). Detect that jump near each recorded event date and divide all
    adj_* history before it by the factor, making the series continuous.

    ponytail: events with |ln(1/factor)| < 0.3 can't be told apart from
    market moves and are left alone; volume is not rescaled (only raw volume
    reaches the dataset, no cross-scale volume features exist yet).
    """
    if not CORPORATE_EVENTS_PATH.exists():
        print("corporate_events.parquet missing — skipping split repair")
        return prices

    ev = pd.read_parquet(CORPORATE_EVENTS_PATH)
    ev = ev[ev["factor"] > 0].copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev = ev[np.abs(np.log(1.0 / ev["factor"])) >= MIN_DETECTABLE_JUMP]

    print()
    print("=" * 80)
    print("REPAIRING UNADJUSTED SPLITS IN adj_* PRICES")
    print("=" * 80)

    n_fixed = 0
    for ticker, g_ev in ev.groupby("ticker"):
        mask = prices["ticker"] == ticker
        if not mask.any():
            continue
        g_idx = prices.index[mask]  # trade_date-sorted (load_prices sorts)
        adj = prices.loc[g_idx, "adj_close"].to_numpy(dtype=float)
        dates = prices.loc[g_idx, "trade_date"].to_numpy()

        # The audit log's factor direction is inconsistent (SBSP3 records 0.2
        # where the observed basis change is x5, ETER3 records 100 for /100),
        # and one event can manifest as several re-anchoring steps days apart
        # (TIMS3's /10000 arrives as two /100 jumps). So: match the jump in
        # BOTH directions, always repair the EARLIEST unrepaired jump first,
        # and rescan until the ticker's windows are clean.
        applied = set()
        for _ in range(2 * len(g_ev) + 2):  # bound: each pass fixes a new day
            with np.errstate(divide="ignore", invalid="ignore"):
                lr = np.log(adj[1:] / adj[:-1])
            best = None  # (jump_row, factor)
            for _, e in g_ev.iterrows():
                lo = np.datetime64(e["date"] + pd.Timedelta(days=EVENT_WINDOW_DAYS[0]))
                hi = np.datetime64(e["date"] + pd.Timedelta(days=EVENT_WINDOW_DAYS[1]))
                win = (dates[1:] >= lo) & (dates[1:] <= hi)
                for factor in (e["factor"], 1.0 / e["factor"]):
                    expected = np.log(1.0 / factor)
                    cand = np.where(win & (np.abs(lr - expected) < JUMP_MATCH_TOL))[0]
                    for c in cand:
                        jump = c + 1  # first row already on the post-event scale
                        if dates[jump] in applied:
                            continue
                        if best is None or jump < best[0]:
                            best = (jump, factor)
                        break
            if best is None:
                break  # all windows clean — the normal case is zero passes
            jump, factor = best
            applied.add(dates[jump])
            prices.loc[g_idx[:jump], ADJ_PRICE_COLS] /= factor
            adj[:jump] /= factor
            n_fixed += 1
            print(f"  {ticker} {pd.Timestamp(dates[jump]).date()}: rescaled "
                  f"{jump} rows before factor-{factor:g} basis change")

    print(f"Repaired {n_fixed} unadjusted events")
    return prices


def filter_tickers_with_no_fundamentals(prices, fundamentals):
    """Drop any ticker from prices that has zero fundamental rows.

    Sparse fundamentals (e.g. PETR4 only goes back to 2010) are fine —
    the model handles NaNs in early rows. Zero fundamentals means we have
    no quality signal at all, which is not acceptable for this agent.
    """

    print()
    print("=" * 80)
    print("FUNDAMENTAL COVERAGE CHECK")
    print("=" * 80)

    quarantined = set(QUARANTINED_TICKERS) & set(prices["ticker"].unique())
    for t in sorted(quarantined):
        print(f"QUARANTINED {t}: {QUARANTINED_TICKERS[t]}")
    prices = prices[~prices["ticker"].isin(quarantined)]

    tickers_with_prices = set(prices["ticker"].unique())
    tickers_with_fundamentals = set(fundamentals["ticker"].unique())

    missing = tickers_with_prices - tickers_with_fundamentals
    covered = tickers_with_prices & tickers_with_fundamentals

    if missing:
        print(f"EXCLUDED (no fundamentals): {sorted(missing)}")
        print("  These tickers have price data but zero fundamental coverage.")
        print("  A conservative long-term agent requires fundamental quality signals.")
        prices = prices[prices["ticker"].isin(covered)]

    # Drop tickers with almost no price history — nothing to learn from them
    row_counts = prices.groupby("ticker").size()
    too_short = set(row_counts[row_counts < MIN_PRICE_ROWS].index)
    if too_short:
        print(f"EXCLUDED (< {MIN_PRICE_ROWS} price rows): {sorted(too_short)}")
        prices = prices[~prices["ticker"].isin(too_short)]
        covered -= too_short

    print(f"Tickers retained: {sorted(covered)}")
    print(f"Price rows after filter: {len(prices)}")

    return prices


# =============================================================================
# LOAD ALL FUNDAMENTALS
# =============================================================================

def load_fundamentals():

    dfs = []
    files = sorted(FUNDAMENTALS_DIR.glob("*.parquet"))

    print()
    print("=" * 80)
    print("LOADING FUNDAMENTALS")
    print("=" * 80)

    for file in files:
        print(f"Loading: {file.name}")
        df = pd.read_parquet(file)
        df["reference_date"] = pd.to_datetime(df["reference_date"])
        dfs.append(df)

    fundamentals = pd.concat(dfs, ignore_index=True)

    # Drop columns that are always null (API doesn't return them)
    cols_to_drop = [
        c for c in FUNDAMENTALS_NULL_COLS
        if c in fundamentals.columns
    ]
    if cols_to_drop:
        fundamentals = fundamentals.drop(columns=cols_to_drop)
        print(f"Dropped always-null columns: {cols_to_drop}")

    # Drop redundant corporate_name — company_info has it with more detail
    if "corporate_name" in fundamentals.columns:
        fundamentals = fundamentals.drop(columns=["corporate_name"])
        print("Dropped redundant 'corporate_name' from fundamentals")

    fundamentals = fundamentals.sort_values(["ticker", "reference_date"])

    print(f"Total fundamentals rows: {len(fundamentals)}")

    return fundamentals


# =============================================================================
# FILL MISSING CAGR VALUES
# =============================================================================

def fill_missing_cagr(fundamentals):

    print()
    print("=" * 80)
    print("FILLING MISSING CAGR VALUES")
    print("=" * 80)

    # Group by ticker and apply CAGR filling
    dfs = []
    for ticker in sorted(fundamentals["ticker"].unique()):
        ticker_df = fundamentals[fundamentals["ticker"] == ticker].copy()
        
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
# LOAD COMPANY INFO
# =============================================================================

def load_company_info():

    print()
    print("=" * 80)
    print("LOADING COMPANY INFO")
    print("=" * 80)

    df = pd.read_parquet(COMPANY_INFO_PATH)

    print(f"Company rows: {len(df)}")

    return df


# =============================================================================
# LOAD DIVIDENDS
# =============================================================================

def load_dividends():

    dfs = []
    files = sorted(DIVIDENDS_DIR.glob("*.parquet"))

    print()
    print("=" * 80)
    print("LOADING DIVIDENDS")
    print("=" * 80)

    for file in files:
        print(f"Loading: {file.name}")
        df = pd.read_parquet(file)
        df["ex_date"] = pd.to_datetime(df["ex_date"])
        dfs.append(df)

    dividends = pd.concat(dfs, ignore_index=True)
    dividends = dividends.sort_values(["ticker", "ex_date"])

    print(f"Total dividend rows: {len(dividends)}")

    return dividends


# =============================================================================
# MERGE DAILY PRICES + QUARTERLY FUNDAMENTALS
# =============================================================================

# `reference_date` from BolsAI is the fiscal quarter-end, not the real filing/
# disclosure date (verified: BolsAI's /fundamentals history has no filing-date
# field at all, so it can't be recovered from the API). CVM requires quarterly
# ITR filings within 45 days of quarter-end and annual DFP within ~90 days —
# merging fundamentals in on reference_date directly makes them "available"
# weeks to months before a real trader could have seen them. These are
# statutory-deadline estimates, not per-company actual filing dates.
# ponytail: conservative fixed buffer, not per-company actual filing dates —
# upgrade if BolsAI ever exposes a real disclosure date.
FILING_LAG_DAYS_QUARTERLY = 45
FILING_LAG_DAYS_ANNUAL = 90


def merge_prices_and_fundamentals(prices, fundamentals):

    print()
    print("=" * 80)
    print("MERGING PRICES + FUNDAMENTALS")
    print("=" * 80)

    merged_dfs = []

    for ticker in sorted(prices["ticker"].unique()):

        print(f"Merging {ticker}")

        p = (
            prices[prices["ticker"] == ticker]
            .copy()
            .sort_values("trade_date")
        )

        f = (
            fundamentals[fundamentals["ticker"] == ticker]
            .copy()
            .sort_values("reference_date")
        )

        lag_days = np.where(
            f["reference_date"].dt.month == 12,
            FILING_LAG_DAYS_ANNUAL,
            FILING_LAG_DAYS_QUARTERLY,
        )
        f["fundamentals_available_date"] = f["reference_date"] + pd.to_timedelta(lag_days, unit="D")
        f = f.sort_values("fundamentals_available_date")

        # merge_asof: uses the most recent fundamental whose estimated filing
        # date has already passed as of each trade_date (no lookahead bias).
        # fundamentals_available_date is kept (not dropped) — recompute_valuation_daily's
        # split-guard needs it to know which rows just picked up a new fundamental.
        merged = pd.merge_asof(
            p,
            f,
            left_on="trade_date",
            right_on="fundamentals_available_date",
            by="ticker",
            direction="backward",
        )

        merged_dfs.append(merged)

    final_df = pd.concat(merged_dfs, ignore_index=True)

    print(f"Merged rows: {len(final_df)}")

    return final_df


# =============================================================================
# ADD STATIC COMPANY INFO
# =============================================================================

def merge_company_info(df, company_info):

    print()
    print("=" * 80)
    print("ADDING COMPANY INFO")
    print("=" * 80)

    # ticker_primary duplicates ticker — drop before merging
    company_info = company_info.drop(
        columns=[c for c in ["ticker_primary"] if c in company_info.columns]
    )

    merged = df.merge(
        company_info,
        on="ticker",
        how="left",
    )

    return merged


# =============================================================================
# ADD MACRO SERIES (SELIC, CDI, IPCA)
# =============================================================================

def merge_macro(dataset):

    print()
    print("=" * 80)
    print("ADDING MACRO SERIES")
    print("=" * 80)

    # Macro is ticker-independent: one value per calendar date applies to all
    # tickers, so no `by=` and no per-ticker loop. Rename each macro date key
    # to avoid colliding with the fundamentals `reference_date` already present.
    for name in ("selic", "cdi", "ipca"):
        print(f"Merging {name}")
        m = pd.read_parquet(MACRO_DIR / f"{name}.parquet")[["reference_date", name]]
        m = m.rename(columns={"reference_date": f"{name}_date"}).sort_values(f"{name}_date")
        dataset = pd.merge_asof(
            dataset.sort_values("trade_date"),
            m,
            left_on="trade_date",
            right_on=f"{name}_date",
            direction="backward",   # latest macro value <= trade_date (no lookahead)
        ).drop(columns=f"{name}_date")

    return dataset.sort_values(["ticker", "trade_date"]).reset_index(drop=True)


# =============================================================================
# MERGE DIVIDENDS AND COMPUTE DIVIDEND FEATURES
# =============================================================================

def merge_dividends(dataset, dividends):

    print()
    print("=" * 80)
    print("MERGING DIVIDENDS")
    print("=" * 80)

    merged_dfs = []
    count = 0

    # ponytail: use groupby instead of repeated filtering for performance
    for ticker, d in dataset.groupby("ticker", sort=False):

        if count % 50 == 0:
            print(f"Processing dividends for ticker #{count}")
        count += 1

        div = (
            dividends[dividends["ticker"] == ticker]
            .copy()
            .sort_values("ex_date")
        )

        if len(div) == 0:
            # No dividends — set div_value_recent (used downstream); yield/count are
            # (re)computed for all tickers in compute_dividend_features.
            d = d.copy()
            d["div_value_recent"] = 0.0
            merged_dfs.append(d)
            continue

        # Merge most recent dividend (ex_date <= trade_date) for each price row
        merged = pd.merge_asof(
            d.sort_values("trade_date"),
            div[["ex_date", "value_per_share"]].rename(
                columns={"ex_date": "div_ex_date", "value_per_share": "div_value_recent"}
            ),
            left_on="trade_date",
            right_on="div_ex_date",
            direction="backward",
        ).drop(columns="div_ex_date")

        merged_dfs.append(merged)

    result = pd.concat(merged_dfs, ignore_index=True)
    print(f"Merged {len(dividends)} dividends into {len(result)} rows")

    return result


def compute_dividend_features(dataset, dividends):
    """Compute rolling dividend yield and frequency after dividends are loaded."""

    print()
    print("=" * 80)
    print("COMPUTING DIVIDEND FEATURES")
    print("=" * 80)

    window = np.timedelta64(252, "D")

    result = []
    for ticker, g in dataset.groupby("ticker", sort=False):
        g = g.sort_values("trade_date").copy()

        div = dividends[dividends["ticker"] == ticker].sort_values("ex_date")

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
# FEATURE ENGINEERING
# =============================================================================

def _rsi(series, n=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


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
        g["log_return"]     = np.log(adj / adj.shift(1))
        g["volatility_20d"] = g["log_return"].rolling(20).std()
        g["volatility_60d"] = g["log_return"].rolling(60).std()
        g["ma_20"]          = g["adj_close"].rolling(20).mean()
        g["ma_60"]          = g["adj_close"].rolling(60).mean()
        # adj_high/adj_low, not raw high/low: raw and adjusted prices live on
        # different scales whenever the cumulative split adjustment != 1, and
        # mixing them made hl_ratio meaningless around splits.
        g["hl_ratio"]       = (g["adj_high"] - g["adj_low"]) / g["adj_close"]
        g["drawdown"]       = (g["adj_close"] - g["adj_close"].cummax()) / g["adj_close"].cummax()
        g["rsi_14"]         = _rsi(g["adj_close"], 14)
        g["return_1m"]      = g["log_return"].rolling(21).sum()
        g["return_3m"]      = g["log_return"].rolling(63).sum()
        g["return_6m"]      = g["log_return"].rolling(126).sum()
        g["return_12m"]     = g["log_return"].rolling(252).sum()
        result.append(g)

    print(f"Price features added for {len(result)} tickers")
    return pd.concat(result, ignore_index=True)


def compute_fundamental_features(df):
    """Called on the fundamentals DataFrame BEFORE the asof merge."""

    print()
    print("=" * 80)
    print("COMPUTING FUNDAMENTAL FEATURES")
    print("=" * 80)

    result = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("reference_date")

        # Value signals (inverse/normalized forms not pre-computed by API)
        g["book_to_market"]        = g["equity"] / g["market_cap"]
        g["earnings_yield"]        = g["net_income"] / g["market_cap"]
        g["cash_ratio"]            = g["cash"] / g["current_liabilities"]
        g["net_debt_to_assets"]    = g["net_debt"] / g["total_assets"]
        g["working_capital_ratio"] = (g["current_assets"] - g["current_liabilities"]) / g["total_assets"]

        # YoY growth (4 quarters back)
        g["revenue_growth_yoy"]       = g["net_revenue"].pct_change(4)
        g["earnings_growth_yoy"]      = g["net_income"].pct_change(4)
        g["ebitda_growth_yoy"]        = g["ebitda"].pct_change(4)
        g["total_assets_growth_yoy"]  = g["total_assets"].pct_change(4)
        g["total_debt_growth_yoy"]    = g["total_debt"].pct_change(4)

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

    # Split guard: right after a fundamental first becomes available the factor
    # should be ~1; a big jump means the share count changed between close_price
    # and today's close (split/grouping) and rescaled ratios are off until the
    # next filing. Uses fundamentals_available_date (reference_date + statutory
    # filing lag), not reference_date directly, since that's when a row could
    # first have picked up this fundamental via merge_asof.
    # ponytail: warning only; a per-ticker split-factor correction if it fires often
    near_filing = (df["trade_date"] - df["fundamentals_available_date"]).dt.days.between(0, 7)
    suspicious = near_filing & ((factor > 1.5) | (factor < 1 / 1.5))
    if suspicious.any():
        bad = sorted(df.loc[suspicious, "ticker"].unique())
        print(f"WARNING: close/close_price jump >50% at filing date for "
              f"{len(bad)} tickers (possible split): {bad[:20]}")

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

    # close_price (price at filing date) is now redundant and misleading
    df = df.drop(columns=["close_price", "fundamentals_available_date"])

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
    df["dividend_coverage_ratio"] = (
        df["ebitda"] /
        (df["div_value_recent"] * df["shares_outstanding"] + 1e-8)
    )

    # --- EARNINGS QUALITY (raw signals, no classification) ---

    # Revenue-to-earnings trend: stable ratio suggests quality
    df["revenue_per_earning"] = df["net_revenue"] / (df["net_income"] + 1e-8)

    # YoY comparison: revenue growth aligned with earnings growth?
    df["revenue_vs_earnings_growth_delta"] = (
        df["revenue_growth_yoy"] - df["earnings_growth_yoy"]
    )

    # EBITDA margin as quality proxy (higher = better operational efficiency, but let model learn)
    df["ebitda_margin"] = df["ebitda"] / (df["net_revenue"] + 1e-8)

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

    # --- SECTOR-RELATIVE METRICS (Z-scores, percentiles) ---

    # ponytail: vectorized z-score via cython groupby transforms (no Python per-group calls)
    # NaN-sector rows are dropped by groupby and stay NaN, matching the old loop's skip.
    sector_grp = df.groupby(["trade_date", "sector"], sort=False)
    for col in ["pl", "pvp", "roe", "debt_equity"]:
        if col in df.columns:
            mean = sector_grp[col].transform("mean")
            std = sector_grp[col].transform("std")
            # std <= 0 or NaN (single-stock sectors) → NaN, same as the old guard
            df[f"{col}_zscore_sector"] = (df[col] - mean) / std.where(std > 0)

    # Dividend yield percentile: percentile rank within sector per date
    df["div_yield_sector_percentile"] = sector_grp["div_yield_12m"].rank(pct=True)

    # --- MOMENTUM DECOMPOSITION (stock vs sector vs market) ---

    # ponytail: use groupby.transform() for vectorized momentum (1000x faster than loops)
    # Market momentum: subtract market mean (per date) from each return
    df["momentum_vs_market_1m"] = (
        df["return_1m"] - df.groupby("trade_date")["return_1m"].transform("mean")
    )
    df["momentum_vs_market_3m"] = (
        df["return_3m"] - df.groupby("trade_date")["return_3m"].transform("mean")
    )
    df["momentum_vs_market_12m"] = (
        df["return_12m"] - df.groupby("trade_date")["return_12m"].transform("mean")
    )

    # Sector momentum: subtract sector mean (per date, sector) from each return
    df["momentum_vs_sector_1m"] = (
        df["return_1m"]
        - df.groupby(["trade_date", "sector"])["return_1m"].transform("mean")
    )
    df["momentum_vs_sector_3m"] = (
        df["return_3m"]
        - df.groupby(["trade_date", "sector"])["return_3m"].transform("mean")
    )
    df["momentum_vs_sector_12m"] = (
        df["return_12m"]
        - df.groupby(["trade_date", "sector"])["return_12m"].transform("mean")
    )

    # --- FUNDAMENTAL TREND SIGNALS (raw, no thresholds) ---

    # ponytail: use groupby.apply() for vectorized trends (sort by reference_date within group)
    df = df.sort_values(["ticker", "reference_date"]).reset_index(drop=True)

    df["roe_trend_4q"] = df.groupby("ticker", sort=False)["roe"].transform(
        lambda x: x.diff(4)
    )
    df["margin_trend_4q"] = df.groupby("ticker", sort=False)["net_margin"].transform(
        lambda x: x.diff(4)
    )
    df["debt_trend_4q"] = df.groupby("ticker", sort=False)["debt_equity"].transform(
        lambda x: x.diff(4)
    )
    df["roa_trend_4q"] = df.groupby("ticker", sort=False)["roa"].transform(
        lambda x: x.diff(4)
    )

    # --- VALUATION RELATIVE TO FUNDAMENTALS (raw relationships) ---

    # PEG ratio: P/L (P/E) relative to earnings growth
    df["peg_ratio"] = df["pl"] / (df["earnings_growth_yoy"] * 100 + 1e-8)

    # P/VP (P/B) relative to ROE (value signal: low P/VP + high ROE = cheap quality)
    df["pvp_to_roe_ratio"] = df["pvp"] / (df["roe"] + 1e-8)

    # Earnings yield (inverse P/L) vs macro rates
    df["earnings_yield"] = 1.0 / (df["pl"] + 1e-8)
    df["earnings_yield_vs_selic"] = df["earnings_yield"] - (df["selic"] / 100)

    print(f"Advanced features computed for {len(df)} rows")
    return df


# =============================================================================
# CLEAN DATA
# =============================================================================

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


# =============================================================================
# BUILD MANIFEST
# =============================================================================

def write_manifest(dataset):
    """Reproducibility record + per-column distribution snapshot, one per build.

    Written next to the parquet as ml_dataset.manifest.json. Comparing two
    manifests (e.g. before/after a code change) surfaces silent distribution
    drift that passes every schema check.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=ROOT,
        ).stdout.strip() or "unknown"
    except OSError:
        commit = "unknown"

    def _f(x):
        return None if pd.isna(x) else round(float(x), 6)

    numeric = dataset.select_dtypes(include="number")
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "rows": len(dataset),
        "tickers": int(dataset["ticker"].nunique()),
        "date_min": str(dataset["trade_date"].min().date()),
        "date_max": str(dataset["trade_date"].max().date()),
        "columns": list(dataset.columns),
        "column_stats": {
            c: {
                "nan_pct": round(float(numeric[c].isna().mean()) * 100, 2),
                "mean": _f(numeric[c].mean()),
                "std": _f(numeric[c].std()),
                "p1": _f(numeric[c].quantile(0.01)),
                "p50": _f(numeric[c].quantile(0.50)),
                "p99": _f(numeric[c].quantile(0.99)),
            }
            for c in numeric.columns
        },
    }
    path = OUTPUT_PATH.with_suffix(".manifest.json")
    path.write_text(json.dumps(manifest, indent=1))
    print(f"Manifest saved to: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    prices       = load_prices()
    prices       = repair_unadjusted_splits(prices)
    fundamentals = load_fundamentals()
    prices       = filter_tickers_with_no_fundamentals(prices, fundamentals)
    fundamentals = compute_fundamental_features(fundamentals)
    fundamentals = fill_missing_cagr(fundamentals)
    company_info = load_company_info()
    dividends    = load_dividends()

    dataset = merge_prices_and_fundamentals(prices, fundamentals)
    dataset = merge_company_info(dataset, company_info)
    dataset = merge_macro(dataset)
    dataset = merge_dividends(dataset, dividends)
    dataset = compute_price_features(dataset)
    dataset = compute_dividend_features(dataset, dividends)
    dataset = compute_macro_features(dataset)
    dataset = recompute_valuation_daily(dataset)
    dataset = compute_advanced_features(dataset)
    dataset = clean_dataset(dataset)

    print()
    print("=" * 80)
    print("SAVING DATASET")
    print("=" * 80)

    dataset.to_parquet(OUTPUT_PATH, index=False)
    write_manifest(dataset)

    print(f"Saved to: {OUTPUT_PATH}")

    print()
    print("=" * 80)
    print("FINAL DATASET SUMMARY")
    print("=" * 80)
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")
    print()
    print("Columns:")
    for col in dataset.columns:
        print(f"  {col}")
    print()
    print(dataset.head())


if __name__ == "__main__":
    main()