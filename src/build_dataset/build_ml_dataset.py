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

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# cagr_handler.py lives in src/ — make it importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cagr_handler import fill_cagr_columns


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]
PRICES_DIR = ROOT / "data/raw/prices"
FUNDAMENTALS_DIR = ROOT / "data/raw/fundamentals"
COMPANY_INFO_PATH = ROOT / "data/raw/company_info/company_info.parquet"
MACRO_DIR = ROOT / "data/raw/macro"
DIVIDENDS_DIR = ROOT / "data/raw/dividends"
OUTPUT_PATH = ROOT / "data/processed/ml_dataset.parquet"

# Tickers with fewer price rows than this carry no usable history (e.g. EGGY3 has 1 row)
MIN_PRICE_ROWS = 10

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

        # merge_asof: uses the most recent fundamental
        # available up to each trade_date (no lookahead bias)
        merged = pd.merge_asof(
            p,
            f,
            left_on="trade_date",
            right_on="reference_date",
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

    # Filter out delisted/inactive companies (status != ATIVO)
    # Can't trade suspended or cancelled stocks; exclude from training
    before = len(merged)
    merged = merged[merged["status"].fillna("").eq("ATIVO")]
    filtered_out = before - len(merged)
    if filtered_out > 0:
        print(f"Filtered out {filtered_out} rows from inactive/delisted companies")

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
        g["hl_ratio"]       = (g["high"] - g["low"]) / g["adj_close"]
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

        # Volatility percentile: where is current vol vs this stock's history?
        g["volatility_20d_percentile"] = g["volatility_20d"].rank(pct=True)
        g["volatility_60d_percentile"] = g["volatility_60d"].rank(pct=True)

        # Price percentile: is price high/low vs own history (last 5 years)?
        # rolling.rank(method="max", pct=True) == share of window values <= current,
        # same as the old rolling.apply lambda but computed in cython (~1000x faster).
        # ponytail: NaNs are excluded from the window count here (old lambda counted them)
        window_252 = 252 * 5  # 5 years
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

    df = df.sort_values(["ticker", "trade_date"]).reset_index(drop=True)

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    prices       = load_prices()
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
    dataset = compute_advanced_features(dataset)
    dataset = clean_dataset(dataset)

    print()
    print("=" * 80)
    print("SAVING DATASET")
    print("=" * 80)

    dataset.to_parquet(OUTPUT_PATH, index=False)

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