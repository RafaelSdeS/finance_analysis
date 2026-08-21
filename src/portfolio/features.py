"""
features.py -- the §4.4-E "keep" feature partition from
docs/PORTFOLIO_ARCHITECTURE_PROPOSAL.md, as a literal list. Verified
column-by-column against the real data/processed/ml_dataset.parquet schema
and manifest.LOOKAHEAD_TAINTED_COLS: 119 real columns total (118 numeric +
`sector`), none of them identifiers, lookahead-tainted, or non-stationary
raw price/volume/currency levels (§4.4 A/B/C/D).

[2026-08-20] `cagr_revenue_5y`/`cagr_earnings_5y` (bare, pre-fill columns)
dropped: they no longer exist in the built dataset -- only the
BolsAI-or-backfilled `..._final` columns do (same root cause as
scale_features.py's RATIO_COLUMNS fix). Count was 121/120 as of 2026-07-24;
this list quietly stopped matching real data before this fix.

`sector` is kept separate (not in the numeric groups): it's a static
low-cardinality categorical, verified absent from
`manifest.LOOKAHEAD_TAINTED_COLS` (see proposal §4.4.B correction), usable
for grouping/one-hot but not as a plain numeric feature.
"""

from src.build_dataset.manifest import LOOKAHEAD_TAINTED_COLS

VALUATION = [
    "pl", "pvp", "ev_ebitda", "ev_ebit", "p_ebitda", "p_ebit", "p_sr", "p_assets",
    "book_to_market", "earnings_yield", "peg_ratio", "pvp_to_roe_ratio", "earnings_yield_vs_selic",
]
PROFITABILITY = [
    "gross_margin", "net_margin", "ebitda_margin", "ebit_margin", "roe", "roa", "roic",
    "ebit_over_assets", "asset_turnover", "revenue_per_earning",
]
LEVERAGE = [
    "current_ratio", "debt_equity", "net_debt_equity", "net_debt_ebitda", "net_debt_ebit",
    "cash_ratio", "net_debt_to_assets", "working_capital_ratio",
]
GROWTH = [
    "cagr_earnings_5y_final", "cagr_revenue_5y_final",
    "revenue_growth_yoy", "earnings_growth_yoy", "ebitda_growth_yoy", "total_assets_growth_yoy",
    "total_debt_growth_yoy", "revenue_vs_earnings_growth_delta", "gross_margin_yoy_1q",
    "net_margin_yoy_1q", "roe_yoy_1q", "debt_equity_qoq", "current_ratio_qoq",
]
PIOTROSKI = [
    "f_score", "f_roa_positive", "f_roa_improving", "f_margin_improving",
    "f_leverage_decreasing", "f_liquidity_improving", "had_negative_earnings_5y",
]
TRENDS = ["roe_trend_4q", "margin_trend_4q", "debt_trend_4q", "roa_trend_4q"]
PRICE_TECH = [
    "volatility_20d", "volatility_60d", "volatility_ratio_20_60", "price_vs_ma20",
    "price_vs_ma60", "hl_ratio", "true_range_ratio", "drawdown", "rsi_14",
    "volume_ratio_20d", "amihud_illiquidity", "turnover_ratio",
]
MOMENTUM = [
    "log_return", "overnight_gap", "intraday_return", "return_1m", "return_3m", "return_6m",
    "return_12m", "excess_return", "real_return", "momentum_vs_market_1m",
    "momentum_vs_market_3m", "momentum_vs_market_12m", "beta_1y",
]
PERCENTILES = [
    "volatility_20d_percentile", "volatility_60d_percentile", "price_percentile_5y",
    "price_percentile_1y", "pl_percentile_5y", "drawdown_percentile",
]
ZHIST = [
    "amihud_illiquidity_zhist_5y", "turnover_ratio_zhist_5y", "pl_zhist_5y", "pvp_zhist_5y",
    "roe_zhist_5y", "net_margin_zhist_5y", "ebitda_margin_zhist_5y", "debt_equity_zhist_5y",
    "net_debt_ebitda_zhist_5y", "earnings_yield_zhist_5y", "book_to_market_zhist_5y",
    "current_ratio_zhist_5y", "asset_turnover_zhist_5y",
]
DIVIDENDS = [
    "div_yield_12m", "div_count_12m", "div_value_12m", "div_value_recent",
    "payout_ratio", "dividend_coverage_ratio", "has_dividends",
]
# earnings_yield_vs_selic (Valuation) and excess_return/real_return (Momentum) are also
# primary regime signals per proposal §4.1 -- not repeated here, already counted above.
MACRO = ["selic_trend_20d", "selic", "cdi", "ipca", "ipca_daily_equiv"]
INFO_AGE_FLAGS = [
    "filing_lag_days", "days_since_fundamental", "n_quarters_available", "has_fundamentals",
    "cagr_earnings_defined", "cagr_revenue_defined", "adj_close_precision_degraded",
]

_NUMERIC_GROUPS = [
    VALUATION, PROFITABILITY, LEVERAGE, GROWTH, PIOTROSKI, TRENDS,
    PRICE_TECH, MOMENTUM, PERCENTILES, ZHIST, DIVIDENDS, MACRO, INFO_AGE_FLAGS,
]

SECTOR = "sector"


def feature_columns(include_sector: bool = False) -> list[str]:
    """The keep-list: 120 numeric features, +1 (`sector`) if include_sector.
    Stable order across calls -- needed for reproducible model input."""
    cols = [c for group in _NUMERIC_GROUPS for c in group]
    tainted = set(cols) & set(LOOKAHEAD_TAINTED_COLS)
    assert not tainted, f"lookahead-tainted column(s) leaked into the keep-list: {tainted}"
    if include_sector:
        cols = cols + [SECTOR]
    return cols
