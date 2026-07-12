"""cvm/ratios.py — BolsAI-schema fundamentals (ratios) for delisted tickers,
built from CVM raw statements + shares outstanding. Written only where no
BolsAI-sourced fundamentals file already exists.
"""

import logging

import pandas as pd

from .. import collectors, config, validate
from .crosswalk import CROSSWALK_PATH
from .shares import SHARES_PATH
from .statements import load_statements

log = logging.getLogger("cvm")


def compute_ratios(q: pd.DataFrame, corporate_name: str) -> pd.DataFrame:
    """Wide quarterly frame (one cnpj) + close_price/shares_outstanding columns
    -> BolsAI-schema fundamentals. Ratio conventions verified against a live
    BolsAI response (BPAN4 2025-09-30): flows are single-quarter (not TTM),
    flow/balance values in R$ thousands, market_cap in R$ units."""
    g = q.copy()

    def col(name):
        return g[name] if name in g.columns else pd.Series(float("nan"), index=g.index)

    g["cash"] = col("cash_caixa").fillna(0) + col("cash_aplic").fillna(0)
    g.loc[col("cash_caixa").isna() & col("cash_aplic").isna(), "cash"] = float("nan")
    g["total_debt"] = col("debt_st").fillna(0) + col("debt_lt").fillna(0)
    g.loc[col("debt_st").isna() & col("debt_lt").isna(), "total_debt"] = float("nan")
    g["net_debt"] = g["total_debt"] - g["cash"]
    # ponytail: EBITDA = EBIT (D&A needs DFC parsing; BolsAI itself ships
    # ebitda==ebit for banks — add DFC_MI parsing if EBITDA precision matters)
    g["ebitda"] = col("ebit")

    g["market_cap"] = g["close_price"] * g["shares_outstanding"]
    k = 1000.0  # statements are in thousands; market_cap in units
    g["pl"] = g["market_cap"] / (col("net_income") * k)
    g["pvp"] = g["market_cap"] / (col("equity") * k)
    g["p_sr"] = g["market_cap"] / (col("net_revenue") * k)
    g["p_ebit"] = g["market_cap"] / (col("ebit") * k)
    g["p_ebitda"] = g["market_cap"] / (g["ebitda"] * k)
    g["p_assets"] = g["market_cap"] / (col("total_assets") * k)
    ev = g["market_cap"] + g["net_debt"] * k
    g["ev_ebit"] = ev / (col("ebit") * k)
    g["ev_ebitda"] = ev / (g["ebitda"] * k)
    g["lpa"] = col("net_income") * k / g["shares_outstanding"]
    g["vpa"] = col("equity") * k / g["shares_outstanding"]

    g["roe"] = col("net_income") / col("equity") * 100
    g["roa"] = col("net_income") / col("total_assets") * 100
    g["gross_margin"] = col("gross_profit") / col("net_revenue") * 100
    g["net_margin"] = col("net_income") / col("net_revenue") * 100
    g["ebit_margin"] = col("ebit") / col("net_revenue") * 100
    g["ebitda_margin"] = g["ebitda"] / col("net_revenue") * 100
    g["ebit_over_assets"] = col("ebit") / col("total_assets") * 100
    g["asset_turnover"] = col("net_revenue") / col("total_assets")
    g["current_ratio"] = col("current_assets") / col("current_liabilities")
    g["debt_equity"] = g["total_debt"] / col("equity")
    g["net_debt_equity"] = g["net_debt"] / col("equity")
    g["net_debt_ebitda"] = g["net_debt"] / g["ebitda"]
    g["net_debt_ebit"] = g["net_debt"] / col("ebit")

    # BolsAI has these; fill_missing_cagr() backfills them in Stage 2 from the
    # net_income/net_revenue history, exactly as it does for BolsAI nulls
    g["cagr_revenue_5y"] = float("nan")
    g["cagr_earnings_5y"] = float("nan")
    g["corporate_name"] = corporate_name

    keep = ["reference_date", "close_price", "shares_outstanding", "market_cap",
            "pl", "pvp", "ev_ebitda", "ev_ebit", "p_ebitda", "p_ebit", "p_sr",
            "lpa", "vpa", "gross_margin", "net_margin", "ebitda_margin", "ebit_margin",
            "roe", "roa", "ebit_over_assets", "asset_turnover", "p_assets",
            "current_ratio", "debt_equity", "net_debt_equity", "net_debt_ebitda",
            "net_debt_ebit", "cagr_revenue_5y", "cagr_earnings_5y",
            "net_income", "equity", "net_revenue", "total_debt", "ebitda", "ebit",
            "net_debt", "cash", "total_assets", "current_assets", "current_liabilities",
            "corporate_name"]
    for c in keep:  # banks lack some accounts (e.g. 3.05) — NaN keeps the schema stable
        if c not in g.columns:
            g[c] = float("nan")
    return g[keep]


def _price_asof(prices: pd.DataFrame, ref_dates: pd.Series) -> pd.Series:
    """Last close at or before each reference date (NaN when none)."""
    px = prices[["trade_date", "close"]].sort_values("trade_date")
    merged = pd.merge_asof(
        pd.DataFrame({"reference_date": ref_dates}).sort_values("reference_date"),
        px, left_on="reference_date", right_on="trade_date", direction="backward")
    return merged.set_index("reference_date")["close"].reindex(ref_dates).to_numpy()


def _shares_asof(shares: pd.DataFrame, cnpj: str, ref_dates: pd.Series):
    tl = shares[shares["cnpj"] == cnpj].sort_values("effective_date")
    if tl.empty:
        return float("nan")
    merged = pd.merge_asof(
        pd.DataFrame({"reference_date": ref_dates}).sort_values("reference_date"),
        tl[["effective_date", "shares"]],
        left_on="reference_date", right_on="effective_date", direction="backward")
    return merged.set_index("reference_date")["shares"].reindex(ref_dates).to_numpy()


def build_fundamentals(tickers: list[str] | None = None) -> None:
    """Per-ticker fundamentals parquet for every crosswalk ticker that has a
    prices file but NO fundamentals file (BolsAI-sourced files are never touched)."""
    xwalk = pd.read_parquet(CROSSWALK_PATH)
    stmts = load_statements()
    shares = pd.read_parquet(SHARES_PATH) if SHARES_PATH.exists() else pd.DataFrame(
        columns=["cnpj", "effective_date", "shares"])

    todo = xwalk if tickers is None else xwalk[xwalk["ticker"].isin(tickers)]
    written = skipped = 0
    for _, row in todo.iterrows():
        ticker, cnpj = row["ticker"], row["cnpj"]
        out = config.FUND_DIR / f"{ticker}.parquet"
        px_path = config.PRICES_DIR / f"{ticker}.parquet"
        if out.exists() or not px_path.exists():
            skipped += 1
            continue
        q = stmts[stmts["cnpj"] == cnpj].sort_values("reference_date")
        if q.empty:
            continue
        prices = pd.read_parquet(px_path)
        q = q.copy()
        q["close_price"] = _price_asof(prices, q["reference_date"])
        q["shares_outstanding"] = _shares_asof(shares, cnpj, q["reference_date"])

        df = compute_ratios(q, row["corporate_name"])
        df["ticker"] = ticker
        saved = collectors._merge_save(df, out, "reference_date",
                                       validate.validate_fundamentals,
                                       f"cvm_fundamentals/{ticker}")
        if saved is not None:
            written += 1
            log.info("fundamentals %s: %d quarters (CVM)", ticker, len(saved))
    log.info("build_fundamentals: %d written, %d skipped (existing/no prices)", written, skipped)
