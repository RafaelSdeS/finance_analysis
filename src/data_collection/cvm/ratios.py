"""cvm/ratios.py — BolsAI-schema fundamentals, rebuilt from CVM raw statements +
shares outstanding for every crosswalk-resolvable ticker (not just delisted names).
"""

import logging

import pandas as pd

from .. import config, storage, validate
from .crosswalk import CROSSWALK_PATH
from .shares import SHARES_PATH, collect_shares
from .statements import FLOW_COLS, collect_statements, load_statements

log = logging.getLogger("cvm")

_TTM_COLS = FLOW_COLS + ["depr_amort"]  # net_revenue, gross_profit, ebit, net_income, depr_amort
_TAX_RATE = 0.34  # Brazilian statutory corporate rate (IRPJ+CSLL combined) -- an approximation
# of each company's actual effective rate, not a parsed tax-expense line. Documented
# approximation, same spirit as the ebitda==ebit shortcut it replaces for corporates.


def _ttm(q: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Single-quarter flow columns -> trailing-twelve-month, on a gap-safe quarter-end
    grid: reindexed onto every expected quarter between the series' min/max first, so a
    missing filing breaks the rolling window (NaN) instead of silently summing two
    non-adjacent quarters. `q` must be one cnpj, sorted by reference_date."""
    if q.empty:
        return q.assign(**{c: pd.Series(dtype=float) for c in cols})
    full_idx = pd.date_range(q["reference_date"].min(), q["reference_date"].max(), freq="QE")
    grid = q.set_index("reference_date")[cols].reindex(full_idx)
    ttm = grid.rolling(4, min_periods=4).sum()
    out = q.copy()
    for c in cols:
        out[c] = ttm[c].reindex(q["reference_date"]).to_numpy()
    return out


def compute_ratios(q: pd.DataFrame, corporate_name: str) -> pd.DataFrame:
    """Wide quarterly frame (one cnpj) + close_price/shares_outstanding columns ->
    BolsAI-schema fundamentals. Flows are standardized to TTM for every ticker
    (locked design decision, 2026-08-19 -- see BOLSAI_EXIT_PLAN.md Task 1): BolsAI
    itself mixes single-quarter and TTM per ticker (measured: 309 vs 269 of 612,
    near a coin flip), which corrupts any cross-sectional comparison
    (`cross_sectional.py` z-scores within sector groups, `alpha.py` trains one
    model across the whole panel) -- both need a column to mean the same thing for
    every row. Balance-sheet items stay point-in-time, never TTM'd. Values are
    full R$ units throughout (DATA_LAYER_CORRECTNESS_PLAN.md §1) -- CVM's raw
    statements are in thousands, scaled up once below, right after TTM."""
    g = _ttm(q.sort_values("reference_date"), [c for c in _TTM_COLS if c in q.columns])

    # §1: scale the RAW inputs, not the derived output names -- cash/total_debt/
    # net_debt/ebitda are derived FROM these just below, so scaling only stored
    # output columns would miss them (leaving ev_*/p_ebitda/net_debt_* wrong).
    # market_cap (close_price * shares_outstanding) is already full units; leave
    # it, and keep this scaling point above it.
    for c in ("net_income", "equity", "net_revenue", "ebit", "total_assets",
              "current_assets", "current_liabilities", "gross_profit", "depr_amort",
              "cash_caixa", "cash_aplic", "debt_st", "debt_lt"):
        if c in g.columns:
            g[c] = g[c] * 1000.0

    def col(name):
        return g[name] if name in g.columns else pd.Series(float("nan"), index=g.index)

    g["cash"] = col("cash_caixa").fillna(0) + col("cash_aplic").fillna(0)
    g.loc[col("cash_caixa").isna() & col("cash_aplic").isna(), "cash"] = float("nan")
    g["total_debt"] = col("debt_st").fillna(0) + col("debt_lt").fillna(0)
    g.loc[col("debt_st").isna() & col("debt_lt").isna(), "total_debt"] = float("nan")
    g["net_debt"] = g["total_debt"] - g["cash"]
    # Real EBITDA for corporates (ebit + TTM D&A from the DFC cash-flow statement).
    # Banks: ebit is already NaN (statements.py's dre_column() never labels a bank's
    # 3.05 pre-tax line as EBIT), so ebitda stays NaN too -- a real industry
    # difference (D&A isn't a meaningful concept on a bank's DRE), not a shortcut.
    g["ebitda"] = col("ebit") + col("depr_amort")

    g["market_cap"] = g["close_price"] * g["shares_outstanding"]
    k = 1.0  # inputs already scaled to full R$ units above (§1) -- kept as a
    # multiplier (not deleted) so every crossing below stays visually marked
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
    # NOPAT / invested capital, invested capital = total_debt + equity - cash.
    # Uses _TAX_RATE (statutory, not effective) -- see module docstring note.
    invested_capital = g["total_debt"] + col("equity") - g["cash"]
    g["roic"] = (col("ebit") * (1 - _TAX_RATE)) / invested_capital * 100

    # BolsAI has these; fill_missing_cagr() backfills them in Stage 2 from the
    # net_income/net_revenue history, exactly as it does for BolsAI nulls
    g["cagr_revenue_5y"] = float("nan")
    g["cagr_earnings_5y"] = float("nan")
    g["corporate_name"] = corporate_name

    keep = ["reference_date", "close_price", "shares_outstanding", "market_cap",
            "pl", "pvp", "ev_ebitda", "ev_ebit", "p_ebitda", "p_ebit", "p_sr",
            "lpa", "vpa", "gross_margin", "net_margin", "ebitda_margin", "ebit_margin",
            "roe", "roa", "roic", "ebit_over_assets", "asset_turnover", "p_assets",
            "current_ratio", "debt_equity", "net_debt_equity", "net_debt_ebitda",
            "net_debt_ebit", "cagr_revenue_5y", "cagr_earnings_5y",
            "net_income", "equity", "net_revenue", "total_debt", "ebitda", "ebit",
            "net_debt", "cash", "total_assets", "current_assets", "current_liabilities",
            "corporate_name"]
    for c in keep:  # banks lack some accounts (e.g. 3.05) — NaN keeps the schema stable
        if c not in g.columns:
            g[c] = float("nan")
    g = g[keep]
    # nonzero/0 divisions land as inf, not NaN (only 0/0 propagates NaN naturally) — clean
    # at the source so raw parquet never stores literal inf. Same pattern as
    # yf_collectors.compute_ratios(), which writes to this same fundamentals schema.
    num = g.select_dtypes(include="number").columns
    g[num] = g[num].replace([float("inf"), float("-inf")], float("nan"))
    return g


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


def build_fundamentals(tickers: list[str] | None = None, rebuild: bool = False) -> None:
    """Per-ticker fundamentals parquet for every crosswalk ticker with a prices file.

    rebuild=False (default): only tickers with no existing fundamentals file get one
    (the original delisted-only behavior). rebuild=True: recompute and overwrite every
    ticker from CVM regardless of what's on disk -- `q` always carries that ticker's
    FULL statement history, so `storage._merge_save()`'s existing concat+dedup(keep=
    "last") already replaces every quarter with the freshly computed CVM value; no
    separate overwrite path needed in storage.py.
    """
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
        if (out.exists() and not rebuild) or not px_path.exists():
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
        saved = storage._merge_save(df, out, "reference_date",
                                     validate.validate_fundamentals,
                                     f"cvm_fundamentals/{ticker}")
        if saved is not None:
            written += 1
            log.info("fundamentals %s: %d quarters (CVM)", ticker, len(saved))
    log.info("build_fundamentals: %d written, %d skipped (existing/no prices)", written, skipped)


def collect_fundamentals_cvm(tickers: list[str], mode: str) -> None:
    """pipeline.py's ("fundamentals", "cvm") DATA_SOURCE entry -- BUG-1's free, correct
    replacement for the yfinance fundamentals path (see BOLSAI_EXIT_PLAN.md Task 5).
    `mode` unused (fn_map signature parity with every other collect_X(tickers, mode)).

    collect_statements()/collect_shares() only ever re-fetch the CURRENT CVM year (both
    already cache-and-skip every prior year), so refreshing them every quarterly update
    is cheap -- picks up newly-filed quarters before rebuilding ratios from them.
    rebuild=True: `tickers` here is the caller's already-scoped list (e.g. `active`
    ATIVO tickers in pipeline.py), not the full universe, so a full per-ticker recompute
    every run is still fast.
    """
    collect_statements()
    collect_shares()
    build_fundamentals(tickers=tickers, rebuild=True)
