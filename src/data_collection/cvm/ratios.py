"""cvm/ratios.py — BolsAI-schema fundamentals, rebuilt from CVM raw statements +
shares outstanding for every crosswalk-resolvable ticker (not just delisted names).
"""

import json
import logging

import numpy as np
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

    # Single-quarter (non-TTM) companions for net_revenue/net_income -- deliberately not
    # a full mirror of every TTM column (see docs re: minimal scope), just the pair that
    # lets a downstream margin/ROE catch a loss-to-profit inflection TTM smooths over.
    # q is the pre-TTM frame _ttm() copied from (same row order), so its own values are
    # the true single-quarter figures.
    q_sorted = q.sort_values("reference_date")
    for c in ("net_revenue", "net_income"):
        g[f"{c}_q"] = q_sorted[c].to_numpy() if c in q_sorted.columns else float("nan")

    # §1: scale the RAW inputs, not the derived output names -- cash/total_debt/
    # net_debt/ebitda are derived FROM these just below, so scaling only stored
    # output columns would miss them (leaving ev_*/p_ebitda/net_debt_* wrong).
    # market_cap (close_price * shares_outstanding) is already full units; leave
    # it, and keep this scaling point above it.
    for c in ("net_income", "equity", "net_revenue", "ebit", "total_assets",
              "current_assets", "current_liabilities", "gross_profit", "depr_amort",
              "cash_caixa", "cash_aplic", "debt_st", "debt_lt",
              "net_revenue_q", "net_income_q"):
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
    g["net_margin_q"] = col("net_income_q") / col("net_revenue_q") * 100
    g["roe_q"] = col("net_income_q") / col("equity") * 100
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
            "net_revenue_q", "net_income_q", "net_margin_q", "roe_q",
            "corporate_name"]
    for c in keep:  # banks lack some accounts (e.g. 3.05) — NaN keeps the schema stable
        if c not in g.columns:
            g[c] = float("nan")
    g = g[keep]
    # nonzero/0 divisions land as inf, not NaN (only 0/0 propagates NaN naturally) — clean
    # at the source so raw parquet never stores literal inf. Same pattern as
    # ratios.compute_ratios(), which writes to this same fundamentals schema.
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
    """Returns (shares, effective_date, prev_shares) per ref_date -- the
    effective_date is needed by `_apply_share_events` to know how stale each
    matched FRE record already was at ref_date; prev_shares (the SAME cnpj's
    immediately prior FRE snapshot) lets it tell whether a recorded
    corporate_events split is already reflected in that transition, instead of
    still needing to be forward-adjusted."""
    tl = shares[shares["cnpj"] == cnpj].sort_values("effective_date")
    if tl.empty:
        nan_shares = pd.Series(float("nan"), index=ref_dates.index)
        nat_dates = pd.Series(pd.NaT, index=ref_dates.index)
        return nan_shares.to_numpy(), nat_dates.to_numpy(), nan_shares.to_numpy()
    tl = tl.assign(prev_shares=tl["shares"].shift(1))
    merged = pd.merge_asof(
        pd.DataFrame({"reference_date": ref_dates}).sort_values("reference_date"),
        tl[["effective_date", "shares", "prev_shares"]],
        left_on="reference_date", right_on="effective_date", direction="backward")
    merged = merged.set_index("reference_date").reindex(ref_dates)
    return (merged["shares"].to_numpy(), merged["effective_date"].to_numpy(),
            merged["prev_shares"].to_numpy())


# Some real corporate actions are recorded twice within a few days at an
# inverse-but-numerically-identical ratio (e.g. TIMS3's 2007 reverse split is
# stored both as "1000:1" and "1:0.001", same net 0.001 multiplier) --
# collapsing near-duplicates avoids applying one real event twice.
_EVENT_DEDUP_WINDOW_DAYS = 10
_EVENT_DEDUP_TOL = 0.02

_CONTINUITY_PATH = config.BR_RAW_DIR / "reference" / "ticker_continuity.json"


def _ticker_family(ticker: str) -> set[str]:
    """Every ticker code connected to `ticker` through ticker_continuity.json's
    rename/merger chain (both directions, any number of hops). Needed because
    a corporate_events row is recorded under whichever code was trading when
    the event happened, which may not be the code a given raw fundamentals
    file is built under: TIMS3's real 2025 split is recorded under "TIMS3",
    but TIMP3's own raw file (the pre-splice donor for TIMS3's early history,
    per ticker_continuity.json's TIMP3->TIMS3 rename) needs to see it too --
    they're one continuous real entity, not two independent histories."""
    if not _CONTINUITY_PATH.exists():
        return {ticker}
    events = json.loads(_CONTINUITY_PATH.read_text()).get("events", [])
    parent = {}

    def find(t):
        parent.setdefault(t, t)
        root = t
        while parent[root] != root:
            root = parent[root]
        while parent[t] != root:  # path compression
            parent[t], t = root, parent[t]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in events:
        if e.get("type") in ("tender", "keep_separate"):
            continue  # no splice -- these stay genuinely independent histories
        old, new = e.get("old"), e.get("new")
        if old and new:
            union(old, new)

    if ticker not in parent:
        return {ticker}
    root = find(ticker)
    return {t for t in parent if find(t) == root}


def _share_events(ticker: str) -> pd.DataFrame:
    """`ticker`'s corporate_events.parquet rows (plus any recorded under a
    continuity-chain relative, see `_ticker_family`), deduplicated to one row
    per real event, reduced to (date, share_multiplier). A forward split
    multiplies shares outstanding by ratio_to/ratio_from; a reverse split
    (INPLIT) divides it -- the same field expresses both directions."""
    if not config.CORP_EVENTS_PATH.exists():
        return pd.DataFrame(columns=["date", "share_multiplier"])
    ev = pd.read_parquet(config.CORP_EVENTS_PATH)
    ev = ev[(ev["ticker"].isin(_ticker_family(ticker)))
            & (ev["ratio_from"] > 0) & (ev["ratio_to"] > 0)].copy()
    if ev.empty:
        return pd.DataFrame(columns=["date", "share_multiplier"])
    ev["date"] = pd.to_datetime(ev["date"])
    ev["share_multiplier"] = ev["ratio_to"] / ev["ratio_from"]
    ev = ev.sort_values("date")

    kept = []
    for _, row in ev.iterrows():
        if kept and (row["date"] - kept[-1]["date"]).days <= _EVENT_DEDUP_WINDOW_DAYS \
                and abs(row["share_multiplier"] - kept[-1]["share_multiplier"]) < _EVENT_DEDUP_TOL:
            continue  # same real event, already counted
        kept.append(row)
    return pd.DataFrame(kept)[["date", "share_multiplier"]].reset_index(drop=True)


def _apply_share_events(shares_vals, effective_dates, ref_dates: pd.Series, ticker: str,
                         prev_shares_vals=None):
    """Forward-adjust a FRE-sourced share count by any real split/inplit that
    happened AFTER the matched FRE record's own effective_date and at-or-before
    ref_date -- the gap `_shares_asof`'s backward-merge alone leaves stale
    (docs/DATA_LAYER_FOLLOWUP_FINDINGS.md: TIMS3's shares_outstanding was
    frozen across its real 2025 100:1 reverse split because FRE never filed
    a post-split capital_social update for its CNPJ). Only closes THAT gap --
    a CNPJ with no FRE row at all still comes out NaN/stale; this is a partial
    mitigation, not a substitute for ticker_continuity.json-level verification.

    `prev_shares_vals` (the matched FRE record's own prior snapshot, from
    `_shares_asof`): guards against double-counting an event that FRE's
    transition ALREADY reflects, distinct from TIMS3's "FRE never updated at
    all" shape. Confirmed on RVEE3: its FRE capital_social filing itself jumps
    10x the day BEFORE the recorded 1:10 split (both describe the SAME real
    action) -- naively reapplying the recorded factor on top of that already-
    adjusted FRE value inflated shares_outstanding 100x instead of 10x
    (docs/DATA_LAYER_FOLLOWUP_FINDINGS.md, 2026-08-22). Defaults to None (all
    prior snapshots unknown) so the pre-existing unit test's 4-arg call keeps
    behaving identically -- no exclusion ever triggers without this data.
    """
    events = _share_events(ticker)
    if events.empty:
        return shares_vals
    out = np.array(shares_vals, dtype=float)
    # list(...) first: forces a plain 0..n-1 positional index on both Series,
    # regardless of whatever index `ref_dates` came in with, so `.iloc[i]`
    # lines up with `out[i]` (itself already 0-indexed by `_shares_asof`'s
    # own positional `.to_numpy()`).
    ref_dates = pd.to_datetime(pd.Series(list(ref_dates)))
    eff_dates = pd.Series(list(effective_dates))
    prev_shares = (pd.Series(float("nan"), index=range(len(out))) if prev_shares_vals is None
                   else pd.Series(list(prev_shares_vals)))
    for i in range(len(out)):
        if pd.isna(out[i]) or pd.isna(eff_dates.iloc[i]):
            continue
        applicable = events[(events["date"] > eff_dates.iloc[i]) & (events["date"] <= ref_dates.iloc[i])]
        if applicable.empty:
            continue
        prev = prev_shares.iloc[i]
        if pd.notna(prev) and prev != 0:
            observed_ratio = out[i] / prev
            applicable = applicable[
                ~np.isclose(applicable["share_multiplier"], observed_ratio, rtol=_EVENT_DEDUP_TOL)
            ]
        if not applicable.empty:
            out[i] *= applicable["share_multiplier"].prod()
    return out


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
        shares_vals, shares_eff_dates, shares_prev_vals = _shares_asof(shares, cnpj, q["reference_date"])
        q["shares_outstanding"] = _apply_share_events(
            shares_vals, shares_eff_dates, q["reference_date"], ticker, shares_prev_vals)

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
