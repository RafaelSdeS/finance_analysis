"""
cvm_statements.py — delisted-company fundamentals from CVM open data.

BolsAI serves full price history for delisted tickers but 404s on their
fundamentals (only companies still in its live registry resolve — verified
2026-07-11, see DELISTED_UNIVERSE.md). CVM's open-data portal has every
filer's raw statements back to 2010, delisted included. This module rebuilds
fundamentals in the same per-ticker parquet schema collect_fundamentals()
writes, so Stage 2 (load_fundamentals glob-and-concat) needs zero changes.

Steps (all CVM sources free & keyless; caches under data/raw/cvm/):
  crosswalk     FCA valor_mobiliario: ticker -> cnpj (verified 3/3 on
                SMLS3/LAME4/HGTX3), cvm_code joined from filing_dates.parquet
  statements    DFP/ITR DRE+BPA+BPP -> one wide quarterly frame per cnpj
  shares        FRE capital_social  -> shares-outstanding timeline per cnpj
  fundamentals  BolsAI-named ratio columns; per-ticker parquet written ONLY
                where no BolsAI file exists (BolsAI stays source of truth
                for active tickers)
  company_info  CANCELADA registry rows (sector, cvm_code — ticker-less on
                BolsAI) joined to tickers via the crosswalk, appended to
                company_info.parquet with status=CANCELADA

Usage (from project root):
    python -m src.data_collection.cvm_statements                  # all steps
    python -m src.data_collection.cvm_statements --step crosswalk
    python -m src.data_collection.cvm_statements --step fundamentals --tickers SMLS3
"""

import argparse
import csv
import io
import logging
import re
import zipfile
from datetime import date

import pandas as pd
import requests

from . import client, collectors, config, validate

log = logging.getLogger("cvm_statements")

CVM_DOC = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/{doc}/DADOS/{doc_l}_cia_aberta_{year}.zip"
START_YEAR = 2010  # CVM open-data coverage floor (same as filing_dates.py).
                   # Note: FCA Codigo_Negociacao is empty in 2010 — "FCA 2010:
                   # 0 ticker rows" is the data, not a bug; coverage starts 2011+.
TIMEOUT = (15, 120)  # (connect, read) — fail fast on a stalled CVM connection
RETRIES = 2

CROSSWALK_PATH = config.CVM_DIR / "fca_crosswalk.parquet"
SHARES_PATH = config.CVM_DIR / "shares.parquet"
FILING_DATES_PATH = config.RAW_DIR / "filing_dates/filing_dates.parquet"

_TICKER = re.compile(r"^[A-Z0-9]{4}(?:[3-8]|11)$")

# CVM standard chart of accounts -> our raw-value columns (values in R$ thousands,
# same unit BolsAI uses). ponytail: non-bank chart only; bank DREs use a different
# 3.x layout, their flow columns come out NaN — same gap BolsAI itself has for banks.
DRE_ACCOUNTS = {
    "3.01": "net_revenue",
    "3.03": "gross_profit",
    "3.05": "ebit",
    "3.11": "net_income",
}
BPA_ACCOUNTS = {
    "1": "total_assets",
    "1.01": "current_assets",
    "1.01.01": "cash_caixa",
    "1.01.02": "cash_aplic",
}
BPP_ACCOUNTS = {
    "2.01": "current_liabilities",
    "2.01.04": "debt_st",
    "2.02.01": "debt_lt",
    "2.03": "equity",
}
FLOW_COLS = list(DRE_ACCOUNTS.values())  # per-quarter; balances are point-in-time


# ---------------------------------------------------------------------------
# download helpers
# ---------------------------------------------------------------------------

def _fetch_zip(doc: str, year: int) -> zipfile.ZipFile | None:
    """One CVM yearly zip; None when the year isn't published (404)."""
    url = CVM_DOC.format(doc=doc.upper(), doc_l=doc.lower(), year=year)
    log.info("%s %d: downloading...", doc, year)
    for attempt in range(RETRIES + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            break
        except requests.RequestException as e:
            if attempt == RETRIES:
                log.warning("%s %d: network error after %d attempts: %s",
                            doc, year, RETRIES + 1, e)
                return None
            log.warning("%s %d: %s — retrying (%d/%d)", doc, year,
                        type(e).__name__, attempt + 1, RETRIES)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    try:
        return zipfile.ZipFile(io.BytesIO(resp.content))
    except zipfile.BadZipFile as e:
        log.warning("%s %d: corrupt zip: %s", doc, year, e)
        return None


def _read_csv(zf: zipfile.ZipFile, name: str) -> list[dict]:
    try:
        raw = zf.read(name).decode("latin-1")
    except KeyError:
        return []
    return list(csv.DictReader(io.StringIO(raw), delimiter=";"))


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


# ---------------------------------------------------------------------------
# step: crosswalk (FCA valor_mobiliario -> ticker/cnpj/cvm_code)
# ---------------------------------------------------------------------------

_FCA_COLS = ["ticker", "cnpj", "corporate_name", "end_trading", "year"]


def build_crosswalk() -> pd.DataFrame:
    """ticker -> cnpj, cvm_code, corporate_name, end_trading. Latest FCA wins per ticker.
    Per-year FCA rows cached to data/raw/cvm/fca_{year}.parquet; only the current
    year is re-downloaded on rerun (new filings arrive all year)."""
    config.CVM_DIR.mkdir(parents=True, exist_ok=True)
    current = date.today().year
    frames = []
    for year in range(START_YEAR, current + 1):
        cache = config.CVM_DIR / f"fca_{year}.parquet"
        if cache.exists() and year < current:
            frames.append(pd.read_parquet(cache))
            continue
        zf = _fetch_zip("FCA", year)
        if zf is None:
            continue
        rows = []
        for r in _read_csv(zf, f"fca_cia_aberta_valor_mobiliario_{year}.csv"):
            ticker = (r.get("Codigo_Negociacao") or "").strip().upper()
            if not _TICKER.match(ticker):
                continue
            rows.append({
                "ticker": ticker,
                "cnpj": _digits(r.get("CNPJ_Companhia")),
                "corporate_name": (r.get("Nome_Empresarial") or "").strip(),
                "end_trading": (r.get("Data_Fim_Negociacao") or "").strip() or None,
                "year": year,
            })
        df_y = pd.DataFrame(rows, columns=_FCA_COLS)  # empty years cached too (2010 has no codes)
        df_y.to_parquet(cache, index=False)
        frames.append(df_y)
        log.info("FCA %d: %d ticker rows", year, len(df_y))

    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_FCA_COLS)
    if all_rows.empty:
        raise RuntimeError("no FCA data downloaded — CVM portal unreachable?")
    df = (all_rows
          .sort_values("year")
          .drop_duplicates("ticker", keep="last")
          .drop(columns="year"))

    # cvm_code via filing_dates (already on disk, cnpj+cvm_code per filing)
    if FILING_DATES_PATH.exists():
        fd = pd.read_parquet(FILING_DATES_PATH)[["cnpj", "cvm_code"]].drop_duplicates("cnpj")
        df = df.merge(fd, on="cnpj", how="left")
    else:
        df["cvm_code"] = None
        log.warning("filing_dates.parquet missing — cvm_code left null "
                    "(run: python -m src.data_collection.filing_dates)")

    df["end_trading"] = pd.to_datetime(df["end_trading"], errors="coerce")
    df.to_parquet(CROSSWALK_PATH, index=False)
    log.info("crosswalk: %d tickers (%d with cvm_code) -> %s",
             len(df), df["cvm_code"].notna().sum(), CROSSWALK_PATH)
    return df


# ---------------------------------------------------------------------------
# step: statements (DFP/ITR DRE+BPA+BPP -> wide quarterly frame)
# ---------------------------------------------------------------------------

def _parse_statement_year(doc: str, year: int) -> pd.DataFrame | None:
    """One year's DRE+BPA+BPP into long rows: cnpj, reference_date, col, value.
    Prefers consolidated (_con); falls back to individual (_ind) per cnpj+date."""
    zf = _fetch_zip(doc, year)
    if zf is None:
        return None

    frames = []
    for stmt, accounts in (("DRE", DRE_ACCOUNTS), ("BPA", BPA_ACCOUNTS), ("BPP", BPP_ACCOUNTS)):
        for scope in ("con", "ind"):
            recs = _read_csv(zf, f"{doc.lower()}_cia_aberta_{stmt}_{scope}_{year}.csv")
            if not recs:
                continue
            rows = []
            for r in recs:
                code = r.get("CD_CONTA", "")
                if code not in accounts or r.get("ORDEM_EXERC") != "ÚLTIMO":
                    continue
                # ITR DRE carries both quarter and YTD rows; keep the ~3-month ones
                if stmt == "DRE" and doc.upper() == "ITR":
                    ini, fim = r.get("DT_INI_EXERC", ""), r.get("DT_FIM_EXERC", "")
                    if ini and fim:
                        span = (pd.Timestamp(fim) - pd.Timestamp(ini)).days
                        if span > 95:
                            continue
                try:
                    value = float(r.get("VL_CONTA", "") or 0)
                except ValueError:
                    continue
                if r.get("ESCALA_MOEDA") == "UNIDADE":
                    value /= 1000.0  # normalize to thousands (BolsAI unit)
                rows.append({
                    "cnpj": _digits(r.get("CNPJ_CIA")),
                    "reference_date": r.get("DT_REFER"),
                    "col": accounts[code],
                    "value": value,
                    "scope": scope,
                })
            if rows:
                frames.append(pd.DataFrame(rows))

    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    # consolidated wins over individual for the same cnpj+date+col
    df["scope"] = pd.Categorical(df["scope"], categories=["con", "ind"], ordered=True)
    df = (df.sort_values("scope")
            .drop_duplicates(["cnpj", "reference_date", "col"], keep="first")
            .drop(columns="scope"))
    df["report_type"] = doc.upper()
    df["reference_date"] = pd.to_datetime(df["reference_date"], errors="coerce")
    return df.dropna(subset=["reference_date"])


def collect_statements() -> None:
    """Download+cache one parquet per (doc, year); skip existing except current year."""
    config.CVM_DIR.mkdir(parents=True, exist_ok=True)
    current = date.today().year
    for doc in ("ITR", "DFP"):
        for year in range(START_YEAR, current + 1):
            out = config.CVM_DIR / f"stmt_{doc.lower()}_{year}.parquet"
            if out.exists() and year < current:
                continue
            df = _parse_statement_year(doc, year)
            if df is None or df.empty:
                log.info("%s %d: nothing published", doc, year)
                continue
            df.to_parquet(out, index=False)
            log.info("%s %d: %d rows -> %s", doc, year, len(df), out.name)


def load_statements() -> pd.DataFrame:
    """All cached statement years -> wide frame: one row per cnpj+reference_date.

    DFP DRE flows are full-year; Q4 flow = annual − (Q1+Q2+Q3 from ITR), NaN when
    any interim quarter is missing. Balance items are point-in-time everywhere.
    """
    files = sorted(config.CVM_DIR.glob("stmt_*.parquet"))
    if not files:
        raise RuntimeError("no statement caches — run collect_statements() first")
    long = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    wide = (long.pivot_table(index=["cnpj", "reference_date", "report_type"],
                             columns="col", values="value", aggfunc="first")
                .reset_index())

    itr = wide[wide["report_type"] == "ITR"].copy()
    dfp = wide[wide["report_type"] == "DFP"].copy()

    # DFP row becomes the Q4 row: balances as reported; flows = annual − sum(ITR flows)
    dfp["year"] = dfp["reference_date"].dt.year
    itr["year"] = itr["reference_date"].dt.year
    itr_sums = itr.groupby(["cnpj", "year"])[
        [c for c in FLOW_COLS if c in itr.columns]
    ].agg(["sum", "count"])
    for col in FLOW_COLS:
        if col not in dfp.columns or (col, "sum") not in itr_sums.columns:
            continue
        key = pd.MultiIndex.from_arrays([dfp["cnpj"], dfp["year"]])
        sums = itr_sums[(col, "sum")].reindex(key).to_numpy()
        counts = itr_sums[(col, "count")].reindex(key).to_numpy()
        q4 = dfp[col].to_numpy() - sums
        dfp[col] = pd.Series(q4, index=dfp.index).where(counts == 3)  # need all 3 interim quarters

    out = (pd.concat([itr, dfp], ignore_index=True)
             .drop(columns=["report_type", "year"])
             .sort_values(["cnpj", "reference_date"])
             # a quarter present in both ITR and DFP (rare restatement overlap): keep ITR
             .drop_duplicates(["cnpj", "reference_date"], keep="first")
             .reset_index(drop=True))
    return out


# ---------------------------------------------------------------------------
# step: shares outstanding (FRE capital_social)
# ---------------------------------------------------------------------------

_FRE_COLS = ["cnpj", "effective_date", "shares"]


def collect_shares() -> pd.DataFrame:
    """Per-cnpj timeline of total shares: (cnpj, effective_date, shares).
    Per-year FRE rows cached to data/raw/cvm/fre_{year}.parquet (zips are ~10 MB
    each); only the current year is re-downloaded on rerun."""
    config.CVM_DIR.mkdir(parents=True, exist_ok=True)
    current = date.today().year
    frames = []
    for year in range(START_YEAR, current + 1):
        cache = config.CVM_DIR / f"fre_{year}.parquet"
        if cache.exists() and year < current:
            frames.append(pd.read_parquet(cache))
            continue
        zf = _fetch_zip("FRE", year)
        if zf is None:
            continue
        rows = []
        for r in _read_csv(zf, f"fre_cia_aberta_capital_social_{year}.csv"):
            if r.get("Tipo_Capital") != "Capital Integralizado":
                continue
            try:
                shares = int(r.get("Quantidade_Total_Acoes", "") or 0)
            except ValueError:
                continue
            if shares <= 0:
                continue
            rows.append({
                "cnpj": _digits(r.get("CNPJ_Companhia")),
                "effective_date": r.get("Data_Autorizacao_Aprovacao") or r.get("Data_Referencia"),
                "shares": shares,
            })
        df_y = pd.DataFrame(rows, columns=_FRE_COLS)
        df_y.to_parquet(cache, index=False)
        frames.append(df_y)
        log.info("FRE %d: %d rows", year, len(df_y))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_FRE_COLS)
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    df = (df.dropna(subset=["effective_date"])
            .sort_values("effective_date")
            .drop_duplicates(["cnpj", "effective_date"], keep="last")
            .reset_index(drop=True))
    df.to_parquet(SHARES_PATH, index=False)
    log.info("shares: %d rows, %d companies -> %s", len(df), df["cnpj"].nunique(), SHARES_PATH)
    return df


# ---------------------------------------------------------------------------
# step: fundamentals (ratios in BolsAI column names, per-ticker parquet)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# step: company_info for delisted (CANCELADA registry x crosswalk)
# ---------------------------------------------------------------------------

def synthesize_company_info() -> None:
    """Append COMPANY_FIELDS rows for delisted tickers to company_info.parquet.
    Sector/cvm_code come from BolsAI's CANCELADA registry (ticker-less there);
    the ticker link comes from the FCA crosswalk, joined on cnpj."""
    xwalk = pd.read_parquet(CROSSWALK_PATH)

    c = client.make_client(config.BOLSAI_BASE, config.BOLSAI_API_KEY)
    try:
        regs, offset = [], 0
        while True:
            d = client.get_json(c, "/companies/",
                                {"status": "CANCELADA", "limit": 500, "offset": offset})
            batch = d.get("data", [])
            if not batch:
                break
            regs += batch
            offset += len(batch)
            if len(batch) < 500:
                break
    finally:
        c.close()

    reg = pd.DataFrame(regs)
    reg["cnpj"] = reg["cnpj"].map(_digits)
    reg = reg.drop_duplicates("cnpj")

    merged = xwalk.merge(reg[["cnpj", "cvm_code", "sector", "corporate_name",
                              "trade_name", "status"]],
                         on="cnpj", how="inner", suffixes=("_fca", ""))
    rows = pd.DataFrame({
        "ticker": merged["ticker"],
        "ticker_primary": merged["ticker"],
        "corporate_name": merged["corporate_name"],
        "trade_name": merged["trade_name"],
        "cvm_code": merged["cvm_code"],
        "cnpj": merged["cnpj"],
        "sector": merged["sector"],
        "status": merged["status"],
    })

    path = config.COMPANY_DIR / "company_info.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=rows.columns)
    out = (pd.concat([existing, rows], ignore_index=True)
             .drop_duplicates("ticker", keep="first"))  # existing (BolsAI ATIVO) wins
    out.to_parquet(path, index=False)
    log.info("company_info: +%d delisted rows -> %d total", len(out) - len(existing), len(out))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Delisted fundamentals from CVM open data")
    p.add_argument("--step", choices=["crosswalk", "statements", "shares",
                                      "fundamentals", "company_info", "all"],
                   default="all")
    p.add_argument("--tickers", nargs="+", help="restrict build_fundamentals")
    args = p.parse_args()

    steps = {
        "crosswalk": build_crosswalk,
        "statements": collect_statements,
        "shares": collect_shares,
        "fundamentals": lambda: build_fundamentals(
            [t.upper() for t in args.tickers] if args.tickers else None),
        "company_info": synthesize_company_info,
    }
    order = list(steps) if args.step == "all" else [args.step]
    for name in order:
        log.info("=== step: %s ===", name)
        steps[name]()


if __name__ == "__main__":
    main()
