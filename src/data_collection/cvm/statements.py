"""cvm/statements.py — DFP/ITR DRE+BPA+BPP -> one wide quarterly frame per cnpj."""

import logging
from datetime import date

import pandas as pd

from .. import config
from . import http

log = logging.getLogger("cvm")

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


def _parse_statement_year(doc: str, year: int) -> pd.DataFrame | None:
    """One year's DRE+BPA+BPP into long rows: cnpj, reference_date, col, value.
    Prefers consolidated (_con); falls back to individual (_ind) per cnpj+date."""
    zf = http.fetch_zip(doc, year)
    if zf is None:
        return None

    frames = []
    for stmt, accounts in (("DRE", DRE_ACCOUNTS), ("BPA", BPA_ACCOUNTS), ("BPP", BPP_ACCOUNTS)):
        for scope in ("con", "ind"):
            recs = http.read_csv(zf, f"{doc.lower()}_cia_aberta_{stmt}_{scope}_{year}.csv")
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
                    "cnpj": http.digits(r.get("CNPJ_CIA")),
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
        for year in range(http.START_YEAR, current + 1):
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
