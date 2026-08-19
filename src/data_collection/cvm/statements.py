"""cvm/statements.py — DFP/ITR DRE+BPA+BPP(+DFC) -> one wide quarterly frame per cnpj."""

import logging
import re
from datetime import date

import pandas as pd

from .. import config
from . import http

log = logging.getLogger("cvm")

# net_revenue (3.01) and gross_profit (3.03) are stable codes across BOTH the corporate and
# bank/financial DRE layouts -- "top line" and "gross result" mean the same concept in each,
# just phrased differently ("Receita de Venda de Bens/Serviços" vs "Receitas de Intermediação
# Financeira"). ebit and net_income are NOT stable across layouts -- see dre_column() below.
DRE_ACCOUNTS = {
    "3.01": "net_revenue",
    "3.03": "gross_profit",
}
# "1" (total_assets) and "2" (total_liabilities) are universal top-line codes -- verified
# stable across 2706 filers in the 2025 ITR set, no divergence. Everything nested underneath
# is NOT: a 17-filer bank sub-layout (verified live, e.g. Banco do Brasil, cnpj
# 00000000000191) puts equity at 2.07 ("Patrimônio Líquido Consolidado"), not 2.03 -- 2.03 is
# "Provisões" (provisions) or "Passivos Financeiros ao Custo Amortizado" (financial
# liabilities) for these filers, a completely different concept. Same failure mode as the DRE
# bug: the account CODE alone doesn't disambiguate, the description does. The same 17 filers
# also drop the current/non-current split entirely (1.01/2.01 hold "Caixa e Equivalentes de
# Caixa" / fair-value-liability lines instead of "Ativo/Passivo Circulante") -- for those,
# current_assets/current_liabilities correctly come out NaN rather than a wrong number,
# consistent with "leave NaN rather than guess" (same policy as bank EBIT).
BPA_ACCOUNTS = {"1": "total_assets"}
BPP_ACCOUNTS = {}  # total_liabilities (code "2") isn't in our schema; nothing else is code-stable
_EQUITY_RE = re.compile(r"patrim[oô]nio.*l[ií]quido", re.I)
_CURRENT_RE = re.compile(r"circulante", re.I)
_DEBT_RE = re.compile(r"empr[eé]stimo|financiamento|dep[oó]sito", re.I)
_CASH_RE = re.compile(r"caixa|equivalente|aplica[cç]", re.I)
FLOW_COLS = ["net_revenue", "gross_profit", "ebit", "net_income"]  # balances are point-in-time
# depr_amort is handled separately in load_statements() -- unlike the DRE flows above, ITR's
# DFC_MI has no single-quarter alternative row (always Jan-1-to-date cumulative), so it needs
# diffing within-year rather than the DFP-annual-minus-ITR-sum treatment FLOW_COLS gets.

# The DRE has TWO incompatible chart layouts -- corporate and bank/financial -- and the
# account CODE alone cannot tell them apart. Verified live against ITR 2025 DRE_con
# (2026-08-19), counting descriptions per code across the whole market:
#
#   code   corporate (n=4322)                       bank (n=154)
#   3.05   Res. Antes do Res. Financeiro e dos       Resultado antes dos Tributos sobre
#          Tributos              (= true EBIT)       o Lucro           (= PRE-TAX, not EBIT)
#   3.09   Res. Líquido das Oper. Continuadas        Lucro/Prejuízo Consolidado do Período
#   3.11   Lucro/Prejuízo Consolidado do Período     Lucro ou Prejuízo Líquido Consolidado
#   3.13   —                                         Lucro/Prejuízo Consolidado do Período
#
# Net income lives at 3.09, 3.11 OR 3.13 depending on layout, and 3.09/3.11 mean DIFFERENT
# things in each -- so the code can't disambiguate; the description can. Likewise 3.05 is a
# real EBIT only for corporates (banks have no equivalent line -- left NaN, not mislabeled).
_NET_INCOME_RE = re.compile(r"lucro.*preju[ií]zo.*per[ií]odo", re.I)
_EBIT_RE = re.compile(r"antes do resultado financeiro", re.I)
# D&A reconciling line inside the indirect-method cash flow statement's operating section
# (DFC_MI, code 6.01.01.*) -- verified on PETR4: 6.01.01.04 "Depreciação, depleção e
# amortização" = 62,317,000 (thousands), sane vs its scale. Code is NOT stable across filers
# (2025 ITR filing set: 6.01.01.02 most common at 1338 filers, down through .01/.03.../.17+),
# so this matches by description too. Direct-method (DFC_MD) filers have no such reconciling
# line at all (~2% of the market, confirmed live) -- depr_amort stays NaN for them, same
# "leave NaN rather than guess" policy as bank EBIT.
_DA_RE = re.compile(r"deprecia[cç][aã]o", re.I)


def dre_column(code: str, desc: str) -> str | None:
    """One DRE row -> our column name, or None to skip. Top-level codes only (`3.NN`)."""
    if code.count(".") != 1 or not code.startswith("3."):
        return None
    if _NET_INCOME_RE.search(desc):        # 3.09 / 3.11 / 3.13 depending on layout
        return "net_income"
    if code in DRE_ACCOUNTS:
        return DRE_ACCOUNTS[code]
    if code == "3.05" and _EBIT_RE.search(desc):  # corporate only; banks -> NaN
        return "ebit"
    return None


def dfc_column(code: str, desc: str) -> str | None:
    """One DFC_MI row -> our column name, or None to skip."""
    if code.startswith("6.01.01.") and _DA_RE.search(desc):
        return "depr_amort"
    return None


def bpa_column(code: str, desc: str) -> str | None:
    """One BPA (assets) row -> our column name, or None to skip."""
    if code in BPA_ACCOUNTS:
        return BPA_ACCOUNTS[code]
    if code == "1.01" and _CURRENT_RE.search(desc):
        return "current_assets"
    if code == "1.01.01" and _CASH_RE.search(desc):
        return "cash_caixa"
    if code == "1.01.02" and _CASH_RE.search(desc):
        return "cash_aplic"
    return None


def bpp_column(code: str, desc: str) -> str | None:
    """One BPP (liabilities+equity) row -> our column name, or None to skip."""
    if code.count(".") == 1 and code.startswith("2.") and _EQUITY_RE.search(desc):
        return "equity"  # 2.03 usually, 2.07/2.08 for the alt bank sub-layout -- see note above
    if code == "2.01" and _CURRENT_RE.search(desc):
        return "current_liabilities"
    if code == "2.01.04" and _DEBT_RE.search(desc):
        return "debt_st"
    if code == "2.02.01" and _DEBT_RE.search(desc):
        return "debt_lt"
    return None


def _parse_statement_year(doc: str, year: int) -> pd.DataFrame | None:
    """One year's DRE+BPA+BPP+DFC_MI into long rows: cnpj, reference_date, col, value, code.
    Prefers consolidated (_con); falls back to individual (_ind) per cnpj+date.

    DFC_MI's depr_amort is left as the RAW CUMULATIVE (Jan-1-to-date) value here -- CVM's cash
    flow statement has no single-quarter alternative row the way DRE does, so de-accumulation
    needs the full year's quarters together and happens in load_statements() alongside the
    existing DFP-Q4-from-ITR logic, not here.
    """
    zf = http.fetch_zip(doc, year)
    if zf is None:
        return None

    frames = []
    for stmt in ("DRE", "BPA", "BPP", "DFC_MI"):
        for scope in ("con", "ind"):
            recs = http.read_csv(zf, f"{doc.lower()}_cia_aberta_{stmt}_{scope}_{year}.csv")
            if not recs:
                continue
            rows = []
            for r in recs:
                code = r.get("CD_CONTA", "")
                desc = r.get("DS_CONTA", "") or ""
                if stmt == "DRE":
                    col = dre_column(code, desc)
                elif stmt == "DFC_MI":
                    col = dfc_column(code, desc)
                elif stmt == "BPA":
                    col = bpa_column(code, desc)
                else:
                    col = bpp_column(code, desc)
                if col is None or r.get("ORDEM_EXERC") != "ÚLTIMO":
                    continue
                # ITR DRE carries both quarter and YTD rows; keep the ~3-month ones.
                # DFC_MI has no such alternative (always Jan-1-to-date) -- handled by the
                # diff-based de-accumulation in load_statements() instead.
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
                    "col": col,
                    "value": value,
                    "code": code,
                    "scope": scope,
                })
            if rows:
                frames.append(pd.DataFrame(rows))

    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    # consolidated wins over individual for the same cnpj+date+col; within a scope, the
    # HIGHEST account code wins -- a filer carrying more than one line matching
    # _NET_INCOME_RE (e.g. 3.11 continuing-ops plus 3.13 final) resolves to the last line
    # of the waterfall, i.e. the actual bottom line.
    df["scope"] = pd.Categorical(df["scope"], categories=["con", "ind"], ordered=True)
    df = (df.sort_values(["scope", "code"], ascending=[True, False])
            .drop_duplicates(["cnpj", "reference_date", "col"], keep="first")
            .drop(columns=["scope", "code"]))
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

    # depr_amort (DFC_MI) is cumulative Jan-1-to-date, unlike the FLOW_COLS above which are
    # already single-quarter in ITR -- needs diffing within each (cnpj, year), not the
    # annual-minus-ITR-sum treatment. Reindexed onto a full Q1-Q4 grid first so a missing
    # quarter produces NaN instead of silently diffing across the gap.
    if "depr_amort" in itr.columns or "depr_amort" in dfp.columns:
        da = pd.concat([
            itr[["cnpj", "reference_date", "depr_amort"]] if "depr_amort" in itr.columns
            else itr[["cnpj", "reference_date"]].assign(depr_amort=float("nan")),
            dfp[["cnpj", "reference_date", "depr_amort"]] if "depr_amort" in dfp.columns
            else dfp[["cnpj", "reference_date"]].assign(depr_amort=float("nan")),
        ], ignore_index=True).dropna(subset=["depr_amort"])
        da["year"] = da["reference_date"].dt.year
        da["q"] = da["reference_date"].dt.quarter
        da = da.drop_duplicates(["cnpj", "year", "q"], keep="last")
        pairs = da[["cnpj", "year"]].drop_duplicates()
        full_idx = pd.MultiIndex.from_frame(
            pairs.merge(pd.DataFrame({"q": [1, 2, 3, 4]}), how="cross"))
        cum = (da.set_index(["cnpj", "year", "q"])["depr_amort"]
                 .reindex(full_idx).sort_index())
        single = cum.groupby(level=["cnpj", "year"]).diff()
        single.loc[single.index.get_level_values("q") == 1] = \
            cum.loc[cum.index.get_level_values("q") == 1]  # Q1 cumulative IS Q1 single
        single = single.reset_index().rename(columns={"depr_amort": "depr_amort_single"})
        single["reference_date"] = pd.PeriodIndex(
            single["year"].astype(str) + "Q" + single["q"].astype(str), freq="Q"
        ).to_timestamp("Q")
        merge_key = single.set_index(["cnpj", "reference_date"])["depr_amort_single"]
        for frame in (itr, dfp):
            if "depr_amort" not in frame.columns:
                frame["depr_amort"] = float("nan")
            frame["depr_amort"] = frame.set_index(["cnpj", "reference_date"]).index.map(merge_key).to_numpy()

    out = (pd.concat([itr, dfp], ignore_index=True)
             .drop(columns=["report_type", "year"])
             .sort_values(["cnpj", "reference_date"])
             # a quarter present in both ITR and DFP (rare restatement overlap): keep ITR
             .drop_duplicates(["cnpj", "reference_date"], keep="first")
             .reset_index(drop=True))
    return out
