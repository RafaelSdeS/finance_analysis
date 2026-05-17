"""
bolsai_daily_sample.py
======================
Gera um CSV com granularidade DIÁRIA:
  - Uma linha por (ticker, trade_date)
  - Colunas de preço: open, high, low, close, adj_close, volume, etc. (diários)
  - Colunas de fundamentos: pl, pvp, roe, ebitda, etc. (trimestrais → forward-filled)
  - Colunas de demonstrações: DRE, BPA, BPP (trimestrais → forward-filled)
  - Colunas macro: selic, ipca, cdi (diárias ou forward-filled)
  - Features de engenharia: retornos diários, log-return, variações

Lógica de join:
  Para cada pregão, os fundamentos usados são os do trimestre mais recente
  já divulgado até aquela data (as-of join / forward-fill).
  Isso evita look-ahead bias: nunca usa dados futuros.

Uso:
  python bolsai_daily_sample.py --api-key sk_SUA_CHAVE
"""

import httpx
import pandas as pd
import numpy as np
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
BASE_URL   = "https://api.usebolsai.com/api/v1"
MAX_HIST   = 80
REPORT_TYPES    = ["DFP", "ITR"]
STATEMENT_TYPES = ["BPA", "BPP", "DRE", "DFC_MI", "DVA"]

SAMPLE_TICKERS = [
    "WEGE3",   # industrial madura, ~60 anos
    "PETR4",   # estatal/commodity, altíssimo volume
    "CASH3",   # fintech, listada 2019, histórico curto
    "PRIO3",   # petróleo independente, crescimento pós-2020
    "ELET3",   # elétrica estatal, privatizada 2023
    "BEEF3",   # frigorífico, setor cíclico
]

MACRO_ENDPOINTS = {
    "macro_selic": "/macro/selic",
    "macro_ipca":  "/macro/ipca",
    "macro_cdi":   "/macro/cdi",
}


# ─────────────────────────────────────────────────────────────────────────────
# Cliente HTTP
# ─────────────────────────────────────────────────────────────────────────────
class BolsaiClient:
    def __init__(self, api_key: str, delay: float = 0.2):
        self.delay = delay
        self._http = httpx.Client(
            timeout=30,
            headers={"X-API-Key": api_key},
            follow_redirects=True,
        )

    def get(self, path: str, params: dict = None) -> Optional[dict]:
        url = f"{BASE_URL}/{path.lstrip('/')}"
        for attempt in range(1, 4):
            try:
                r = self._http.get(url, params=params or {})
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 60))
                    log.warning(f"Rate-limit → aguardando {wait}s")
                    time.sleep(wait)
                    continue
                if r.status_code in (404, 422, 500):
                    return None
                r.raise_for_status()
                time.sleep(self.delay)
                return r.json()
            except Exception as e:
                if attempt == 3:
                    log.debug(f"Falha {url}: {e}")
                    return None
                time.sleep(2 ** attempt)
        return None

    def close(self):
        self._http.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fetch: preços diários completos (paginado por data)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_prices(client: BolsaiClient, ticker: str) -> pd.DataFrame:
    """Retorna DataFrame diário com todos os pregões disponíveis."""
    all_prices = []
    start = "2000-01-01"

    while True:
        raw = client.get(f"/stocks/{ticker}/history", params={"start": start, "limit": MAX_HIST})
        if not raw:
            break
        page = raw.get("prices", [])
        if not page:
            break
        all_prices.extend(page)
        if len(page) < MAX_HIST:
            break  # última página
        last_date = page[-1]["trade_date"]
        next_start = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        if next_start <= start:
            break
        start = next_start

    if not all_prices:
        return pd.DataFrame()

    df = pd.DataFrame(all_prices)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df.insert(0, "ticker", ticker)
    df.rename(columns={
        "open":           "price_open",
        "high":           "price_high",
        "low":            "price_low",
        "close":          "price_close",
        "adjusted_open":  "price_adj_open",
        "adjusted_high":  "price_adj_high",
        "adjusted_low":   "price_adj_low",
        "adjusted_close": "price_adj_close",
        "volume":         "volume",
        "adjusted_volume":"volume_adjusted",
        "traded_amount":  "traded_amount",
        "num_trades":     "num_trades",
    }, inplace=True)
    df.sort_values("trade_date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.info(f"  [{ticker}] preços: {len(df)} pregões "
             f"({df['trade_date'].min().date()} → {df['trade_date'].max().date()})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fetch: fundamentos trimestrais
# ─────────────────────────────────────────────────────────────────────────────
def fetch_fundamentals(client: BolsaiClient, ticker: str) -> tuple[pd.DataFrame, dict]:
    """Retorna DataFrame trimestral de fundamentos + dict de info da empresa."""
    raw = client.get(f"/fundamentals/{ticker}/history", params={"limit": MAX_HIST})
    if not raw:
        return pd.DataFrame(), {}

    history = raw.get("history", [])
    if not history:
        return pd.DataFrame(), {}

    info = {
        "corporate_name":  raw.get("corporate_name"),
        "sector":          raw.get("sector"),
        "subsector":       raw.get("subsector"),
        "segment":         raw.get("segment"),
        "cvm_code":        raw.get("cvm_code"),
        "listing_segment": raw.get("listing_segment"),
        "stock_type":      raw.get("type"),
    }

    df = pd.DataFrame(history)
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    df.sort_values("reference_date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Prefixo para distinguir de colunas de preço
    fund_cols = [c for c in df.columns if c != "reference_date"]
    df.rename(columns={c: f"fund_{c}" for c in fund_cols}, inplace=True)

    log.info(f"  [{ticker}] fundamentos: {len(df)} trimestres, {len(fund_cols)} indicadores")
    return df, info


# ─────────────────────────────────────────────────────────────────────────────
# Fetch: demonstrações financeiras trimestrais (pivotadas)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_financials(client: BolsaiClient, ticker: str) -> pd.DataFrame:
    """Retorna DataFrame trimestral com linhas contábeis como colunas."""
    fin_frames = []
    cvm_code = None

    for rtype in REPORT_TYPES:
        for stype in STATEMENT_TYPES:
            raw = client.get(
                f"/financials/{ticker}",
                params={"report_type": rtype, "statement_type": stype, "limit": MAX_HIST}
            )
            if not raw:
                continue
            if cvm_code is None:
                cvm_code = raw.get("cvm_code")
            stmts = raw.get("statements", [])
            if not stmts:
                continue
            df_s = pd.DataFrame(stmts)
            df_s["_stype"] = stype
            fin_frames.append(df_s)

    if not fin_frames:
        return pd.DataFrame()

    df_fin = pd.concat(fin_frames, ignore_index=True)
    df_fin["reference_date"] = pd.to_datetime(df_fin["reference_date"])
    df_fin["value"] = pd.to_numeric(df_fin["value"], errors="coerce")

    # Label: "DRE_3_01_ReceitaLiquida"
    df_fin["_label"] = (
        df_fin["_stype"] + "_"
        + df_fin["account_code"].astype(str).str.replace(".", "_", regex=False)
        + "_"
        + df_fin["account_name"].astype(str).str[:25].str.replace(" ", "_", regex=False)
                                                   .str.replace("/", "_", regex=False)
    )

    # Pivot: uma coluna por linha contábil
    df_pivot = (
        df_fin
        .groupby(["reference_date", "_label"])["value"]
        .last()
        .unstack("_label")
        .reset_index()
    )
    df_pivot.columns.name = None
    df_pivot.sort_values("reference_date", inplace=True)
    df_pivot.reset_index(drop=True, inplace=True)

    log.info(f"  [{ticker}] financials: {len(df_pivot)} trimestres, {df_pivot.shape[1]-1} contas")
    return df_pivot


# ─────────────────────────────────────────────────────────────────────────────
# Fetch: macro diário
# ─────────────────────────────────────────────────────────────────────────────
def fetch_macro(client: BolsaiClient) -> pd.DataFrame:
    """Retorna DataFrame diário com indicadores macro."""
    frames = []
    for col_name, path in MACRO_ENDPOINTS.items():
        raw = client.get(path, params={"limit": 500})
        if not raw:
            continue
        records = raw.get("data", [])
        if not records:
            continue
        df_m = pd.DataFrame(records)
        df_m["date"] = pd.to_datetime(df_m["date"])
        df_m["value"] = pd.to_numeric(df_m["value"], errors="coerce")
        df_m.rename(columns={"date": "trade_date", "value": col_name}, inplace=True)
        frames.append(df_m[["trade_date", col_name]].set_index("trade_date"))
        log.info(f"  Macro [{col_name}]: {len(df_m)} registros")

    if not frames:
        return pd.DataFrame()

    df_macro = pd.concat(frames, axis=1).reset_index()
    df_macro.sort_values("trade_date", inplace=True)
    return df_macro


# ─────────────────────────────────────────────────────────────────────────────
# As-of join: fundamentals trimestrais → diário (sem look-ahead bias)
# ─────────────────────────────────────────────────────────────────────────────
def asof_join_quarterly_to_daily(df_daily: pd.DataFrame,
                                  df_quarterly: pd.DataFrame,
                                  date_col_q: str = "reference_date") -> pd.DataFrame:
    """
    Para cada pregão, usa os fundamentos do trimestre mais recente
    JÁ DIVULGADO até aquela data (merge_asof com direction='backward').
    Isso evita look-ahead bias.
    """
    df_q = df_quarterly.copy().sort_values(date_col_q)
    df_d = df_daily.copy().sort_values("trade_date")

    merged = pd.merge_asof(
        df_d,
        df_q,
        left_on="trade_date",
        right_on=date_col_q,
        direction="backward",
    )
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering diária
# ─────────────────────────────────────────────────────────────────────────────
def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona retornos e features técnicas simples."""
    df = df.copy()
    c = "price_adj_close" if "price_adj_close" in df.columns else "price_close"

    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df["return_1d"]   = df[c].pct_change(fill_method=None).round(6)
        df["return_5d"]   = df[c].pct_change(5, fill_method=None).round(6)
        df["return_21d"]  = df[c].pct_change(21, fill_method=None).round(6)
        df["return_63d"]  = df[c].pct_change(63, fill_method=None).round(6)
        df["log_return_1d"] = np.log(df[c] / df[c].shift(1))
        df["volatility_21d"] = df["return_1d"].rolling(21).std().round(6)
        df["ma_21d"]  = df[c].rolling(21).mean().round(4)
        df["ma_63d"]  = df[c].rolling(63).mean().round(4)
        df["ma_ratio"] = (df["ma_21d"] / df["ma_63d"]).round(6)  # momentum proxy

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["volume_ma_21d"] = df["volume"].rolling(21).mean().round(0)
        df["volume_ratio"]  = (df["volume"] / df["volume_ma_21d"]).round(4)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal por ticker
# ─────────────────────────────────────────────────────────────────────────────
def build_ticker_dataset(client: BolsaiClient, ticker: str) -> pd.DataFrame:
    log.info(f"\n{'─'*50}")
    log.info(f"Processando {ticker}...")

    # 1. Preços diários (base do dataset)
    df_prices = fetch_prices(client, ticker)
    if df_prices.empty:
        log.warning(f"  [{ticker}] sem preços — pulando")
        return pd.DataFrame()

    # 2. Fundamentos trimestrais
    df_fund, info = fetch_fundamentals(client, ticker)

    # 3. Demonstrações financeiras trimestrais
    df_fin = fetch_financials(client, ticker)

    # 4. As-of join: fundamentos → diário
    if not df_fund.empty:
        df_prices = asof_join_quarterly_to_daily(df_prices, df_fund)

    # 5. As-of join: financials → diário
    if not df_fin.empty:
        df_prices = asof_join_quarterly_to_daily(df_prices, df_fin)

    # 6. Adiciona info estática da empresa
    for col, val in info.items():
        df_prices[col] = val

    # 7. Features de preço
    df_prices = add_price_features(df_prices)

    # Remove reference_date duplicada se existir (veio do join)
    df_prices.drop(columns=["reference_date"], errors="ignore", inplace=True)

    log.info(f"  [{ticker}] dataset final: {len(df_prices)} dias × {df_prices.shape[1]} colunas")
    return df_prices


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run(api_key: str, output: str, tickers: list):
    client = BolsaiClient(api_key=api_key, delay=0.2)

    # Macro (uma vez para todos)
    log.info("Coletando dados macro...")
    df_macro = fetch_macro(client)

    frames = []
    for ticker in tickers:
        df = build_ticker_dataset(client, ticker)
        if df.empty:
            continue

        # As-of join com macro (forward-fill automático do merge_asof)
        if not df_macro.empty:
            df = pd.merge_asof(
                df.sort_values("trade_date"),
                df_macro.sort_values("trade_date"),
                on="trade_date",
                direction="backward",
            )

        frames.append(df)

    client.close()

    if not frames:
        log.error("Nenhum dado coletado.")
        return

    df_all = pd.concat(frames, ignore_index=True)

    # Ordena colunas: ID → preço → features → fundamentos → financials → macro
    id_cols    = ["ticker", "trade_date", "corporate_name", "sector",
                  "subsector", "segment", "cvm_code", "listing_segment", "stock_type"]
    price_cols = [c for c in df_all.columns if c.startswith("price_") or
                  c in ("volume", "volume_adjusted", "traded_amount", "num_trades")]
    feat_cols  = [c for c in df_all.columns if c.startswith("return_") or
                  c.startswith("log_") or c.startswith("volatility_") or
                  c.startswith("ma_") or c.startswith("volume_")]
    fund_cols  = [c for c in df_all.columns if c.startswith("fund_")]
    fin_cols   = [c for c in df_all.columns if any(
                  c.startswith(s) for s in ("BPA_","BPP_","DRE_","DFC_","DVA_"))]
    macro_cols = [c for c in df_all.columns if c.startswith("macro_")]

    ordered = (
        [c for c in id_cols if c in df_all.columns]
        + [c for c in price_cols if c in df_all.columns]
        + [c for c in feat_cols if c in df_all.columns]
        + [c for c in fund_cols if c in df_all.columns]
        + [c for c in fin_cols if c in df_all.columns]
        + [c for c in macro_cols if c in df_all.columns]
    )
    # Qualquer coluna não categorizada vai pro final
    remaining = [c for c in df_all.columns if c not in ordered]
    df_all = df_all[ordered + remaining]

    df_all.sort_values(["ticker", "trade_date"], inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    df_all.to_csv(output, index=False, encoding="utf-8-sig")

    # ── Relatório ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DATASET DIÁRIO GERADO: {output}")
    print(f"{'='*60}")
    print(f"Linhas  : {len(df_all):,}")
    print(f"Colunas : {df_all.shape[1]}")
    print(f"\nPor ticker:")
    for t, g in df_all.groupby("ticker"):
        name  = g["corporate_name"].iloc[0] if "corporate_name" in g else ""
        nulls = g[[c for c in fund_cols if c in g.columns]].isnull().mean().mean()
        print(f"  {t:<8} {len(g):>5} dias  "
              f"{g['trade_date'].min().date()} → {g['trade_date'].max().date()}  "
              f"fund_nulls={nulls:.0%}  ({name})")

    print(f"\nGrupos de colunas:")
    print(f"  ID / empresa   : {len([c for c in id_cols if c in df_all.columns])}")
    print(f"  Preço diário   : {len([c for c in price_cols if c in df_all.columns])}")
    print(f"  Features       : {len([c for c in feat_cols if c in df_all.columns])}")
    print(f"  Fundamentos    : {len([c for c in fund_cols if c in df_all.columns])}")
    print(f"  Demonstrações  : {len([c for c in fin_cols if c in df_all.columns])}")
    print(f"  Macro          : {len([c for c in macro_cols if c in df_all.columns])}")
    print(f"\nNulls médios por grupo:")
    for label, cols in [("Preço", price_cols), ("Fundamentos", fund_cols),
                         ("Financials", fin_cols), ("Macro", macro_cols)]:
        valid = [c for c in cols if c in df_all.columns]
        if valid:
            pct = df_all[valid].isnull().mean().mean()
            print(f"  {label:<15}: {pct:.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Dataset diário: preços + fundamentos forward-filled"
    )
    p.add_argument("--api-key", required=True)
    p.add_argument("--output",  default="bolsai_daily_sample.csv")
    p.add_argument("--tickers", nargs="*", default=SAMPLE_TICKERS)
    args = p.parse_args()
    run(args.api_key, args.output, [t.upper() for t in args.tickers])