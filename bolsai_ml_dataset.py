"""
bolsai_ml_dataset.py
====================
Gera um único CSV "wide" para Machine Learning com:
  - Uma linha por (ticker, reference_date)
  - Colunas: identificação + info da empresa + 27+ fundamentos históricos
             + preços (open/high/low/close/adj_close/volume)
             + dividendos agregados (DY trailing, total pago no trimestre)
             + dados macro do período (Selic, IPCA, Ibovespa)
             + features de engenharia prontas (retornos, variações, flags)

Saída: bolsai_ml_dataset.csv  (~350 ações × ~44 trimestres = ~15 mil linhas)

Requisitos:
  pip install httpx pandas tqdm

Uso:
  python bolsai_ml_dataset.py --api-key sk_SUA_CHAVE
  python bolsai_ml_dataset.py --api-key sk_SUA_CHAVE --workers 5 --delay 0.15
"""

import httpx
import pandas as pd
import numpy as np
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
BASE_URL        = "https://api.usebolsai.com/api/v1"
MAX_HIST        = 80    # limite máximo da API para fundamentals e prices (422 acima disso)
MAX_HIST_PRICES = 80   # mesmo limite se aplica a /stocks/{ticker}/history
MAX_HIST_MACRO  = 500   # macro aceita valores maiores
STOCKS_PAGE_SIZE = 500  # API retorna até 500 tickers por página
REPORT_TYPES    = ["DFP", "ITR"]
STATEMENT_TYPES = ["BPA", "BPP", "DRE", "DFC_MI", "DVA"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("bolsai_ml.log")],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Cliente HTTP com retry + rate-limit handling
# ─────────────────────────────────────────────────────────────────────────────
class BolsaiClient:
    def __init__(self, api_key: str, delay: float = 0.2):
        self.delay  = delay
        self._http  = httpx.Client(
            timeout=30,
            headers={"X-API-Key": api_key},
            follow_redirects=True,
        )

    def get(self, path: str, params: dict = None, retries: int = 4) -> Optional[dict]:
        url = f"{BASE_URL}/{path.lstrip('/')}"
        for attempt in range(1, retries + 1):
            try:
                r = self._http.get(url, params=params or {})
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 60))
                    log.warning(f"Rate-limit → aguardando {wait}s")
                    time.sleep(wait)
                    continue
                if r.status_code in (404, 422, 500):
                    # 404 = não existe; 422 = ticker inválido; 500 = bug no servidor
                    # Todos são silenciosos — sem retry
                    return None
                r.raise_for_status()
                time.sleep(self.delay)
                return r.json()
            except Exception as e:
                if attempt == retries:
                    log.debug(f"Falha definitiva {url}: {e}")
                    return None
                time.sleep(2 ** attempt)
        return None

    def close(self):
        self._http.close()


# ─────────────────────────────────────────────────────────────────────────────
# Coleta de dados por ticker → retorna DataFrame long (uma linha por trimestre)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_all_for_ticker(client: BolsaiClient, ticker: str) -> pd.DataFrame:
    """
    Puxa TODOS os dados históricos de um ticker e retorna um DataFrame
    com uma linha por trimestre (reference_date), pronto para merge.
    """

    # ── 1. Info cadastral — vem embutida no response de fundamentals ────────
    # (endpoint /stocks/{ticker} retorna 404; não usar)
    info_cols = {}  # será preenchido após fetch de fundamentals

    # ── 2. Histórico de fundamentos (núcleo do dataset) ───────────────────
    fund_raw = client.get(f"/fundamentals/{ticker}/history", params={"limit": MAX_HIST})
    if not fund_raw:
        return pd.DataFrame()

    history = fund_raw.get("history", [])
    if not history:
        return pd.DataFrame()

    # Info cadastral vem no topo do response de fundamentals
    info_cols = {
        "corporate_name":  fund_raw.get("corporate_name", fund_raw.get("name")),
        "sector":          fund_raw.get("sector",         fund_raw.get("setor")),
        "subsector":       fund_raw.get("subsector",      fund_raw.get("subsetor")),
        "segment":         fund_raw.get("segment",        fund_raw.get("segmento")),
        "cvm_code":        fund_raw.get("cvm_code",       fund_raw.get("codigo_cvm")),
        "listing_segment": fund_raw.get("listing_segment"),
        "stock_type":      fund_raw.get("type",           fund_raw.get("tipo")),
        "isin":            fund_raw.get("isin"),
    }

    df = pd.DataFrame(history)
    df.insert(0, "ticker", ticker)

    # Garante coluna de data padronizada
    date_col = next(
        (c for c in df.columns if "date" in c.lower() or "periodo" in c.lower()),
        None
    )
    if date_col and date_col != "reference_date":
        df.rename(columns={date_col: "reference_date"}, inplace=True)
    if "reference_date" not in df.columns:
        return pd.DataFrame()

    df["reference_date"] = pd.to_datetime(df["reference_date"])
    df.sort_values("reference_date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Adiciona colunas estáticas de info
    for col, val in info_cols.items():
        df[col] = val

    # ── 3. Preços históricos ────────────────────────────────────────────────
    price_raw = client.get(f"/stocks/{ticker}/history", params={"limit": MAX_HIST_PRICES})
    if price_raw:
        prices = price_raw.get("prices", [])
        df_p = pd.DataFrame(prices)
        if not df_p.empty:
            # Campo de data confirmado: trade_date
            df_p["reference_date"] = pd.to_datetime(df_p["trade_date"])
            df_p["quarter"] = df_p["reference_date"].dt.to_period("Q")
            df["quarter"]   = df["reference_date"].dt.to_period("Q")

            # Agrega preços diários → estatísticas por trimestre
            # Schema confirmado pelo curl:
            #   open, high, low, close (raw)
            #   adjusted_open, adjusted_high, adjusted_low, adjusted_close
            #   volume, adjusted_volume, traded_amount, num_trades
            agg_dict = {
                "open":             "first",
                "high":             "max",
                "low":              "min",
                "close":            "last",
                "adjusted_open":    "first",
                "adjusted_high":    "max",
                "adjusted_low":     "min",
                "adjusted_close":   "last",
                "volume":           "sum",
                "adjusted_volume":  "sum",
                "traded_amount":    "sum",
                "num_trades":       "sum",
            }
            # Filtra apenas colunas que existem no df_p
            agg_dict = {k: v for k, v in agg_dict.items() if k in df_p.columns}

            df_p_q = df_p.groupby("quarter").agg(agg_dict).reset_index()

            # Renomeia para prefixo price_ para clareza no dataset ML
            rename_map = {
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
            }
            df_p_q.rename(columns=rename_map, inplace=True)

            df = df.merge(df_p_q, on="quarter", how="left")

    # ── 4. Demonstrações financeiras (pivotadas como features) ─────────────
    # Dividendos: endpoint /stocks/{ticker}/dividends retorna 404 — não existe na API
    fin_frames = []
    cvm_code_from_fin = None
    for rtype in REPORT_TYPES:
        for stype in STATEMENT_TYPES:
            raw = client.get(
                f"/financials/{ticker}",
                params={"report_type": rtype, "statement_type": stype, "limit": MAX_HIST}
            )
            if not raw:
                continue
            # cvm_code vem no topo do response de financials (confirmado pelo teste)
            if cvm_code_from_fin is None:
                cvm_code_from_fin = raw.get("cvm_code")
            stmts = raw.get("statements", raw.get("data", []))
            if not stmts:
                continue
            df_s = pd.DataFrame(stmts)
            df_s["_rtype"] = rtype
            df_s["_stype"] = stype
            fin_frames.append(df_s)

    if fin_frames:
        df_fin = pd.concat(fin_frames, ignore_index=True)

        # Detecta colunas chave
        fin_date  = next((c for c in df_fin.columns if "date" in c.lower()), None)
        fin_code  = next((c for c in df_fin.columns if "account_code" in c.lower()
                          or "codigo" in c.lower()), None)
        fin_name  = next((c for c in df_fin.columns if "account_name" in c.lower()
                          or "descricao" in c.lower() or "nome" in c.lower()), None)
        fin_val   = next((c for c in df_fin.columns if "value" in c.lower()
                          or "valor" in c.lower() or "amount" in c.lower()), None)

        if fin_date and fin_val:
            df_fin["reference_date"] = pd.to_datetime(df_fin[fin_date])
            df_fin[fin_val] = pd.to_numeric(df_fin[fin_val], errors="coerce")
            df_fin["quarter"] = df_fin["reference_date"].dt.to_period("Q")

            # Cria label legível: "DRE_3.01_ReceitaLiquida"
            if fin_code and fin_name:
                df_fin["_label"] = (
                    df_fin["_stype"] + "_"
                    + df_fin[fin_code].astype(str).str.replace(".", "_", regex=False)
                    + "_"
                    + df_fin[fin_name].astype(str).str[:30].str.replace(" ", "_", regex=False)
                )
            elif fin_code:
                df_fin["_label"] = df_fin["_stype"] + "_" + df_fin[fin_code].astype(str)
            else:
                df_fin["_label"] = df_fin["_stype"]

            # Pivot: uma coluna por linha contábil
            df_pivot = (
                df_fin
                .groupby(["quarter", "_label"])[fin_val]
                .last()   # pega o valor mais recente se houver duplicatas
                .unstack("_label")
                .reset_index()
            )
            df_pivot.columns.name = None

            if "quarter" not in df.columns:
                df["quarter"] = df["reference_date"].dt.to_period("Q")
            df = df.merge(df_pivot, on="quarter", how="left")

    # Complementa cvm_code com o valor vindo dos financials se necessário
    if cvm_code_from_fin and not df.get("cvm_code", pd.Series([None])).iloc[0]:
        df["cvm_code"] = cvm_code_from_fin

    # ── 6. Feature Engineering ─────────────────────────────────────────────
    # Força conversão numérica em todas as colunas relevantes antes de qualquer cálculo
    for col in df.columns:
        if col not in ("ticker", "reference_date", "quarter",
                       "corporate_name", "sector", "subsector", "segment",
                       "listing_segment", "stock_type", "cvm_code", "isin"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Retorno de preço trimestral
    if "price_close" in df.columns:
        df["return_q"]     = df["price_close"].pct_change(fill_method=None).round(6)
        df["return_1y"]    = df["price_close"].pct_change(4, fill_method=None).round(6)
        df["return_3y"]    = df["price_close"].pct_change(12, fill_method=None).round(6)
        df["log_return_q"] = np.log(df["price_close"] / df["price_close"].shift(1))

    # Variação trimestral dos principais fundamentos
    for col in ["pl", "pvp", "roe", "ev_ebitda", "net_margin", "market_cap"]:
        if col in df.columns:
            df[f"{col}_chg_q"]  = df[col].pct_change(fill_method=None).round(6)
            df[f"{col}_chg_1y"] = df[col].pct_change(4, fill_method=None).round(6)

    # Limpeza de coluna auxiliar
    if "quarter" in df.columns:
        df["quarter"] = df["quarter"].astype(str)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Dados macro: merge por trimestre no dataset final
# ─────────────────────────────────────────────────────────────────────────────
MACRO_ENDPOINTS = {
    "selic":     "/macro/selic",
    "ipca":      "/macro/ipca",
    "igpm":      "/macro/igpm",
    "cdi":       "/macro/cdi",
    "ibovespa":  "/macro/ibovespa",
    "cambio_usd":"/macro/exchange-rate",
}

def fetch_macro_quarterly(client: BolsaiClient) -> pd.DataFrame:
    """Retorna DataFrame com uma linha por trimestre com indicadores macro."""
    frames = []
    for name, path in MACRO_ENDPOINTS.items():
        raw = client.get(path, params={"limit": MAX_HIST_MACRO})
        if not raw:
            continue
        records = (
            raw.get("data") or raw.get("rates") or raw.get("history")
            or raw.get("results") or [raw]
        )
        if not isinstance(records, list):
            continue
        df_m = pd.DataFrame(records)
        d_col = next((c for c in df_m.columns if "date" in c.lower()), None)
        v_col = next((c for c in df_m.columns
                      if any(k in c.lower() for k in ["value","rate","taxa","valor","close"])), None)
        if not d_col or not v_col:
            continue
        df_m["reference_date"] = pd.to_datetime(df_m[d_col])
        df_m[v_col] = pd.to_numeric(df_m[v_col], errors="coerce")
        df_m["quarter"] = df_m["reference_date"].dt.to_period("Q")

        agg = df_m.groupby("quarter")[v_col].mean().reset_index()
        agg.rename(columns={v_col: f"macro_{name}"}, inplace=True)
        frames.append(agg.set_index("quarter"))
        log.info(f"Macro [{name}]: {len(agg)} trimestres")

    if not frames:
        return pd.DataFrame()

    df_macro = pd.concat(frames, axis=1).reset_index()
    df_macro["quarter"] = df_macro["quarter"].astype(str)
    return df_macro


# ─────────────────────────────────────────────────────────────────────────────
# Orquestrador principal
# ─────────────────────────────────────────────────────────────────────────────
def run(api_key: str, workers: int, delay: float, output: str, tickers_override: list):
    client = BolsaiClient(api_key=api_key, delay=delay)
    out_path = Path(output)

    # 1. Lista completa de tickers (paginada — API retorna até 500 por vez, total ~5373)
    if tickers_override:
        tickers = [t.upper() for t in tickers_override]
        log.info(f"Tickers especificados manualmente: {len(tickers)}")
    else:
        log.info("Buscando lista completa de ações (paginada)...")
        tickers = []
        offset = 0
        while True:
            raw_list = client.get("/stocks/", params={"limit": STOCKS_PAGE_SIZE, "offset": offset}) or {}
            page_tickers = raw_list.get("tickers", [])
            if not page_tickers:
                break
            tickers.extend(page_tickers)
            total = raw_list.get("total", 0)
            log.info(f"  Coletados {len(tickers)}/{total} tickers...")
            if len(tickers) >= total:
                break
            offset += STOCKS_PAGE_SIZE
        log.info(f"Total de ações disponíveis: {len(tickers)}")
        # Filtra tickers inválidos (com espaço, vazios, etc.)
        tickers = [t for t in tickers if t and " " not in t and len(t) <= 8]
        log.info(f"Tickers válidos após filtro: {len(tickers)}")

    # 2. Dados macro (uma vez só)
    log.info("Coletando dados macroeconômicos...")
    df_macro = fetch_macro_quarterly(client)

    # 3. Coleta paralela por ticker
    all_frames = []
    errors     = []

    log.info(f"Iniciando coleta com {workers} worker(s) e delay={delay}s...")

    def _collect(ticker):
        try:
            df = fetch_all_for_ticker(client, ticker)
            return ticker, df, None
        except Exception as e:
            return ticker, pd.DataFrame(), str(e)

    CHECKPOINT_EVERY = 100   # salva CSV parcial a cada N tickers com sucesso
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_collect, t): t for t in tickers}
        for future in tqdm(as_completed(futures), total=len(tickers), desc="Coletando"):
            ticker, df, err = future.result()
            if err:
                errors.append({"ticker": ticker, "error": err})
                log.warning(f"[{ticker}] erro: {err}")
            elif df.empty:
                errors.append({"ticker": ticker, "error": "empty"})
            else:
                all_frames.append(df)
                completed += 1

                # Checkpoint parcial
                if completed % CHECKPOINT_EVERY == 0:
                    checkpoint_path = out_path.with_suffix(f".partial_{completed}.csv")
                    pd.concat(all_frames, ignore_index=True).to_csv(
                        checkpoint_path, index=False, encoding="utf-8-sig"
                    )
                    log.info(f"Checkpoint salvo: {checkpoint_path} ({completed} tickers)")

    if not all_frames:
        log.error("Nenhum dado coletado. Verifique sua API key e conexão.")
        client.close()
        return

    # 4. Concatena todos os tickers
    log.info("Concatenando todos os tickers...")
    df_all = pd.concat(all_frames, ignore_index=True)
    log.info(f"Shape após concat: {df_all.shape}")

    # 5. Merge com macro por trimestre
    if not df_macro.empty and "quarter" in df_all.columns:
        log.info("Mergeando dados macroeconômicos...")
        df_all = df_all.merge(df_macro, on="quarter", how="left")
        log.info(f"Shape após merge macro: {df_all.shape}")

    # 6. Ordenação e limpeza final
    df_all["reference_date"] = pd.to_datetime(df_all["reference_date"])
    df_all.sort_values(["ticker", "reference_date"], inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    # Remove colunas 100% nulas
    before = df_all.shape[1]
    df_all.dropna(axis=1, how="all", inplace=True)
    after = df_all.shape[1]
    if before != after:
        log.info(f"Removidas {before - after} colunas completamente nulas.")

    # Coloca colunas de identificação na frente
    id_cols = ["ticker", "reference_date", "quarter",
               "corporate_name", "sector", "subsector", "segment",
               "listing_segment", "stock_type", "cvm_code", "isin"]
    id_cols = [c for c in id_cols if c in df_all.columns]
    other_cols = [c for c in df_all.columns if c not in id_cols]
    df_all = df_all[id_cols + other_cols]

    # 7. Salva o dataset principal
    log.info(f"Salvando dataset em {out_path}...")
    df_all.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 8. Salva log de erros se houver
    if errors:
        err_path = out_path.parent / "bolsai_ml_errors.csv"
        pd.DataFrame(errors).to_csv(err_path, index=False)
        log.warning(f"{len(errors)} tickers com erro → {err_path}")

    # 9. Resumo final
    log.info("=" * 60)
    log.info("DATASET GERADO COM SUCESSO")
    log.info(f"  Arquivo  : {out_path.resolve()}")
    log.info(f"  Linhas   : {len(df_all):,}  (ticker × trimestre)")
    log.info(f"  Colunas  : {df_all.shape[1]}")
    log.info(f"  Tickers  : {df_all['ticker'].nunique()}")
    log.info(f"  Período  : {df_all['reference_date'].min().date()} → "
             f"{df_all['reference_date'].max().date()}")
    log.info(f"  Nulls    : {df_all.isnull().mean().mean():.1%} (média por célula)")
    log.info("=" * 60)

    # 10. Imprime amostra das colunas para o usuário saber o que tem
    print("\n── COLUNAS DO DATASET ──────────────────────────────────────")
    for i, col in enumerate(df_all.columns, 1):
        dtype = str(df_all[col].dtype)
        null_pct = df_all[col].isnull().mean()
        print(f"  {i:3}. {col:<55} {dtype:<10} {null_pct:.0%} nulo")
    print(f"\nTotal: {df_all.shape[1]} colunas × {len(df_all):,} linhas\n")

    client.close()
    return df_all


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="bolsai → CSV único para ML (uma linha por ticker×trimestre)"
    )
    p.add_argument("--api-key",  required=True,
                   help="Sua API key bolsai (plano pago). Ex: sk_xxxxxxxx")
    p.add_argument("--workers",  type=int,   default=5,
                   help="Threads paralelas (plano pago suporta mais; default=5)")
    p.add_argument("--delay",    type=float, default=0.15,
                   help="Delay entre requisições em segundos (default=0.15)")
    p.add_argument("--output",   type=str,   default="bolsai_ml_dataset.csv",
                   help="Nome do arquivo de saída (default: bolsai_ml_dataset.csv)")
    p.add_argument("--tickers",  nargs="*",
                   help="Opcional: processar apenas esses tickers (ex: PETR4 VALE3)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        api_key=args.api_key,
        workers=args.workers,
        delay=args.delay,
        output=args.output,
        tickers_override=args.tickers or [],
    )