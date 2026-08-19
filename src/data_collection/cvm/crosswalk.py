"""cvm/crosswalk.py — FCA valor_mobiliario: ticker -> cnpj/cvm_code/corporate_name."""

import logging
import re
from datetime import date

import pandas as pd

from .. import config
from . import http
from .filing_dates import OUTPUT_PATH as FILING_DATES_PATH

log = logging.getLogger("cvm")

CROSSWALK_PATH = config.CVM_DIR / "fca_crosswalk.parquet"

_TICKER = re.compile(r"^[A-Z0-9]{4}(?:[3-8]|11)$")
_FCA_COLS = ["ticker", "cnpj", "corporate_name", "end_trading", "year"]

# Tickers with no Codigo_Negociacao in any FCA year (crosswalk.py's documented 2010-2017
# blank-field + survivor-style limits, or a filer whose FCA simply never lists this exact
# code) -- confirmed 2026-08-19 to all have real CVM statement data under these CNPJs
# (verified via cvm/statements.py's cnpj-keyed lookup), just no ticker link. Same pattern as
# sec/crosswalk.py's CIK_OVERRIDES. cnpj/corporate_name seeded from company_info.parquet.
TICKER_CNPJ_OVERRIDES = {
    "AMAR3": ("61189288000189", "MARISA LOJAS SA"),
    "BPAC11": ("30306294000145", "BANCO BTG PACTUAL S/A"),
    "CMIN3": ("08902291000115", "CSN MINERAÇÃO S.A."),
    "CSNA3": ("33042730000104", "CIA SIDERURGICA NACIONAL"),
    "EGGY3": ("81616807000155", "GRANJA FARIA S.A."),
    "EPAR3": ("42331462000131", "EMBPAR PARTICIPAÇÕES S/A"),
    "EQMA3B": ("06272793000184", "EQUATORIAL MARANHÃO DISTRIBUIDORA DE ENERGIA S.A."),
    "EQPA3": ("04895728000180", "EQUATORIAL PARÁ DISTRIBUIDORA DE ENERGIA S.A."),
    "HAGA4": ("30540991000166", "HAGA S.A. INDÚSTRIA E COMÉRCIO"),
    "LUXM4": ("92660570000126", "TREVISA  INVESTIMENTOS SA"),
    "MAPT4": ("93828986000173", "CEMEPE INVESTIMENTOS SA"),
    "MBRF3": ("03853896000140", "MARFRIG GLOBAL FOODS SA"),
    "MRSA3B": ("01417222000177", "MRS LOGÍSTICA S/A"),
    "NORD3": ("60884319000159", "NORDON INDUSTRIAS METALURGICAS S.A."),
    "OBTC3": ("59693110000129", "ORANJEBTC S.A. - EDUCAÇÃO E INVESTIMENTO"),
    "PLAS3": ("51928174000150", "PLASCAR PARTICIPAÇÕES INDUSTRIAIS S.A"),
    "WDCN3": ("05917486000140", "LIVETECH DA BAHIA INDÚSTRIA E COMÉRCIO S.A."),
}


def build_crosswalk() -> pd.DataFrame:
    """ticker -> cnpj, cvm_code, corporate_name, end_trading. Latest FCA wins per ticker.
    Per-year FCA rows cached to data/raw/br/cvm/fca_{year}.parquet; only the current
    year is re-downloaded on rerun (new filings arrive all year).

    Two real limits, both verified live against CVM's own zips (2026-08-15), not
    just this repo's cache:
      1. Codigo_Negociacao (the trading code itself) is 100% blank in every FCA
         filing 2010-2017 -- populated only from 2018 on. A company that delisted
         before 2018 can never get a ticker from this source, regardless of
         START_YEAR (http.py) or how many years are re-scanned.
      2. FCA reports the code AS OF FILING, survivor-style, same failure mode as
         SEC's company_tickers.json (see sec/universe.py's docstring): a ticker
         that was renamed or delisted stops appearing in ANY subsequent year's
         file. Confirmed on KROT3 -> COGN3 (2019 rename): KROT3 appears in zero
         FCA years 2018-2026, only COGN3 does. This crosswalk cannot recover a
         renamed/delisted ticker no matter which years are scanned -- see
         build_dataset/terminal_events.find_rename_candidates() for the
         registry-status-based recovery path instead.
    """
    config.CVM_DIR.mkdir(parents=True, exist_ok=True)
    current = date.today().year
    frames = []
    for year in range(http.START_YEAR, current + 1):
        cache = config.CVM_DIR / f"fca_{year}.parquet"
        if cache.exists() and year < current:
            frames.append(pd.read_parquet(cache))
            continue
        zf = http.fetch_zip("FCA", year)
        if zf is None:
            continue
        rows = []
        for r in http.read_csv(zf, f"fca_cia_aberta_valor_mobiliario_{year}.csv"):
            ticker = (r.get("Codigo_Negociacao") or "").strip().upper()
            if not _TICKER.match(ticker):
                continue
            rows.append({
                "ticker": ticker,
                "cnpj": http.digits(r.get("CNPJ_Companhia")),
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

    override_rows = pd.DataFrame([
        {"ticker": t, "cnpj": cnpj, "corporate_name": name, "end_trading": None}
        for t, (cnpj, name) in TICKER_CNPJ_OVERRIDES.items() if t not in set(df["ticker"])
    ])
    if not override_rows.empty:
        df = pd.concat([df, override_rows], ignore_index=True)
        log.info("crosswalk: +%d ticker(s) from TICKER_CNPJ_OVERRIDES", len(override_rows))

    # cvm_code via filing_dates (already on disk, cnpj+cvm_code per filing)
    if FILING_DATES_PATH.exists():
        fd = pd.read_parquet(FILING_DATES_PATH)[["cnpj", "cvm_code"]].drop_duplicates("cnpj")
        df = df.merge(fd, on="cnpj", how="left")
    else:
        df["cvm_code"] = None
        log.warning("filing_dates.parquet missing — cvm_code left null "
                    "(run: python -m src.data_collection.br.cvm_statements --step filing_dates)")

    df["end_trading"] = pd.to_datetime(df["end_trading"], errors="coerce")
    df.to_parquet(CROSSWALK_PATH, index=False)
    log.info("crosswalk: %d tickers (%d with cvm_code) -> %s",
             len(df), df["cvm_code"].notna().sum(), CROSSWALK_PATH)
    return df
