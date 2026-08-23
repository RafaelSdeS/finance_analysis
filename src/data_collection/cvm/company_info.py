"""cvm/company_info.py — status (ATIVO/CANCELADA) refresh + new-row synthesis
for company_info.parquet, sourced from CVM's own CAD registry via
cvm/delistings.py's build_delist_events() -- not BolsAI's CANCELADA registry.

Two things this buys for free that the old BolsAI-only collection never did:
refreshing `status` on every run (catches ATIVO->CANCELADA transitions, not
just a one-time snapshot at initial collection) and covering the full
crosswalk-resolvable universe, not just what BolsAI's own /companies/
registry happened to carry (see crosswalk.py's TICKER_CNPJ_OVERRIDES note on
BolsAI's registry gaps).
"""

import logging

import numpy as np
import pandas as pd

from .. import config
from .crosswalk import CROSSWALK_PATH, build_crosswalk
from .delistings import build_delist_events
from .sectors import sector_by_ticker

log = logging.getLogger("cvm")

_COLS = ["ticker", "ticker_primary", "corporate_name", "trade_name",
         "cvm_code", "cnpj", "sector", "status"]

# CVM's SIT is a COMPANY-level (CNPJ) status, not a ticker-level one. A renamed
# ticker (old code retired, company keeps trading under a new one) still shows
# ATIVO at the CNPJ -- verified live 2026-08-19: BMGB11/BRDT3/OMGE3 all come
# back "ATIVO" from CVM despite not having traded since 2019-2021 (exactly
# CLAUDE.md's documented "unspliced rename" pattern, the same one
# terminal_events.find_rename_candidates() exists to catch). Blindly trusting
# CVM's ATIVO here would silently re-admit dead ticker codes into the active
# universe. But the same check must not block a REAL fix: ITUB3 also came back
# CANCELADA->ATIVO in this same run, and its last stored price is 2026-07-10 --
# genuinely still trading, BolsAI's CANCELADA was simply stale there. The
# discriminator is recency of the ticker's own last observed price, not the
# direction of the transition.
_REACTIVATION_STALE_DAYS = 120


def _recently_traded(ticker: str) -> bool:
    path = config.PRICES_DIR / f"{ticker}.parquet"
    if not path.exists():
        return False
    last = pd.read_parquet(path, columns=["trade_date"])["trade_date"].max()
    return (pd.Timestamp.now() - last).days <= _REACTIVATION_STALE_DAYS


def synthesize_company_info() -> None:
    """Refresh `status`/`sector` for every ticker CVM's CAD registry resolves, and
    append rows for delisted tickers not yet in company_info.parquet. Reuses
    build_delist_events() (already downloads+dedupes cad_cia_aberta.csv) rather
    than re-fetching; `trade_name` is left NaN for newly-added rows -- CVM's CAD
    has no clean equivalent split, honest gap rather than a guess.

    Calls build_crosswalk() first (cache-and-skip on every year but the current one,
    same cheap pattern as cvm/ratios.py's collect_fundamentals_cvm()) so a ticker that
    just IPO'd and filed its first FCA this year is discoverable without BolsAI's
    /stocks/ registry -- see BOLSAI_EXIT_PLAN.md S4."""
    build_crosswalk()
    xwalk = pd.read_parquet(CROSSWALK_PATH)[["ticker", "cnpj", "corporate_name", "cvm_code"]]
    delist = build_delist_events().drop_duplicates("ticker", keep="last")  # ticker, cnpj, delist_date, motivo_cancel, sit
    sector_by = sector_by_ticker()

    path = config.COMPANY_INFO_PATH
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=_COLS)

    status_by_ticker = delist.set_index("ticker")["sit"]
    refreshed = existing.copy()
    known = refreshed["ticker"].isin(status_by_ticker.index)
    old_status = refreshed.loc[known, "status"].to_numpy()
    new_status = refreshed.loc[known, "ticker"].map(status_by_ticker).to_numpy()

    # Block the risky direction (existing status not ATIVO -> CVM says ATIVO)
    # unless the ticker's own price history confirms it's actually still trading.
    reactivating = (old_status != "ATIVO") & (new_status == "ATIVO")
    blocked = 0
    for i in np.flatnonzero(reactivating):
        if not _recently_traded(refreshed.loc[known].iloc[i]["ticker"]):
            new_status[i] = old_status[i]  # keep the old (non-ATIVO) status
            blocked += 1

    changed = known.copy()
    changed[known] = old_status != new_status
    refreshed.loc[known, "status"] = new_status

    # sector: CVM's SETOR_ATIV replaces BolsAI's taxonomy wherever it resolves
    # (deliberate source swap, not a gap-fill -- see BOLSAI_EXIT_PLAN.md Task 4;
    # nothing downstream pattern-matches the literal string, verified against
    # cross_sectional.py). Falls back to the existing value where CVM has none.
    has_sector = refreshed["ticker"].isin(sector_by.index)
    refreshed.loc[has_sector, "sector"] = refreshed.loc[has_sector, "ticker"].map(sector_by)

    new_tickers = delist[~delist["ticker"].isin(existing["ticker"])].merge(
        xwalk, on=["ticker", "cnpj"], how="left")
    new_rows = pd.DataFrame({
        "ticker": new_tickers["ticker"],
        "ticker_primary": new_tickers["ticker"],
        "corporate_name": new_tickers["corporate_name"],
        "trade_name": None,
        "cvm_code": new_tickers["cvm_code"],
        "cnpj": new_tickers["cnpj"],
        "sector": new_tickers["ticker"].map(sector_by),
        "status": new_tickers["sit"],
    })

    out = (pd.concat([refreshed, new_rows], ignore_index=True)
             .drop_duplicates("ticker", keep="last"))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    log.info("company_info: %d status change(s) (%d reactivation(s) blocked as stale), "
              "+%d new row(s) -> %d total",
              int(changed.sum()), blocked, len(new_rows), len(out))
