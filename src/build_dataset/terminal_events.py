"""
terminal_events.py — realized payoff for BR tickers that die inside the
built panel, sourced from CVM's own cancellation registry
(data_collection/cvm/delistings.py) instead of leaving every delisted
ticker's forward-return label an unexplained NaN.

Renames/mergers continuity.py already spliced (data/raw/br/reference/
ticker_continuity.json) never reach this module as "dead" -- they're folded
into the surviving ticker before Stage 2 output, so there's no double-
handling here. What's left standalone-dead falls into two buckets straight
from CVM's own SIT/MOTIVO_CANCEL fields:

  - sit == 'ATIVO': the ticker's own series stopped but the COMPANY is still
    registered active -- an unspliced rename/share-class consolidation, not
    a real delisting. No terminal payoff; see find_rename_candidates().
  - otherwise, with a resolved MOTIVO_CANCEL: a real terminal event.
    Bankruptcy/liquidation reasons pay 0; every other reason (voluntary
    cancellation, incorporation/merger) pays the ticker's own last observed
    adj_close. Measured 2026-08-15 against the real BR panel: of 114 tickers
    that died inside it with enough history, 64 died while RISING over their
    final 60 trading days (median +2.9%) -- acquisition-at-a-premium is the
    dominant BR terminal event, not wipeout, so the last quoted price is a
    much better estimate than a blanket -100%.
  - unresolvable (no crosswalk match, no registry row): stays NaN -- today's
    behavior, unchanged.

Run after build_ml_dataset.py (needs the built panel on disk):
    python -m src.build_dataset.terminal_events
"""

import pandas as pd

from src.data_collection.cvm.delistings import build_delist_events
from src.data_collection.cvm.delistings import OUTPUT_PATH as DELIST_EVENTS_PATH

from .paths import OUTPUT_PATH, TERMINAL_EVENTS_PATH
from .quality_filters import STALE_TICKER_DAYS

# Registry reasons that mean the equity was actually wiped out. Every other
# resolved reason (voluntary cancellation, incorporation/merger) is treated
# as "acquired" and paid the last observed price -- see module docstring for
# the measurement backing that split.
FAILURE_REASONS = {
    "LIQUIDAÇÃO EXTRAJUDICIAL",
    "ELISÃO POR EXTINÇÃO DA CIA",
    "CANCELAMENTO DE OFÍCIO",
}


def _dead_tickers(df: pd.DataFrame) -> set:
    """Tickers whose series ended well before the panel's own last date --
    the same STALE_TICKER_DAYS threshold quality_filters.py already uses to
    call a ticker delisted/renamed rather than a live coverage gap."""
    last = df.groupby("ticker")["trade_date"].max()
    end = df["trade_date"].max()
    return set(last[(end - last).dt.days > STALE_TICKER_DAYS].index)


def build_terminal_events(df: pd.DataFrame, delist_events: pd.DataFrame) -> pd.DataFrame:
    """One row per dead ticker with a resolvable CVM cancellation reason.

    df: the built panel (ticker, trade_date, adj_close) -- only tickers that
    actually died INSIDE df are considered; a crosswalk/registry match on a
    still-live ticker is not a terminal event.

    Returns columns: ticker, delist_date, event_type ('failure'|'acquired'),
    terminal_payoff.
    """
    dead = _dead_tickers(df)
    events = delist_events[
        delist_events["ticker"].isin(dead) & (delist_events["sit"] != "ATIVO")
    ].dropna(subset=["motivo_cancel"]).copy()

    last_close = (
        df[df["adj_close"] > 0]
        .sort_values("trade_date")
        .groupby("ticker")["adj_close"].last()
        .rename("last_adj_close")
    )
    events = events.merge(last_close, on="ticker", how="inner")

    events["event_type"] = events["motivo_cancel"].apply(
        lambda m: "failure" if m in FAILURE_REASONS else "acquired"
    )
    events["terminal_payoff"] = events["last_adj_close"].where(events["event_type"] == "acquired", 0.0)

    return events[["ticker", "delist_date", "event_type", "terminal_payoff"]].reset_index(drop=True)


# Confirmed via web research 2026-08-16 (docs/DATA_INTEGRITY_TEST_PLAN.md investigation
# into the terminal-events coverage gap). CVM's registry shows every one of these
# entities' `sit` as ATIVO or SUSPENSO(A) at the COMPANY level, with no resolved
# motivo_cancel -- so build_terminal_events()'s registry path can't classify them, even
# though the TICKER genuinely stopped trading and the real-world outcome is well
# documented in public filings/press. Same precedent as sec/crosswalk.py's
# CIK_OVERRIDES: a small, hand-verified table for cases the automated join can't reach.
#
# "failure": bankruptcy/liquidation decreed, equity wiped out -> payoff 0.
# "acquired": bought out / merger / going-private tender offer -> payoff = last observed
#   adj_close, same convention build_terminal_events() already uses. For the three
#   100%-stock-swap mergers (OMGE3, IGTA3, CESP3/CESP5) this is a known approximation --
#   the true payoff is the successor entity's later share value (Serena/MEGA3, IGTI11,
#   AURE3 respectively), not a cash-confirmed figure like the four OPAs below it.
#
# Explicitly NOT included here (research surfaced these but they need different
# handling, not a terminal payoff):
#   - Confirmed renames needing a verified exchange ratio before ticker_continuity.json
#     can splice them: BBRK3->NEXP3, CELP5/6/7->EQPA5/6/7, LIQO3->ATMP3->CTAX3.
#   - ~13 tickers (incl. AMER3/Americanas, LIGT3) that are STILL ACTIVELY TRADING --
#     their panel just stopped being collected (missing from company_info.parquet,
#     invisible to both --mode update and collect_delisted.py). A terminal payoff for
#     these would fabricate a death for a live company; see the collection-gap fix
#     instead.
#   - ~9 tickers genuinely still mid-crisis with no resolved outcome (Oi, Rossi,
#     Bardella, João Fortes, Coteminas, Mendes Júnior) -- correctly left unlabeled.
MANUAL_TERMINAL_EVENTS = {
    "BPHA3": "failure",   # Brasil Pharma: bankruptcy decreed Jun 2019 (2nd Bankruptcy Court of SP)
    "SLED3": "failure",   # Saraiva: falência decreed Oct 2023
    "SLED4": "failure",   # Saraiva: falência decreed Oct 2023
    "FRTA3": "failure",   # Pomifrutas: autofalência Apr 2024, assets auctioned
    "TEKA3": "failure",   # Teka: RJ since 2012 failed, bankruptcy decreed ~Feb 2025
    "TEKA4": "failure",   # Teka: RJ since 2012 failed, bankruptcy decreed ~Feb 2025
    "ENBR3": "acquired",  # EDP going-private tender offer, settled Jul 2023 at R$23.73/share
    "CPRE3": "acquired",  # CPFL Energias Renováveis OPA, concluded Jun 2020 at R$18.24/share
    "PRBC4": "acquired",  # Parana Banco 2017 OPA at R$11.59/share
    "CEPE3": "acquired",  # Celpe: Neoenergia OPA + Eletrobras stake auction Oct 2022, ~R$42-46
    "CEPE5": "acquired",  # Celpe: Neoenergia OPA + Eletrobras stake auction Oct 2022, ~R$42-46
    "CEPE6": "acquired",  # Celpe: Neoenergia OPA + Eletrobras stake auction Oct 2022, ~R$42-46
    "ELPL3": "acquired",  # Eletropaulo: Enel OPA to cancel registration, notice ~Oct 2019
    "EEEL3": "acquired",  # CEEE-T: CPFL privatization auction Jul 2021 (57% premium) + tag-along OPA
    "EEEL4": "acquired",  # CEEE-T: CPFL privatization auction Jul 2021 (57% premium) + tag-along OPA
    "OMGE3": "acquired",  # Omega Geração: 100% stock-swap into Omega Energia (now Serena/MEGA3)
    "IGTA3": "acquired",  # Iguatemi: 100% stock-swap into Jereissati's Iguatemi S.A. (IGTI11)
    "CESP3": "acquired",  # CESP: incorporated into VTRM Energia, successor Auren Energia (AURE3)
    "CESP5": "acquired",  # CESP: incorporated into VTRM Energia, successor Auren Energia (AURE3)
}


def apply_manual_overrides(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Append MANUAL_TERMINAL_EVENTS for tickers build_terminal_events() couldn't
    resolve from the CVM registry alone. Registry-derived rows always win on overlap
    (though by construction there is none today -- every MANUAL_TERMINAL_EVENTS ticker
    was confirmed unresolved by the registry path first). delist_date is the ticker's
    own last positive-adj_close trade date in the panel (no CVM-sourced cancellation
    date exists for these), same anchor build_terminal_events() uses for payoff.
    """
    new_tickers = [t for t in MANUAL_TERMINAL_EVENTS if t not in set(events["ticker"])]
    if not new_tickers:
        return events

    last = (
        df[df["ticker"].isin(new_tickers) & (df["adj_close"] > 0)]
        .sort_values("trade_date")
        .groupby("ticker")
        .agg(delist_date=("trade_date", "last"), last_adj_close=("adj_close", "last"))
    )

    rows = [
        {
            "ticker": ticker,
            "delist_date": last.loc[ticker, "delist_date"],
            "event_type": event_type,
            "terminal_payoff": 0.0 if event_type == "failure" else last.loc[ticker, "last_adj_close"],
        }
        for ticker, event_type in MANUAL_TERMINAL_EVENTS.items()
        if ticker in new_tickers and ticker in last.index
    ]
    if not rows:
        return events
    return pd.concat([events, pd.DataFrame(rows)], ignore_index=True)


def find_rename_candidates(df: pd.DataFrame, delist_events: pd.DataFrame) -> pd.DataFrame:
    """Dead tickers whose CVM registry status is still ATIVO -- the company
    survived, this ticker code just didn't. Cross-referenced against every
    OTHER ticker sharing the same cnpj that's still trading in df, as a
    candidate for hand-adding to ticker_continuity.json.

    Report only -- NEVER auto-applied. continuity.py's rename/merger/
    keep_separate distinction is a judgement call this join cannot make.
    """
    dead = _dead_tickers(df)
    still_trading = set(df["ticker"].unique()) - dead

    orphaned = delist_events[
        delist_events["ticker"].isin(dead) & (delist_events["sit"] == "ATIVO")
    ][["ticker", "cnpj"]]
    candidates = delist_events[delist_events["ticker"].isin(still_trading)][["ticker", "cnpj"]]

    merged = orphaned.merge(candidates, on="cnpj", suffixes=("_old", "_new"))
    return merged.drop_duplicates().reset_index(drop=True)


def load_terminal_events(path=None) -> pd.DataFrame | None:
    """None when the (optional, separately-run) build step hasn't been run
    yet -- callers treat that as "no terminal events known", not an error."""
    if path is None:
        path = TERMINAL_EVENTS_PATH
    return pd.read_parquet(path) if path.exists() else None


def main():
    df = pd.read_parquet(OUTPUT_PATH, columns=["ticker", "trade_date", "adj_close"])
    delist_events = (
        pd.read_parquet(DELIST_EVENTS_PATH) if DELIST_EVENTS_PATH.exists()
        else build_delist_events()
    )

    events = build_terminal_events(df, delist_events)
    events = apply_manual_overrides(df, events)
    events.to_parquet(TERMINAL_EVENTS_PATH, index=False)
    n_failure = int((events["event_type"] == "failure").sum())
    n_acquired = int((events["event_type"] == "acquired").sum())
    print(f"terminal_events: {len(events)} dead tickers resolved "
          f"({n_failure} failure, {n_acquired} acquired) -> {TERMINAL_EVENTS_PATH}")

    candidates = find_rename_candidates(df, delist_events)
    if len(candidates):
        print(f"\n{len(candidates)} rename candidate(s) -- NOT applied, hand-verify and "
              f"add to data/raw/br/reference/ticker_continuity.json:")
        for _, row in candidates.iterrows():
            print(f"  {row['ticker_old']} -> {row['ticker_new']}  (cnpj {row['cnpj']})")


if __name__ == "__main__":
    main()
