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
#   - Confirmed renames, now spliced into ticker_continuity.json (resolved 2026-08-21):
#     BBRK3->NEXP3, INPR3->VIVR3, FJTA3->TASA3, BMGB11->BMGB4, GPCP4->DEXP4, STBP11->STBP3.
#   - CELP5/CELP6/CELP7->EQPA5/6/7: deliberately left unspliced (see CELP3->EQPA3's own
#     note in ticker_continuity.json) -- preferred classes traded in parallel with CELP3
#     for years, no surviving preferred successor with fundamentals coverage.
#   - LIQO3->ATMP3->CTAX3: fully identified 2026-08-21 (Contax Participações -> Liq
#     Participações [2018-03] -> Atma Participações [2020-03] -> Contax Participações
#     again [2024-06], the SAME "CTAX3" code reused for two non-adjacent eras of the
#     same entity). Genuinely too dangerous to splice: our CTAX3.parquet holds the
#     FIRST (2005-2018) era; ATMP3 was never collected at all (2020-2024 gap); a naive
#     LIQO3->ATMP3->CTAX3 chain would resolve `first_trade.get("CTAX3")` to the OLD
#     2005 era and silently delete the entire LIQO3 leg (continuity.py's boundary logic
#     assumes each ticker code is used exactly once). Left as a genuine, documented
#     collection gap + a `continuity.py` limitation, not something to force through here.
#   - CNTO3->SBFG3: identity confirmed (Centauro renamed Grupo SBF, 2021-03-31), but our
#     SBFG3.parquet and CNTO3.parquet share the exact same first trade date (2019-04-17,
#     same opening close R$12.30) -- the price vendor evidently backfilled SBFG3's full
#     history under the current ticker code rather than starting it at the rename
#     boundary. Splicing would double-count already-duplicated history. No fix applied.
#   - NINJ3->ARND3 (GetNinjas renamed Arandu Investimentos): same vendor-backfill
#     pattern as CNTO3/SBFG3 -- ARND3.parquet starts 2021-05-18, one day into NINJ3's
#     own 2021-05-17 IPO, not at NINJ3's real 2025-01-09 death. Not spliced.
#   - AZUL4: real, complex, ongoing debt-to-equity judicial-recovery restructuring
#     (AZUL4 -> AZUL54 -> AZUL53 -> AZUL3 through 2025-2026), with public reporting
#     suggesting original shareholders were heavily diluted. No confirmed payoff or
#     dilution ratio found with reasonable search effort; left genuinely unresolved
#     rather than guessing between "failure" and "acquired".
#   - Collection gaps: 13 tickers confirmed STILL ACTIVELY TRADING today, our own
#     pipeline just stopped collecting them (ATIVO in company_info.parquet, so this is
#     NOT the "missing from company_info" mechanism AMER3/LIGT3 hit -- a distinct
#     collection-path gap): AESB3 (AES Brasil), AGXY3 (AgroGalaxy), AHEB5/AHEB6 (São
#     Paulo Turismo), BOBR4 (Bombril, mid judicial recovery but still trading), CEED4
#     (CEEE-D), CSRN3/CSRN5/CSRN6 (Cosern), CTSA3/CTSA4 (Cia. Tecidos Santanense),
#     MTSA3 (Metisa), YBRA4 (Ybyra Capital). Confirmed 2026-08-16: AMER3, LIGT3, and
#     ~11 others of the same "missing from company_info" shape.
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
    # Added 2026-08-21 (DATA_LAYER_CORRECTNESS_PLAN.md §2b/§6 triage, web-verified):
    "BRFS3": "acquired",  # BRF fully incorporated into Marfrig (MBRF3), completed 2025-09-22, each
                           # BRFS3 share -> 0.8521 MRFG3/MBRF3 shares (see ticker_continuity.json's
                           # keep_separate entry -- MBRF3 pre-existed independently since 2007, so
                           # not spliced). KNOWN BIAS, larger than the other stock-swap cases here:
                           # BRFS3's own last observed price (R$17.95) is ~29% ABOVE the true
                           # post-merger value (0.8521 x MBRF3's ~R$16 = ~R$13.9) -- unlike
                           # CIEL3/KRSA3 below, whose last price closely tracks the real cash payout.
                           # Not caught by _dead_tickers()'s STALE_TICKER_DAYS heuristic yet (last
                           # trade is <365 days before the current panel end) but the merger is a
                           # confirmed, completed, real-world fact, so listed here regardless.
    "CIEL3": "acquired",  # Cielo: Bradesco+Banco do Brasil (EloPar) going-private OPA, R$5.82/share
                           # cash, concluded Aug 2024. Last observed adj_close (R$5.83) matches the
                           # confirmed cash price almost exactly.
    "KRSA3": "acquired",  # Kora Saúde: Viso/HIG-controlled OPA, R$8.80/share cash, CVM-approved
                           # Feb 2025, converting registry A->B and exiting Novo Mercado. Last
                           # observed adj_close (R$8.87) matches the confirmed cash price closely.
    "JPSA3": "acquired",  # Jereissati Participações: 100%-stock-swap merger into a reorganized
                           # Iguatemi S.A. (IGTI11/IGTI3), Nov 2021 -- the SAME transaction that
                           # IGTA3 above already covers from Iguatemi's side (every 7 JPSA3 ON ->
                           # 1 IGTI11 unit). Same "known approximation" treatment as IGTA3/OMGE3/
                           # CESP3/CESP5: true payoff is IGTI11's later share value, not a
                           # cash-confirmed figure, so not spliced into ticker_continuity.json.
    "JBSS3": "acquired",  # JBS: delisted from B3 2025-06-09 (last trade 2025-06-06 @ R$39.03) as
                           # part of the NYSE dual-listing restructuring -- 2 ordinary JBSS3 shares
                           # converted into 1 Level II BDR (JBSS32), which we don't track as a
                           # normal equity ticker. Real, clean, value-preserving conversion (not a
                           # distress event), so "acquired" off the last observed price, same
                           # precedent as the stock-swap mergers above.
    "GPCP3": "acquired",  # GPC Participações (common shares) renamed Dexxos Participações in the
                           # same 2021-06-08 event as GPCP4->DEXP4 (see ticker_continuity.json), but
                           # its natural successor code DEXP3 is unsafe to splice -- that ticker
                           # code was already in use by an unrelated entity back to 2000, ~21 years
                           # before Dexxos reused it. Real, ongoing company, not a distress event;
                           # "acquired" off GPCP3's own last observed price as an approximation.
    "FJTA4": "acquired",  # Forjas Taurus preferred shares renamed Taurus Armas (TASA4) in the same
                           # 2019-11-12 event as FJTA3->TASA3 (see ticker_continuity.json), but its
                           # natural successor code TASA4 is unsafe to splice -- that ticker code
                           # was already in use by an unrelated entity back to 2000, ~19 years
                           # before Taurus reused it. Real, ongoing company, not a distress event;
                           # "acquired" off FJTA4's own last observed price as an approximation.
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
