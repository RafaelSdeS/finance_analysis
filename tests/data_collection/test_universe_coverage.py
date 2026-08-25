#!/usr/bin/env python3
"""
Whole-universe coverage floors for data/raw/br/ -- pins DEFECT-2, DEFECT-3,
and DEFECT-4 (docs/BR_DATA_RECONSTRUCTION_PLAN.md §2), found auditing the
2026-08-23 recollection that regressed BR prices from 1,328 to 383 files
and fundamentals from 612 to 382, with zero test catching either drop.

Needs real data/raw/br/ on disk (git-tracked) -- DATA group, not FAST.

NO_YF_COVERAGE (S2): tickers CVM marks ATIVO but yfinance has never served
(genuinely delisted under this code, e.g. CIEL3/AZUL4, not renamed -- a
rename is recoverable via ticker_continuity.json instead). Populate by hand
after collect_delisted.py confirms yfinance truly has no coverage; an entry
here is a documented, deliberate gap, not a silent one.

Run from project root: python tests/data_collection/test_universe_coverage.py
or: pytest tests/data_collection/test_universe_coverage.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection import config

NO_YF_COVERAGE: set[str] = set()

# S6: hard floors, bumped deliberately after a verified real collection --
# never bumped to make a red test green without checking why.
MIN_PRICE_FILES = 1200
MIN_FUNDAMENTALS_FILES = 550
MIN_DIVIDENDS_FILES = 300

# S4: a survivor-only panel means the delisted-recovery step (collect_delisted.py
# / cvm_statements.py) was never run -- see DEFECT-3.
MIN_STALE_TICKERS = 700
STALE_DAYS = 90


def _price_tickers() -> set[str]:
    return {p.stem for p in config.PRICES_DIR.glob("*.parquet")}


def _fundamentals_tickers() -> set[str]:
    return {p.stem for p in config.FUND_DIR.glob("*.parquet")}


def _crosswalk_tickers() -> set[str]:
    path = config.CVM_DIR / "fca_crosswalk.parquet"
    if not path.exists():
        pytest.skip("no fca_crosswalk.parquet on disk -- run cvm_statements --step crosswalk first")
    return set(pd.read_parquet(path)["ticker"])


def test_every_active_ticker_has_prices():
    """S2. DEFECT-3: an ATIVO ticker with no price file is either uncollected
    (the bug) or a documented dead-code exception (NO_YF_COVERAGE)."""
    ci = pd.read_parquet(config.COMPANY_INFO_PATH)
    active = set(ci.loc[ci.status == "ATIVO", "ticker"].dropna())
    have = _price_tickers()

    missing = active - have - NO_YF_COVERAGE
    assert not missing, f"{len(missing)} ATIVO ticker(s) with no price file and no " \
        f"documented reason: {sorted(missing)[:20]}"


def test_units_are_collected():
    """S3. DEFECT-1b's end-to-end check: every ATIVO suffix-11 unit
    (TAEE11, SANB11, KLBN11, ...) has a price file, not just BOVA11/BPAC11."""
    ci = pd.read_parquet(config.COMPANY_INFO_PATH)
    active_units = set(ci.loc[(ci.status == "ATIVO") & ci.ticker.str.endswith("11"), "ticker"])
    have = _price_tickers()

    missing = active_units - have - NO_YF_COVERAGE
    assert not missing, f"{len(missing)} ATIVO unit(s) missing a price file: {sorted(missing)}"


def test_panel_contains_dead_tickers():
    """S4. DEFECT-3: a panel where almost every price file is fresh is
    survivor-only -- the delisted-recovery step was never run."""
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=STALE_DAYS)
    stale = 0
    for f in config.PRICES_DIR.glob("*.parquet"):
        last = pd.read_parquet(f, columns=["trade_date"])["trade_date"].max()
        if last < cutoff:
            stale += 1

    assert stale >= MIN_STALE_TICKERS, \
        f"only {stale} tickers with a price file >{STALE_DAYS}d stale " \
        f"(need >={MIN_STALE_TICKERS}) -- panel looks survivor-only"


def test_crosswalk_tickers_have_fundamentals():
    """S5. DEFECT-2: pipeline.py scopes the fundamentals stage to `active`
    (ATIVO only, via cvm_ratios.collect_fundamentals_cvm(active, mode)),
    stranding every delisted ticker CVM's crosswalk can otherwise resolve --
    independent of whether a price file exists for it. CVM statements are
    CNPJ-keyed and do cover delisted companies; the source is fine, the
    scoping is the bug. The real fix is a `rebuild=True` full-crosswalk
    pass (cvm.ratios.build_fundamentals(tickers=None, rebuild=True)), not
    gating on the prices stage's own output."""
    xwalk = _crosswalk_tickers()
    have_fund = _fundamentals_tickers()

    missing = xwalk - have_fund
    rate = len(missing) / len(xwalk) if xwalk else 0.0
    assert rate <= 0.10, \
        f"{len(missing)}/{len(xwalk)} ({rate:.1%}) crosswalk tickers have no " \
        f"fundamentals file: {sorted(missing)[:20]}"


def test_no_silent_universe_collapse():
    """S6. The 2026-08-23 recollection dropped 949 price files and 230
    fundamentals files and nothing failed. These floors are the guard --
    bump them by hand after a verified real collection, never to silence
    a real regression."""
    n_prices = len(_price_tickers())
    n_fund = len(_fundamentals_tickers())
    n_div = len(list(config.DIVIDENDS_DIR.glob("*.parquet")))

    assert n_prices >= MIN_PRICE_FILES, f"prices: {n_prices} files, floor is {MIN_PRICE_FILES}"
    assert n_fund >= MIN_FUNDAMENTALS_FILES, f"fundamentals: {n_fund} files, floor is {MIN_FUNDAMENTALS_FILES}"
    assert n_div >= MIN_DIVIDENDS_FILES, f"dividends: {n_div} files, floor is {MIN_DIVIDENDS_FILES}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
