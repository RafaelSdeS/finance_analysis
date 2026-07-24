"""quality_filters.py — coverage and filing-lag data quality gates.

Everything here either drops rows/tickers outright or attaches the
real-vs-statutory filing date fundamentals become visible on.
"""

import numpy as np
import pandas as pd

from .paths import FILING_DATES_PATH

# Tickers with fewer price rows than this carry no usable history (e.g. EGGY3 has 1 row)
MIN_PRICE_ROWS = 10

# Tickers whose raw price feed is broken beyond programmatic repair.
# Quarantined deliberately — document the reason, don't silently drop.
QUARANTINED_TICKERS = {
    "WDCN3": "raw close alternates between two price bases (~6x apart) "
             "hundreds of times 2021-2025; not a split, no factor to repair with",
    "CAMB4": "delisted/suspended 2019 (price data ends 2019-12-20); "
             "BolsAI still reports fundamentals through 2026-03-31 (stale data)",
    "LLIS3": "delisted/suspended 2023 (price data ends 2023-02-08); "
             "BolsAI still reports fundamentals through 2026-03-31 (stale data)",
    "CCTY3": "raw price feed is not real trading data for this company (cvm_code 27570, "
             "Belora RDVC City Desenvolvimento Imobiliario S.A.) -- confirmed identical to "
             "CCRO3/Motiva's (cvm_code 18821) orphaned post-rename stub across BOTH BolsAI's "
             "live API and yfinance independently (2026-07-14 audit); no reliable source found "
             "for this ticker's true price history",
}

# Tickers whose raw price file's EARLIEST rows are stale data from an
# unrelated, long-dead earlier holder of the same (recycled) B3 ticker code
# -- not real history of the CURRENT listing. Confirmed per-ticker via
# company_info/fundamentals cross-checks (not a blanket heuristic): a dataset-
# wide scan (2026-07-15) found 118 tickers with a similar >=2-year internal
# price-history gap, but only entries below have been individually verified
# enough to act on -- see ORPHAN_FRAGMENT_TICKERS.md for the rest, flagged
# but not fixed.
ORPHAN_PREFIX_TICKERS = {
    "BRDT3": {  # -> VBBR3 via ticker_continuity.json
        "drop_before": "2003-06-01",
        "reason": "6 sparse trades 2001-12-03..2003-01-07 (1-2 trades/session, round-number "
                  "volumes), then a 14.9-year silence before real IPO-era trading resumes "
                  "2017-12-15 (49M-share volume day). BRDT3/Vibra Energia (cnpj 34274233000102 "
                  "per company_info) was a wholly-owned Petrobras subsidiary with no independent "
                  "listing before its Dec-2017 IPO -- fundamentals for this ticker only start "
                  "2016-12-31, confirming no real filings exist for the 2001-2003 window either. "
                  "These rows belong to an unrelated, long-dead earlier holder of the recycled "
                  "BRDT3 ticker code (confirmed 2026-07-15).",
    },
}


def drop_orphan_prefix_rows(prices):
    """Drop the hand-verified orphan-prefix rows in ORPHAN_PREFIX_TICKERS.

    Must run before apply_ticker_continuity() so the garbage never reaches
    first-trade-date boundary computation, MIN_PRICE_ROWS counting, or any
    feature/rolling-window logic.
    """
    for ticker, info in ORPHAN_PREFIX_TICKERS.items():
        mask = (prices["ticker"] == ticker) & (prices["trade_date"] < pd.Timestamp(info["drop_before"]))
        n = int(mask.sum())
        if n:
            print(f"Dropping {n} orphan-prefix row(s) for {ticker} (before {info['drop_before']}): "
                  f"{info['reason']}")
            prices = prices[~mask]
    return prices

# Tickers with genuinely zero fundamental coverage everywhere (not delisted,
# not redundant with a sibling class), but excludable for a documented reason
# other than "missing data" — e.g. an ETF, not an operating company.
KNOWN_NO_FUNDAMENTALS = {
    "BOVA11": "benchmark ETF (IBOV proxy), not an operating company — fundamentals not applicable",
}

# Ticker stopped trading this many days before the dataset's last observed
# date is treated as delisted/renamed rather than a live coverage gap.
STALE_TICKER_DAYS = 730


def _ticker_root(ticker: str) -> str:
    """Strip the trailing share-class digits (PETR4 -> PETR, ALUP11 -> ALUP)."""
    return ticker.rstrip("0123456789")


def filter_tickers_with_no_fundamentals(prices, fundamentals):
    """Drop any ticker from prices that has zero fundamental rows.

    Sparse fundamentals (e.g. PETR4 only goes back to 2010) are fine —
    the model handles NaNs in early rows. Zero fundamentals means we have
    no quality signal at all, which is not acceptable for this agent.

    Returns (prices, dropped_report): dropped_report is a structured record
    of every excluded ticker and why, threaded through to write_manifest()
    so this source of universe/survivorship bias is queryable from the build
    manifest instead of only ever existing as stdout log lines (2026-07-23
    audit finding -- dropped tickers, esp. delisted/failed companies with no
    fundamentals coverage, are exactly the survivorship-relevant ones).
    """

    print()
    print("=" * 80)
    print("FUNDAMENTAL COVERAGE CHECK")
    print("=" * 80)

    dropped_report = {
        "quarantined": {}, "known_non_company": {}, "delisted_stale": [],
        "redundant_sibling": {}, "gap_unexplained": [], "too_short_history": [],
    }

    quarantined = set(QUARANTINED_TICKERS) & set(prices["ticker"].unique())
    for t in sorted(quarantined):
        print(f"QUARANTINED {t}: {QUARANTINED_TICKERS[t]}")
        dropped_report["quarantined"][t] = QUARANTINED_TICKERS[t]
    prices = prices[~prices["ticker"].isin(quarantined)]

    tickers_with_prices = set(prices["ticker"].unique())
    tickers_with_fundamentals = set(fundamentals["ticker"].unique())

    missing = tickers_with_prices - tickers_with_fundamentals
    covered = tickers_with_prices & tickers_with_fundamentals

    if missing:
        last_trade = prices.groupby("ticker")["trade_date"].max()
        dataset_max_date = prices["trade_date"].max()
        covered_roots = {_ticker_root(t): t for t in covered}

        known, dead, redundant, gap = [], [], [], []
        for t in sorted(missing):
            if t in KNOWN_NO_FUNDAMENTALS:
                known.append(t)
            elif (dataset_max_date - last_trade[t]).days > STALE_TICKER_DAYS:
                dead.append(t)
            elif _ticker_root(t) in covered_roots:
                redundant.append((t, covered_roots[_ticker_root(t)]))
            else:
                gap.append(t)

        print(f"EXCLUDED (no fundamentals): {len(missing)} tickers — safe to exclude "
              f"unless flagged as a GAP below")
        if known:
            print(f"  known non-company ({len(known)}):")
            for t in known:
                print(f"    {t}: {KNOWN_NO_FUNDAMENTALS[t]}")
                dropped_report["known_non_company"][t] = KNOWN_NO_FUNDAMENTALS[t]
        if dead:
            print(f"  delisted/renamed, last traded >{STALE_TICKER_DAYS}d before dataset end "
                  f"({len(dead)}): {dead}")
            dropped_report["delisted_stale"] = dead
        if redundant:
            print(f"  redundant, company already covered via sibling ticker ({len(redundant)}):")
            for t, sib in redundant:
                print(f"    {t} -> {sib}")
                dropped_report["redundant_sibling"][t] = sib
        if gap:
            print(f"  ⚠ GAP — recent price data but zero fundamentals anywhere, "
                  f"needs investigation ({len(gap)}): {gap}")
            dropped_report["gap_unexplained"] = gap
        prices = prices[prices["ticker"].isin(covered)]

    # Drop tickers with almost no price history — nothing to learn from them
    row_counts = prices.groupby("ticker").size()
    too_short = set(row_counts[row_counts < MIN_PRICE_ROWS].index)
    if too_short:
        print(f"EXCLUDED (< {MIN_PRICE_ROWS} price rows): {sorted(too_short)}")
        prices = prices[~prices["ticker"].isin(too_short)]
        covered -= too_short
        dropped_report["too_short_history"] = sorted(too_short)

    print(f"Tickers retained: {sorted(covered)}")
    print(f"Price rows after filter: {len(prices)}")

    return prices, dropped_report


# `reference_date` from BolsAI is the fiscal quarter-end, not the real filing/
# disclosure date (verified: BolsAI's /fundamentals history has no filing-date
# field at all). The real publication date (CVM's DT_RECEB) is collected by
# src/data_collection/cvm/filing_dates.py and attached per quarter; these statutory
# deadlines (ITR 45d, DFP ~90d) are the fallback for quarters missing from the
# CVM register.
FILING_LAG_DAYS_QUARTERLY = 45
FILING_LAG_DAYS_ANNUAL = 90

# Data quality: drop fundamentals filed > 180 days late (too uncertain for agent)
# Analysis: 2.1% of filings have lag > 180d; mostly historical (2010-2017 CVM backfill)
MAX_ACCEPTABLE_FILING_LAG_DAYS = 180


def _statutory_available_date(reference_dates):
    """Fallback availability: quarter-end + statutory CVM filing deadline."""
    lag_days = np.where(
        reference_dates.dt.month == 12,
        FILING_LAG_DAYS_ANNUAL,
        FILING_LAG_DAYS_QUARTERLY,
    )
    return reference_dates + pd.to_timedelta(lag_days, unit="D")


def attach_filing_dates(fundamentals, company_info):
    """Set fundamentals_available_date = real CVM receipt date (DT_RECEB),
    statutory deadline where the quarter is missing from the CVM register.

    Measured on Q1-2025: median real lag is 44 days (the statutory buffer is
    well calibrated), but 8.6% of companies file late — up to 443 days — so
    the fixed buffer alone leaks their fundamentals before publication.
    """
    print()
    print("=" * 80)
    print("ATTACHING FILING DATES (CVM DT_RECEB)")
    print("=" * 80)

    fundamentals = fundamentals.copy()

    if not FILING_DATES_PATH.exists():
        print("filing_dates.parquet missing — statutory deadlines only "
              "(run: python -m src.data_collection.cvm_statements --step filing_dates)")
        fundamentals["fundamentals_available_date"] = (
            _statutory_available_date(fundamentals["reference_date"])
        )
        return fundamentals

    # a quarter can appear in both ITR and DFP registers — one row per
    # (cnpj, quarter), earliest receipt, or the merge would duplicate rows
    filings = (
        pd.read_parquet(FILING_DATES_PATH)
        .groupby(["cnpj", "reference_date"], as_index=False)["received_date"].min()
        .rename(columns={"cnpj": "_cnpj"})
    )
    cnpj_map = company_info.assign(
        cnpj=company_info["cnpj"].str.replace(r"\D", "", regex=True)
    ).set_index("ticker")["cnpj"]

    fundamentals["_cnpj"] = fundamentals["ticker"].map(cnpj_map)
    fundamentals = fundamentals.merge(filings, on=["_cnpj", "reference_date"], how="left")

    # a filing can't precede its own quarter-end; treat such rows as unknown
    bad = fundamentals["received_date"] < fundamentals["reference_date"]
    fundamentals.loc[bad, "received_date"] = pd.NaT

    n_real = int(fundamentals["received_date"].notna().sum())
    print(f"Real filing dates: {n_real}/{len(fundamentals)} quarters "
          f"({100 * n_real / len(fundamentals):.1f}%), statutory fallback for the rest")

    fundamentals["fundamentals_available_date"] = fundamentals["received_date"].fillna(
        _statutory_available_date(fundamentals["reference_date"])
    )

    # Calculate filing lag for quality filtering
    fundamentals["filing_lag_days"] = (fundamentals["received_date"] - fundamentals["reference_date"]).dt.days

    return fundamentals.drop(columns=["_cnpj", "received_date"])


def filter_excessive_filing_lag(fundamentals, max_lag_days=MAX_ACCEPTABLE_FILING_LAG_DAYS):
    """Drop fundamentals filed more than max_lag_days after quarter-end.

    Rationale: if filed 180+ days late, too uncertain what was known at decision time.
    Typical delays: median 45d, 99th %ile 299d. Extreme outliers (>200d) are mostly
    2010-2017 historical data; current delays (2024-2026) max 536d but rare.

    Analysis: removes 2.1% of rows (883 filings with lag 180-365d, 58 with lag >365d).
    """
    print()
    print("=" * 80)
    print(f"FILTERING EXCESSIVE FILING LAGS (> {max_lag_days} days)")
    print("=" * 80)

    before = len(fundamentals)

    # Some rows may have NaN filing_lag_days (statutory fallback, no real CVM date)
    # Keep those (they're not excessive lags, just unknown real delays)
    excessive = fundamentals["filing_lag_days"].notna() & (fundamentals["filing_lag_days"] > max_lag_days)
    n_dropped = excessive.sum()

    if n_dropped > 0:
        # Capture stats before filtering
        dropped_lags = fundamentals.loc[excessive, "filing_lag_days"]
        lag_min = dropped_lags.min()
        lag_max = dropped_lags.max()
        pct_dropped = 100 * n_dropped / before

        fundamentals = fundamentals[~excessive].copy()
        print(f"Dropped {n_dropped} rows ({pct_dropped:.1f}%) with filing lag > {max_lag_days} days")
        print(f"  Lag range of dropped rows: {lag_min:.0f}–{lag_max:.0f} days")
    else:
        print(f"No rows exceeded {max_lag_days}-day lag threshold")

    print(f"Rows retained: {len(fundamentals)}")

    return fundamentals
