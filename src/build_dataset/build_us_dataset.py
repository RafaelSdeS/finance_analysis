"""
build_us_dataset.py — US analogue of build_ml_dataset.py.

Full rationale/measurements: docs/US_DATASET_BUILD_PLAN.md. This file only
holds what's genuinely NEW for the US side (sector mapping, macro, daily
valuation, universe gate) plus main()'s call order. Every other stage
(price features, dividend features, cross-sectional, cleaning, chunked
compute) is reused unchanged from the BR modules.

Deliberately SKIPPED vs. the BR pipeline (see plan §2/§7 for why):
  - repair_unadjusted_splits / apply_ticker_continuity (no US
    corporate_events.parquet / ticker_continuity.json to drive them)
  - attach_filing_dates / filter_excessive_filing_lag (BR/CVM-specific;
    US fundamentals already carry a real point-in-time
    fundamentals_available_date from SEC `filed`, and the 180-day BR gate
    would delete 27.7% of real US rows -- measured, see plan §D2)

Usage:
    python -m src.build_dataset.build_us_dataset
"""

import numpy as np
import pandas as pd

from .build_ml_dataset import compute_features_chunked
from .cross_sectional import BENCHMARK_COLS
from .features import compute_fundamental_features, compute_price_features, fill_missing_cagr
from .loaders import load_dividends, load_fundamentals, load_prices
from .manifest import write_manifest, write_split_config
from .merge import merge_dividends, merge_prices_and_fundamentals
from .paths import (
    US_COMPANY_INFO_PATH,
    US_DIVIDENDS_DIR,
    US_FUNDAMENTALS_DIR,
    US_MACRO_DIR,
    US_OUTPUT_PATH,
    US_PRICES_DIR,
    US_SPLIT_CONFIG_PATH,
)
from .quality_filters import drop_orphan_prefix_rows, filter_tickers_with_no_fundamentals

# SPY, the US equivalent of BOVA11 -- true market benchmark for beta_1y/
# momentum_vs_market_* (cross_sectional.py). Not an operating company, so it
# has no fundamentals and is captured before the coverage filter runs, same
# pattern as build_ml_dataset.py's BENCHMARK_TICKER.
BENCHMARK_TICKER = "SPY"

KNOWN_NO_FUNDAMENTALS_US = {
    "SPY": "benchmark ETF (S&P 500 proxy), not an operating company — fundamentals not applicable",
}

# Universe gate (plan §D1): computed per-ticker from its OWN full price
# history, so a formerly-liquid delisted name still qualifies. Measured
# 2026-07-31 against the real 8,143-ticker fundamentals-covered panel:
# 2,960 tickers / 15,419,040 rows at these thresholds (vs. 29.5M unfiltered).
# NOT point-in-time -- a lifetime-median statistic, same class of gate as
# build_top50_universe.py's BR selection; record this caveat in the manifest,
# don't present it as a point-in-time-clean universe.
MIN_PRICE_ROWS = 250
MIN_MEDIAN_CLOSE = 1.0
MIN_MEDIAN_DOLLAR_VOLUME = 1_000_000


# =============================================================================
# SECTOR (SIC DIVISION)
# =============================================================================

# Standard SIC-code division ranges (inclusive), the coarse ~10-group
# classification the SEC itself defines. sic_description (399 distinct
# values, company_info.parquet) is far too fine-grained for
# cross_sectional.py's per-(date, sector) z-scores/percentiles -- most groups
# would be sectors-of-one, which that module already NaNs out by design.
# Raw sic_description is kept as its own column for anyone who wants a finer
# grouping later; this is only the join-time `sector` column.
SIC_DIVISIONS = [
    (100, 999, "Agriculture, Forestry, Fishing"),
    (1000, 1499, "Mining"),
    (1500, 1799, "Construction"),
    (1800, 1999, "Not Used"),
    (2000, 3999, "Manufacturing"),
    (4000, 4999, "Transportation, Communications, Electric, Gas, Sanitary"),
    (5000, 5199, "Wholesale Trade"),
    (5200, 5999, "Retail Trade"),
    (6000, 6799, "Finance, Insurance, Real Estate"),
    (7000, 8999, "Services"),
    (9100, 9999, "Public Administration"),
]


def sic_to_sector(sic: pd.Series) -> pd.Series:
    """Map a numeric SIC code to its coarse division name; NaN/unmatched -> NaN."""
    sic = pd.to_numeric(sic, errors="coerce")
    sector = pd.Series(np.nan, index=sic.index, dtype=object)
    for lo, hi, name in SIC_DIVISIONS:
        sector = sector.where(~sic.between(lo, hi), name)
    return sector


# =============================================================================
# COMPANY INFO
# =============================================================================

def merge_company_info_us(df, company_info):
    """Join ticker -> sic/sic_description/sector. No cvm_code sibling-fill,
    no CVM crosswalk fallback, no status inference -- none of those concepts
    exist on the US side (company_siblings()'s cvm_code role has no direct
    analogue; out of scope for v1, see plan §4.3/§7)."""

    print()
    print("=" * 80)
    print("ADDING US COMPANY INFO")
    print("=" * 80)

    # `cik` already arrived on `df` from the per-filing fundamentals row
    # (loaders.load_fundamentals's US branch) -- that's the CIK the actual
    # filing was made under, which is what fundamentals_available_date
    # traces back to. company_info.parquet's `cik` is the SAME crosswalk
    # value, just re-derived from the ticker; dropped here rather than
    # merged in a second time (pandas would otherwise suffix both cik_x/cik_y).
    info = company_info.drop(columns=["cik"]).copy()
    info["sector"] = sic_to_sector(info["sic"])

    merged = df.merge(info, on="ticker", how="left")
    print(f"Company info merged for {merged['ticker'].nunique()} tickers "
          f"({merged['sector'].isna().sum()} rows with no sector, null/unmapped SIC)")
    return merged


# =============================================================================
# MACRO (RISK-FREE RATE + INFLATION)
# =============================================================================

# CPI is published ~2 weeks after month-end; shift its availability date the
# same way merge_macro() shifts BR's ipca, so the asof-merge below can never
# leak a future CPI reading into an earlier trade_date. This is a real
# no-lookahead requirement, not a nicety -- see plan §4.2.
CPI_PUBLICATION_LAG_DAYS = 15
TRADING_DAYS_PER_MONTH = 21

# selic_trend_20d's lookback (merge_macro.py's SELIC_TREND_LOOKBACK_DAYS) --
# duplicated here (not imported) since these are two independently-named
# constants in two independent US/BR merge functions, not one shared value.
RISK_FREE_TREND_LOOKBACK_DAYS = 20


def merge_macro_us(dataset):
    """US analogue of merge.merge_macro(). Emits columns literally named
    `selic`/`ipca` (not renamed) so compute_macro_features/compute_advanced_features
    ("earnings_yield_vs_selic") work with zero edits -- see plan §4.2 for the
    naming tradeoff. `cdi` has no US equivalent and is simply not emitted;
    nothing downstream reads it unconditionally.

    selic <- DTB3 (risk_free_3m.parquet), already daily annualized % -> converted
    to a daily-equivalent rate, same footing as BR's selic.
    ipca  <- CPIAUCSL (cpi_sa.parquet) index level -> month-over-month % change,
    availability-shifted by the real publication lag before the merge.
    """

    print()
    print("=" * 80)
    print("ADDING US MACRO SERIES (RISK-FREE RATE, INFLATION)")
    print("=" * 80)

    rf = pd.read_parquet(US_MACRO_DIR / "risk_free_3m.parquet")[["reference_date", "risk_free_3m"]]
    rf = rf.sort_values("reference_date").copy()
    rf["selic"] = ((1 + rf["risk_free_3m"] / 100) ** (1 / 252) - 1) * 100
    rf["selic_trend_20d"] = rf["selic"] - rf["selic"].shift(RISK_FREE_TREND_LOOKBACK_DAYS)
    rf = rf.drop(columns=["risk_free_3m"])

    cpi = pd.read_parquet(US_MACRO_DIR / "cpi_sa.parquet")[["reference_date", "cpi_sa"]]
    cpi = cpi.sort_values("reference_date").copy()
    cpi["ipca"] = cpi["cpi_sa"].pct_change(fill_method=None) * 100
    cpi["ipca_daily_equiv"] = (
        (1 + cpi["ipca"] / 100) ** (1 / TRADING_DAYS_PER_MONTH) - 1
    ) * 100
    # Availability shift -- CPI_sa for month M is released ~mid-month M+1;
    # without this, merge_asof would show a reading before it was ever public.
    cpi["reference_date"] = cpi["reference_date"] + pd.DateOffset(months=1) + pd.Timedelta(days=CPI_PUBLICATION_LAG_DAYS)
    cpi = cpi.drop(columns=["cpi_sa"])

    macro = rf.merge(cpi, on="reference_date", how="outer")
    macro = macro.sort_values("reference_date").ffill().rename(columns={"reference_date": "macro_date"})

    dataset = dataset.sort_values("trade_date")
    merged = pd.merge_asof(
        dataset, macro, left_on="trade_date", right_on="macro_date", direction="backward",
    )
    del dataset
    del merged["macro_date"]

    return merged.sort_values(["ticker", "trade_date"], ignore_index=True)


# =============================================================================
# DAILY VALUATION RATIOS (computed fresh, not re-anchored)
# =============================================================================

def _safe_div(numerator, denominator, min_abs=1e-6):
    denom = denominator.where(denominator.abs() > min_abs)
    return numerator / denom


def compute_valuation_daily_us(df):
    """US analogue of features.recompute_valuation_daily().

    BR's version RESCALES an existing vendor ratio (computed at the filing-date
    close) by close/close_price. That has no US equivalent to rescale: SEC
    fundamentals carry no price/close_price/market_cap at all (measured 0%
    coverage across market_cap/pl/pvp/ev_*/p_*/ebitda*, 200-ticker raw sample,
    2026-07-31) -- there was never a price at collection time to anchor to.
    So these are computed directly from the daily close, which also sidesteps
    BR's known mid-quarter-split skew entirely.

    ebitda itself is not collected (no D&A concept mapped yet -- plan §4.4,
    Phase E), so ebitda_margin/ev_ebitda/p_ebitda/net_debt_ebitda and their
    *_zhist_5y variants stay NaN in this build; not fixed here.
    """

    print()
    print("=" * 80)
    print("COMPUTING US DAILY VALUATION RATIOS")
    print("=" * 80)

    df["market_cap"] = df["close"] * df["shares_outstanding"]

    df["pl"] = _safe_div(df["market_cap"], df["net_income"])
    df["pvp"] = _safe_div(df["market_cap"], df["equity"])
    df["p_sr"] = _safe_div(df["market_cap"], df["net_revenue"])
    df["p_assets"] = _safe_div(df["market_cap"], df["total_assets"])
    df["p_ebit"] = _safe_div(df["market_cap"], df["ebit"])
    df["book_to_market"] = _safe_div(df["equity"], df["market_cap"])

    ev = df["market_cap"] + df["net_debt"]
    df["ev_ebit"] = _safe_div(ev, df["ebit"])

    df["has_fundamentals"] = df["reference_date"].notna().astype(float)

    print(f"Daily valuation ratios computed for {len(df)} rows")
    return df


# =============================================================================
# UNIVERSE GATE
# =============================================================================

def _qualifying_tickers(stats, min_rows, min_median_close, min_median_dollar_volume):
    """stats: per-ticker DataFrame indexed by ticker with n/med_close/med_dv
    columns -- shared by build_universe_gate (in-memory) and
    build_universe_gate_from_files (per-file scan) so the two can never
    silently disagree on what "qualifies" means."""
    qualifies = (
        (stats["n"] >= min_rows)
        & (stats["med_close"] >= min_median_close)
        & (stats["med_dv"] >= min_median_dollar_volume)
    )
    return set(stats.index[qualifies])


def build_universe_gate(prices, min_rows=MIN_PRICE_ROWS, min_median_close=MIN_MEDIAN_CLOSE,
                         min_median_dollar_volume=MIN_MEDIAN_DOLLAR_VOLUME):
    """Quality/scale gate over the full US universe (plan §D1) -- a lifetime
    per-ticker statistic, NOT point-in-time (a formerly-liquid delisted name
    still qualifies on its own history, but you couldn't have known in 1995
    what a ticker's *lifetime* median volume would turn out to be). Same class
    of gate as build_top50_universe.py; point-in-time universe construction
    stays a separate downstream step.

    Takes an already-loaded `prices` DataFrame -- fine for tests and small
    universes, but see build_universe_gate_from_files for the real US build
    (loading all 9,593 tickers just to gate them down to ~2,960 is itself an
    OOM risk, docs/US_DATASET_BUILD_PLAN.md §8.0).

    Returns the set of qualifying tickers.
    """
    print()
    print("=" * 80)
    print("US UNIVERSE GATE")
    print("=" * 80)

    dollar_volume = prices["close"] * prices["volume"]
    g = prices.assign(_dv=dollar_volume).groupby("ticker")
    stats = g.agg(n=("close", "size"), med_close=("close", "median"), med_dv=("_dv", "median"))

    tickers = _qualifying_tickers(stats, min_rows, min_median_close, min_median_dollar_volume)
    print(f"Universe gate: {len(tickers)}/{len(stats)} tickers qualify "
          f"(n>={min_rows}, med_close>=${min_median_close}, med_dollar_volume>=${min_median_dollar_volume:,.0f})")
    return tickers


class _MergeBatcher:
    """Callable batch_fn for compute_features_chunked (build_ml_dataset.py),
    PLUS an explicit release() to drop its captured prices/fundamentals/
    company_info/dividends references once Pass 1 is done with it.

    A plain closure + `del batch_fn` inside compute_features_chunked is NOT
    enough: main() keeps its OWN reference to this same object bound in its
    frame for the entire (synchronous, nested) compute_features_chunked
    call -- a caller's frame doesn't go away just because the callee deletes
    its own copy of a reference, so refcounting alone never reaches 0 there
    (confirmed via a minimal repro before landing this). release() instead
    MUTATES this instance's own attributes to None -- visible through every
    reference to it, main()'s included -- which is what actually frees the
    tables before Pass 2/3, which never call batch_fn again. main() must
    ALSO drop its own separate `prices`/`fundamentals`/`company_info` names
    for this to work (mutating THIS object's attributes doesn't touch a
    caller's own separate variable pointing at the same DataFrame).
    """

    def __init__(self, prices, fundamentals, company_info, dividends):
        self.prices = prices
        self.fundamentals = fundamentals
        self.company_info = company_info
        self.dividends = dividends

    def __call__(self, batch_tickers):
        bt = set(batch_tickers)
        p = self.prices[self.prices["ticker"].isin(bt)]
        f = self.fundamentals[self.fundamentals["ticker"].isin(bt)]
        d = self.dividends[self.dividends["ticker"].isin(bt)]

        merged = merge_prices_and_fundamentals(p, f)
        merged = merge_company_info_us(merged, self.company_info)
        merged = merge_macro_us(merged)
        merged = merge_dividends(merged, d)
        return merged

    def release(self):
        self.prices = self.fundamentals = self.company_info = self.dividends = None


def make_merge_batch_fn(prices, fundamentals, company_info, dividends):
    """Returns a _MergeBatcher (see its docstring for release()) that does
    the 4 merges (prices+fundamentals, company_info, macro, dividends)
    scoped to just one ticker-batch at a time, instead of once over the
    full universe.

    Why: merge_prices_and_fundamentals's OUTPUT -- the daily panel with every
    fundamentals column forward-filled onto it -- is ~1,000 B/row at US
    fundamentals' width; at the full ~15.4M-row universe that alone is
    ~15GB, before company_info/macro/dividends are even joined. Measured
    directly: an 800/3,134-ticker slice already peaked at 5.9GB RSS through
    this merge (docs/US_DATASET_BUILD_PLAN.md §8.0.1) -- extrapolating
    linearly to the full universe lands past this machine's available RAM,
    which is exactly what OOM-killed the real run. None of the 4 merges are
    cross-sectional (unlike compute_cross_sectional_features/Pass 2, which
    genuinely needs the whole universe at once) -- each operates strictly
    per-ticker or via join, so scoping them to a batch changes nothing about
    correctness, only how much is resident at once.

    `prices`/`fundamentals`/`company_info`/`dividends` are the already-loaded
    (narrow) raw tables -- kept resident for Pass 1 only (measured ~5.5GB for
    prices+fundamentals alone; release() drops them before Pass 2/3), which
    is what makes this safe: only the wide MERGED product is ever bounded to
    one batch.
    """
    return _MergeBatcher(prices, fundamentals, company_info, dividends)


def build_universe_gate_from_files(dir, min_rows=MIN_PRICE_ROWS, min_median_close=MIN_MEDIAN_CLOSE,
                                    min_median_dollar_volume=MIN_MEDIAN_DOLLAR_VOLUME):
    """Same gate as build_universe_gate, computed from a per-file, column-
    projected scan (only `close`/`volume`) instead of an already-loaded
    `prices` DataFrame -- avoids ever holding the full 9,593-ticker/34M-row
    US universe resident just to filter it down to ~2,960 tickers
    (measured: ~16GB peak in load_prices alone vs. ~8GB available,
    docs/US_DATASET_BUILD_PLAN.md §8.0 Failure 1). Feed the result straight
    into load_prices(tickers=...) so only qualifying files are ever loaded.

    Returns the set of qualifying tickers.
    """
    print()
    print("=" * 80)
    print("US UNIVERSE GATE (per-file scan)")
    print("=" * 80)

    rows = []
    for file in sorted(dir.glob("*.parquet")):
        d = pd.read_parquet(file, columns=["close", "volume"])
        dv = d["close"] * d["volume"]
        rows.append({"ticker": file.stem, "n": len(d),
                      "med_close": d["close"].median(), "med_dv": dv.median()})
    stats = pd.DataFrame(rows).set_index("ticker")

    tickers = _qualifying_tickers(stats, min_rows, min_median_close, min_median_dollar_volume)
    print(f"Universe gate: {len(tickers)}/{len(stats)} tickers qualify "
          f"(n>={min_rows}, med_close>=${min_median_close}, med_dollar_volume>=${min_median_dollar_volume:,.0f})")
    return tickers


# =============================================================================
# MAIN
# =============================================================================

def main():

    US_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Gate BEFORE loading: computing the gate from an already-loaded `prices`
    # frame (build_universe_gate) would first require loading all 9,593
    # tickers / 34M rows just to filter them down to ~2,960 -- ~16GB peak
    # against ~8GB available, an OOM before a single row is written
    # (docs/US_DATASET_BUILD_PLAN.md §8.0 Failure 1). The per-file scan reads
    # only close/volume per ticker.
    universe = build_universe_gate_from_files(US_PRICES_DIR)
    prices = load_prices(dir=US_PRICES_DIR, tickers=universe | {BENCHMARK_TICKER})
    fundamentals = load_fundamentals(dir=US_FUNDAMENTALS_DIR, optimize_dtypes=True)
    prices = drop_orphan_prefix_rows(prices)  # no-op for US tickers, kept for parity

    # Capture SPY (market benchmark) before the fundamentals-coverage filter
    # drops it -- same reasoning as build_ml_dataset.main()'s BOVA11 capture.
    benchmark_prices = prices[prices["ticker"] == BENCHMARK_TICKER].copy()
    if benchmark_prices.empty:
        raise ValueError(f"{BENCHMARK_TICKER} not found in prices -- required as the "
                          f"market benchmark for beta_1y/momentum_vs_market_*")
    benchmark = compute_price_features(benchmark_prices)[BENCHMARK_COLS]

    prices, dropped_no_fundamentals = filter_tickers_with_no_fundamentals(
        prices, fundamentals, known_no_fundamentals=KNOWN_NO_FUNDAMENTALS_US
    )
    fundamentals = compute_fundamental_features(fundamentals)
    # anchor_month=None: US fiscal year ends vary by company (8.6% of tickers
    # have zero December-ending quarters, e.g. Agilent's Oct 31 FYE) with no
    # reliable per-ticker FYE column to anchor to instead -- see
    # cagr_handler.calc_annual_cagr's docstring.
    fundamentals = fill_missing_cagr(fundamentals, anchor_month=None)
    company_info = pd.read_parquet(US_COMPANY_INFO_PATH)
    dividends = load_dividends(dir=US_DIVIDENDS_DIR)

    # Merge per-batch (inside compute_features_chunked's existing Pass-1 loop)
    # instead of once over the full universe -- see make_merge_batch_fn's
    # docstring for why the full-universe merge OOMs at US scale even though
    # none of the 4 merges are cross-sectional. Only needed for Pass 1 (the
    # merge itself); Pass 2/3 never touch batch_fn again.
    batch_fn = make_merge_batch_fn(prices, fundamentals, company_info, dividends)
    tickers = sorted(prices["ticker"].unique())

    # Drop OUR OWN references to prices/fundamentals/company_info now.
    # compute_features_chunked is called synchronously below, so main()'s own
    # locals stay bound for the ENTIRE call (all 3 passes) regardless of what
    # happens inside it -- if we kept prices/fundamentals/company_info
    # (~3GB+) referenced here too, they'd sit resident through Pass 2/3 as
    # well, which don't need them at all. This alone isn't sufficient though:
    # batch_fn (a _MergeBatcher instance, make_merge_batch_fn) ALSO holds its
    # own references to the same tables, and batch_fn itself stays bound in
    # THIS frame for the whole call too -- see _MergeBatcher.release()'s
    # docstring for why compute_features_chunked calling `.release()` on it
    # after Pass 1 (mutating the instance in place) is what actually drops
    # the last reference, not a plain `del`. Both halves are required
    # together. This exact leak was a real 3rd OOM in this build
    # (docs/US_DATASET_BUILD_PLAN.md §8.0.2 follow-up): batch_fn/tickers
    # fixed Pass 1, but the tables it captured kept living through Pass 2/3
    # too, on top of the fix in §8.3.
    del prices, fundamentals, company_info

    compute_features_chunked(None, dividends, benchmark, US_OUTPUT_PATH,
                              valuation_fn=compute_valuation_daily_us,
                              tickers=tickers, batch_fn=batch_fn)
    del dividends

    print()
    print("=" * 80)
    print("WRITING US MANIFEST & SPLIT CONFIG")
    print("=" * 80)

    # Stream from disk column-by-column instead of a dense pd.read_parquet:
    # at US scale (15.4M rows x ~190 cols) that read-back alone is ~20GB,
    # comfortably over this machine's available RAM (docs/US_DATASET_BUILD_PLAN.md
    # §8.2). write_split_config only ever needs trade_date.
    manifest = write_manifest(
        dropped_no_fundamentals=dropped_no_fundamentals,
        output_path=US_OUTPUT_PATH,
        parquet_path=US_OUTPUT_PATH,
    )
    write_split_config(pd.read_parquet(US_OUTPUT_PATH, columns=["trade_date"]), path=US_SPLIT_CONFIG_PATH)

    print(f"Saved to: {US_OUTPUT_PATH}")
    print(f"Rows: {manifest['rows']}  Columns: {len(manifest['columns'])}")


if __name__ == "__main__":
    main()
