"""
features.py — H-series monthly point-in-time feature/target panel
(MEDIUM_HORIZON_RESEARCH_PLAN.md Task H0/H1/H2 data layer).

Stage 2 (src/build_dataset/) already computes almost everything this needs:
technicals, fundamental ratios/trends, sector z-scores, momentum
decomposition, and a dividend yield that already sums JCP + Dividendo
(verified 2026-07-19 -- compute_dividend_features() sums value_per_share
across every `type`, no filter -- see MEDIUM_HORIZON_RESEARCH_PLAN.md sec
1.B). So this module is mostly SELECT, not COMPUTE: it resamples
ml_dataset.parquet onto the monthly decision grid, restricts to the
point-in-time active universe, and adds the two things that don't already
exist as columns: the forward-return targets (spine.py) and the
freshness-gated interaction features (sec 1.B's exp(-t/45) treatment).

`status` is never read (CLAUDE.md's feature-level lookahead trap: it's a
current-day snapshot joined onto every historical row).
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .paths import BOVA11_PATH, DATASET_PATH, MEMBERSHIP_PATH
from .spine import active_universe_by_date, build_forward_targets, monthly_decision_dates
from .stats import rank_normalize, sector_demean

# H1's candidate characteristics (MEDIUM_HORIZON_RESEARCH_PLAN.md sec 1.B /
# user addendum): value, quality, income, momentum (already market/sector-
# relative from Stage 2's cross_sectional.py), low-vol/liquidity, and
# own-history context. All already exist as ml_dataset.parquet columns --
# nothing here is computed from scratch.
CHARACTERISTIC_COLUMNS = (
    "earnings_yield", "book_to_market", "pl", "pvp", "ev_ebitda",
    "roe", "net_margin", "roe_trend_4q", "debt_equity",
    "div_yield_12m",
    "momentum_vs_market_12m", "momentum_vs_sector_12m",
    "volatility_60d", "turnover_ratio", "amihud_illiquidity",
    "pl_zhist_5y", "roe_zhist_5y",
)

# Already sector-relative BY CONSTRUCTION upstream (cross_sectional.py's
# momentum_vs_sector_*). H1 still screens their raw form for completeness,
# but a second sector-demean pass over an already sector-demeaned value is
# a no-op (modulo the sector-of-one NaN guard re-applying) -- so their
# "_sector_neutral" gate column is just themselves.
ALREADY_SECTOR_RELATIVE = frozenset({"momentum_vs_sector_12m"})

FRESHNESS_HALFLIFE_DAYS = 45.0  # fixed a priori (half a quarter), not tuned in H1

# TOP50_ML_READINESS_AUDIT.md's validated start date: fundamentals coverage ramps
# 0%->94% Jan-Apr 2011 (Brazil's DFP annual-filing deadline). Screening earlier than
# this pulls in a ~structural-NaN era for exactly the characteristics H1 tests --
# caught 2026-07-19: an unwindowed first run (2000-2026, no restriction) produced a
# suspicious 10/16-characteristic PASS; this was the missing bound.
WINDOW_START = "2011-04-01"


def days_since_filing(days_since_fundamental: pd.Series, filing_lag_days: pd.Series) -> pd.Series:
    """trade_date - fundamentals_available_date, derived from two existing
    columns rather than re-reading raw filing dates. days_since_fundamental
    (Stage 2 features.py) is trade_date - reference_date (the FISCAL
    period end); filing_lag_days is fundamentals_available_date -
    reference_date (the disclosure lag). Subtracting the second from the
    first cancels reference_date and leaves trade_date -
    fundamentals_available_date, i.e. genuine information age.

    Using days_since_fundamental directly here would be wrong: a company
    that files late already looks "stale" on day 1 of its data actually
    becoming knowable, which is the opposite of what an information-decay
    gate should measure."""
    return days_since_fundamental - filing_lag_days


def freshness_factor(age_days: pd.Series, tau: float = FRESHNESS_HALFLIFE_DAYS) -> pd.Series:
    """exp(-t/tau) -- multiplicative freshness gate (sec 1.B). A no-op
    wherever the pipeline ranks (Spearman IC and rank-normalized features
    are invariant to monotone transforms of t); has bite only in H2's
    cardinal freshness x characteristic interaction terms."""
    return np.exp(-age_days.clip(lower=0) / tau)


def _load_daily_prices() -> tuple:
    """(prices_wide, bench) for spine.build_forward_targets: adj_close per
    ticker and BOVA11, indexed by trade_date. This is the cheap, narrow
    read (3 columns) that also determines the trading calendar and monthly
    decision dates; build_monthly_panel's characteristic read is then
    filtered to just the resulting ~90 decision dates, not the full daily
    history."""
    table = pq.read_table(DATASET_PATH, columns=["ticker", "trade_date", "adj_close"])
    prices = table.to_pandas()
    prices_wide = prices.pivot(index="trade_date", columns="ticker", values="adj_close").sort_index()

    bova = pd.read_parquet(BOVA11_PATH, columns=["trade_date", "adj_close"])
    bench = bova.set_index("trade_date")["adj_close"].sort_index()
    bench = bench.reindex(prices_wide.index).ffill()
    return prices_wide, bench


def build_monthly_panel(k_horizons: tuple = (21, 63), window_start: str = WINDOW_START) -> pd.DataFrame:
    """The full H1/H2 monthly panel: one row per (decision_date, ticker),
    columns = CHARACTERISTIC_COLUMNS (raw + "_sector_neutral" variant),
    freshness_factor, sector, and one target_rank_k{k}/fwd_rel_return_k{k}
    pair per horizon in k_horizons. Restricted to decision dates >=
    window_start (see WINDOW_START's docstring for why this bound exists)."""
    prices_wide, bench = _load_daily_prices()
    decision_dates = monthly_decision_dates(prices_wide.index)
    decision_dates = decision_dates[decision_dates >= pd.Timestamp(window_start)]

    membership = pd.read_parquet(MEMBERSHIP_PATH)
    universe = active_universe_by_date(membership, decision_dates)

    extra_cols = ["days_since_fundamental", "filing_lag_days", "sector", "market_cap", "beta_1y"]
    load_cols = list(dict.fromkeys(
        ["ticker", "trade_date"] + list(CHARACTERISTIC_COLUMNS) + extra_cols
    ))
    table = pq.read_table(
        DATASET_PATH, columns=load_cols,
        filters=[("trade_date", "in", list(decision_dates))],
    )
    monthly = table.to_pandas().rename(columns={"trade_date": "decision_date"})

    panel = universe.merge(monthly, on=["decision_date", "ticker"], how="left")
    panel["age_days"] = days_since_filing(panel["days_since_fundamental"], panel["filing_lag_days"])
    panel["freshness_factor"] = freshness_factor(panel["age_days"])

    for col in CHARACTERISTIC_COLUMNS:
        if col in ALREADY_SECTOR_RELATIVE:
            panel[f"{col}_sector_neutral"] = panel[col]
        else:
            panel[f"{col}_sector_neutral"] = sector_demean(panel[col], panel["decision_date"], panel["sector"])

    for k in k_horizons:
        t = build_forward_targets(prices_wide, bench, decision_dates, k, universe)
        t = t.rename(columns={
            "fwd_rel_return": f"fwd_rel_return_k{k}",
            "fwd_rel_return_winsorized": f"fwd_rel_return_winsorized_k{k}",
            "target_rank": f"target_rank_k{k}",
            "n_universe_members": f"n_universe_members_k{k}",
        })
        panel = panel.merge(t, on=["decision_date", "ticker"], how="left")

    for k in k_horizons:
        # Sector-neutral IC needs BOTH sides demeaned, not just the characteristic --
        # correlating a purely stock-specific feature against a target that still
        # carries the sector tilt structurally attenuates the measured IC even when
        # the true within-sector relationship is strong (confirmed empirically: a
        # synthetic fixture with near-deterministic within-sector signal measured IC
        # ~0.33 instead of ~0.99 before this fix). Standard (Barra-style) sector
        # neutralization demeans both sides -- this is that, not an optional extra.
        fwd_col = f"fwd_rel_return_k{k}"
        panel[f"fwd_rel_return_sector_neutral_k{k}"] = sector_demean(
            panel[fwd_col], panel["decision_date"], panel["sector"])
        panel[f"target_rank_sector_neutral_k{k}"] = rank_normalize(
            panel[f"fwd_rel_return_sector_neutral_k{k}"], panel["decision_date"])

    return panel
