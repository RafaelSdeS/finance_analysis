"""Vendor-neutral fundamentals algebra, shared by every source.

Moved out of yf_collectors.py (docs/DATA_LAYER_ORGANIZATION_PLAN.md §O1): `compute_ratios`
is pure algebra over a dict of line items, not yfinance-specific — SEC's XBRL/EX-27/Item-6
tiers (sec/companyfacts.py, sec/fds.py, sec/fundamentals.py, sec/selected_financial_data.py,
sec/tenq.py) all call it with `unit_scale=1`, and the BolsAI BR collector (br/collectors.py)
uses `FUND_FULL_COLS` for its on-disk schema. Neither wants a yfinance import to get there.

Sequenced ahead of DATA_LAYER_CORRECTNESS_PLAN.md §1 (which edits `compute_ratios`'
`unit_scale` handling) so that step doesn't also have to update six import sites.
"""

import numpy as np

K = 1000  # BolsAI fundamentals are stored in BRL thousands; yfinance reports full BRL.

# Full on-disk fundamentals schema (validate.FUND_COLS only lists the required subset).
FUND_FULL_COLS = [
    "ticker", "reference_date", "close_price", "shares_outstanding", "market_cap",
    "pl", "pvp", "ev_ebitda", "ev_ebit", "p_ebitda", "p_ebit", "p_sr", "lpa", "vpa",
    "gross_margin", "net_margin", "ebitda_margin", "ebit_margin", "roe", "roa", "roic",
    "ebit_over_assets", "asset_turnover", "p_assets", "current_ratio", "debt_equity",
    "net_debt_equity", "net_debt_ebitda", "net_debt_ebit", "cagr_revenue_5y", "cagr_earnings_5y",
    "net_income", "equity", "net_revenue", "total_debt", "ebitda", "ebit", "net_debt",
    "cash", "total_assets", "current_assets", "current_liabilities",
]


def compute_ratios(r: dict, unit_scale: float = K) -> dict:
    """Recompute BolsAI-equivalent ratios from raw fundamentals figures.
    Formulas for market_cap/lpa/vpa/pl/pvp/roe/roa/net_margin/ebitda_margin/
    net_debt/debt_equity/ev_ebitda are the exact ones already verified at 5%
    tolerance against live BolsAI data in tests/data_collection/validate_vs_yfinance.py's
    check_internal_consistency(). The rest extend the same algebraic pattern.
    All divisions propagate NaN naturally on missing/zero inputs — no extra guards needed.

    `unit_scale` converts the "thousands"-denominated fields (net_income, equity,
    etc. — BolsAI's storage convention, see module-level `K`) up to market_cap's
    full-currency-unit scale before combining them. Defaults to `K` for the BR/
    yfinance callers below; SEC EDGAR's XBRL figures are already full-dollar
    (verified 2026-07-28: AAPL NetIncomeLoss reported as 4,834,000,000, not
    4,834,000), so sec/ratios.py passes unit_scale=1 — same formulas, no
    thousands conversion needed. Public (not `_compute_ratios`) because it's
    now shared across sources, not yfinance-internal.
    """
    # np.float64 (not plain float) so x/0 -> inf/nan instead of ZeroDivisionError.
    g = lambda key: np.float64(r.get(key, np.nan))
    net_income, equity, net_revenue = g("net_income"), g("equity"), g("net_revenue")
    total_assets, total_debt, ebitda, ebit = g("total_assets"), g("total_debt"), g("ebitda"), g("ebit")
    cash, current_assets, current_liabilities = g("cash"), g("current_assets"), g("current_liabilities")
    shares, close_price = g("shares_outstanding"), g("close_price")
    cost_of_revenue = g("cost_of_revenue")

    market_cap = close_price * shares
    net_debt = total_debt - cash
    ev = market_cap + net_debt * unit_scale

    # Zero denominators (pre-revenue/holding-company quarters) are expected and
    # handled below by the inf->NaN cleanup, not a bug — silence numpy's warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "market_cap": market_cap,
            "lpa": net_income * unit_scale / shares,
            "vpa": equity * unit_scale / shares,
            "pl": market_cap / (net_income * unit_scale),
            "pvp": market_cap / (equity * unit_scale),
            "roe": net_income / equity * 100,
            "roa": net_income / total_assets * 100,
            "net_margin": net_income / net_revenue * 100,
            "ebitda_margin": ebitda / net_revenue * 100,
            "net_debt": net_debt,
            "debt_equity": total_debt / equity,
            "ev_ebitda": ev / (ebitda * unit_scale),
            "ev_ebit": ev / (ebit * unit_scale),
            "p_ebitda": market_cap / (ebitda * unit_scale),
            "p_ebit": market_cap / (ebit * unit_scale),
            "p_sr": market_cap / (net_revenue * unit_scale),
            "ebit_margin": ebit / net_revenue * 100,
            "ebit_over_assets": ebit / total_assets * 100,
            "asset_turnover": net_revenue / total_assets,
            "p_assets": market_cap / (total_assets * unit_scale),
            "current_ratio": current_assets / current_liabilities,
            "net_debt_equity": net_debt / equity,
            "net_debt_ebitda": net_debt / ebitda,
            "net_debt_ebit": net_debt / ebit,
            # ponytail: approximation — no tax-effected NOPAT available from yfinance.
            "roic": ebit / (total_debt + equity - cash) * 100,
            "gross_margin": (net_revenue - cost_of_revenue) / net_revenue * 100,
            # filled later by cagr_handler.fill_cagr_columns() over the combined
            # historical series — yfinance alone has ~1.5y depth, not enough for 5y CAGR.
            "cagr_revenue_5y": np.nan,
            "cagr_earnings_5y": np.nan,
        }
    # nonzero/0 divisions land here as inf, not NaN (only 0/0 propagates NaN
    # naturally) — clean at the source so raw parquet never stores literal inf.
    return {k: (np.nan if isinstance(v, float | np.floating) and np.isinf(v) else v)
            for k, v in out.items()}
