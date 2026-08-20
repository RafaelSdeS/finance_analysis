"""
validate.py — lightweight per-collector data quality gate (runs before write).

This is the *schema/sanity* gate, distinct from the cross-source check in
tests/data_collection/validate_vs_yfinance.py (the validation STAGE). Returns a
ValidationResult; collectors refuse to save on errors, log on warnings.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PRICE_COLS = ["ticker", "trade_date", "open", "high", "low", "close",
              "adj_open", "adj_high", "adj_low", "adj_close",
              "volume", "volume_adjusted", "traded_amount", "num_trades"]

FUND_COLS = ["ticker", "reference_date", "net_income", "equity", "net_revenue",
             "total_assets", "ebitda", "shares_outstanding", "market_cap"]

DIVIDEND_COLS = ["ticker", "ex_date", "payment_date", "type", "value_per_share", "adjusted"]

CORP_EVENT_COLS = ["ticker", "date", "type", "ratio_from", "ratio_to", "factor"]


@dataclass
class ValidationResult:
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def error(self, msg: str):
        self.errors.append(msg)
        self.passed = False

    def warn(self, msg: str):
        self.warnings.append(msg)


def _common(df: pd.DataFrame, date_col: str, required: list[str]) -> ValidationResult:
    r = ValidationResult()
    if df.empty:
        r.error("empty dataframe")
        return r
    missing = [c for c in required if c not in df.columns]
    if missing:
        r.error(f"missing columns: {missing}")
        return r
    future = df[df[date_col] > pd.Timestamp.now() + pd.Timedelta(days=2)]
    if not future.empty:
        r.error(f"{len(future)} rows with future {date_col}")
    # A duplicated date is never legitimate for the SAME ticker (one
    # bar/filing/dividend-record per date) -- measured 0 occurrences across
    # all of data/raw/br/{prices,fundamentals,dividends} (2026-08-16).
    # Scoped to (ticker, date_col), not date_col alone: corporate_events.parquet
    # is a single multi-ticker file where two different companies legitimately
    # share a date (measured 255 real cross-ticker collisions, 0 same-ticker) --
    # a bare date_col check would treat that as corruption and refuse the write.
    dup_subset = ["ticker", date_col] if "ticker" in df.columns else [date_col]
    dupes = df.duplicated(subset=dup_subset)
    if dupes.any():
        r.error(f"{dupes.sum()} duplicate {'/'.join(dup_subset)} values present")
    return r


def validate_prices(df: pd.DataFrame) -> ValidationResult:
    r = _common(df, "trade_date", PRICE_COLS)
    if not r.passed:
        return r
    # Raw OHLC NaN is the BOVA11/CAMB3 failure mode (2026-08-16 finding): every
    # other check here is a comparison, and NaN compares False in pandas, so a
    # NaN bar silently passed close<=0/bracket checks. Measured real BR data:
    # NaN here is ~0% (2 rows in 2.29M) -- a real defect, not a vendor norm.
    # adj_* is deliberately NOT included: ~0.01% NaN there is the documented
    # 2-decimal-precision underflow on deep-history microcaps (CLAUDE.md) --
    # an error here would refuse to (re-)collect known-quarantined tickers.
    # [§2a, DATA_LAYER_CORRECTNESS_PLAN.md, 2026-08-20] This file predates
    # Stage 2's adj_close_precision_degraded flag by construction (it's a raw
    # collector validator; the flag is computed later in features.py), so it
    # can never check that flag directly -- and until 2026-08-20 the flag
    # itself couldn't fire on NaN/zero adj_close either (near_floor requires
    # adj_close > 0), so "already flagged downstream" was false for exactly
    # the rows this check would have caught. The flag now covers isna()/<=0
    # too, so the claim holds again -- but that's a property of features.py,
    # not something this validator enforces or can see.
    raw_ohlc = ["open", "high", "low", "close"]
    nan_ohlc = df[raw_ohlc].isna().any(axis=1)
    if nan_ohlc.any():
        r.error(f"{nan_ohlc.sum()} rows with NaN in {raw_ohlc}")
    if (df["close"] <= 0).any():
        r.error(f"{(df['close'] <= 0).sum()} rows with close <= 0")
    if (df["volume"] < 0).any():
        r.error("negative volume present")
    for open_c, high_c, low_c, close_c in (
        ("open", "high", "low", "close"),
        ("adj_open", "adj_high", "adj_low", "adj_close"),
    ):
        non_positive = (df[[open_c, high_c, low_c, close_c]] <= 0).any(axis=1)
        if non_positive.any():
            r.error(f"{non_positive.sum()} rows with non-positive {open_c}/{high_c}/{low_c}/{close_c}")
        bad_hl = df[high_c] < df[low_c]
        if bad_hl.any():
            r.error(f"{bad_hl.sum()} rows with {high_c} < {low_c}")
        eps = 1e-6  # float noise from independent adj_* computations, not a real violation
        bracket_violation = (
            (df[open_c] < df[low_c] - eps) | (df[open_c] > df[high_c] + eps)
            | (df[close_c] < df[low_c] - eps) | (df[close_c] > df[high_c] + eps)
        )
        if bracket_violation.any():
            r.error(f"{bracket_violation.sum()} rows with {open_c}/{close_c} outside [{low_c}, {high_c}]")
    # daily gaps > 5 calendar days that aren't a weekend straddle → flag, don't fail
    gaps = df.sort_values("trade_date")["trade_date"].diff().dt.days
    if (gaps > 5).sum() > 0:
        r.warn(f"{(gaps > 5).sum()} gaps > 5 days (holidays/halts?)")
    return r


def validate_fundamentals(df: pd.DataFrame) -> ValidationResult:
    r = _common(df, "reference_date", FUND_COLS)
    if not r.passed:
        return r
    # CAGR nulls are expected in the first ~20 quarters (need 5y history) AND in
    # any quarter whose 5y-ago base earnings were negative (CAGR undefined).
    # Only flag if the LATE null rate is implausibly high (>50%) → possible data issue.
    if "cagr_earnings_5y" in df.columns:
        late = df.sort_values("reference_date").iloc[20:]
        if len(late):
            null_rate = late["cagr_earnings_5y"].isna().mean()
            if null_rate > 0.5:
                r.warn(f"cagr_earnings_5y null rate {null_rate:.0%} after q20 (negative-base years, or data issue)")
    return r


def validate_us_fundamentals(df: pd.DataFrame) -> ValidationResult:
    """Sanity gate for SEC fundamentals (combined xbrl/ex27/item6 tiers).

    collect_fundamentals_us writes df.to_parquet() directly, unlike every
    other collector, which all go through _merge_save()'s validate-then-write
    (found auditing the US pipeline, 2026-07-30 -- SEC fundamentals had never
    been through any automated schema/sanity gate). Row-level anomalies here
    are warned, not blocked: build_company_fundamentals() rebuilds a
    company's entire multi-decade history in one shot each run, so refusing
    the whole write over one bad historical row (BR's all-or-nothing gate,
    fine for an incremental batch) would cost far more good data than it
    protects. Schema differs from BR's FUND_COLS (keyed on `end`, no
    `ticker`/`reference_date` columns), so this doesn't reuse validate_fundamentals.
    """
    r = ValidationResult()
    if df.empty:
        r.error("empty dataframe")
        return r
    numeric = df.select_dtypes(include="number")
    if not numeric.empty and np.isinf(numeric.to_numpy(dtype="float64", na_value=0.0)).any():
        r.warn("Inf value(s) present")
    if "total_assets" in df.columns and (df["total_assets"] < 0).any():
        r.warn(f"{(df['total_assets'] < 0).sum()} row(s) with negative total_assets (accounting-impossible)")
    if "shares_outstanding" in df.columns and (df["shares_outstanding"] < 0).any():
        r.warn(f"{(df['shares_outstanding'] < 0).sum()} row(s) with negative shares_outstanding")
    if "shares_outstanding_rejected_outlier" in df.columns and df["shares_outstanding_rejected_outlier"].any():
        n = int(df["shares_outstanding_rejected_outlier"].sum())
        r.warn(f"{n} row(s) had an implausible shares_outstanding value rejected "
               f"(companyfacts.reject_sequential_outliers) -- now NaN, not a guessed value")
    if "end" in df.columns and df["end"].duplicated().any():
        r.warn(f"{df['end'].duplicated().sum()} duplicate 'end' period(s)")

    # Unit-invariant accounting identities -- hold regardless of whether a
    # filing was parsed in units/thousands/millions, so a violation means a
    # wrong row or inconsistent per-item scaling, not a mis-detected units
    # caption. Cheap (no thresholds to tune) and would have caught the
    # fundamentals._FLOORS cases (CVBF, BPOP) automatically. 2026-08-01 audit.
    def _identity_warn(mask, label):
        if mask.any():
            r.warn(f"{int(mask.sum())} row(s) with {label}")

    cols = df.columns
    if {"equity", "total_assets"} <= set(cols):
        _identity_warn(df["equity"] > df["total_assets"], "equity > total_assets")
    if {"cash", "total_assets"} <= set(cols):
        _identity_warn(df["cash"] > df["total_assets"], "cash > total_assets")
    if {"current_assets", "total_assets"} <= set(cols):
        _identity_warn(df["current_assets"] > df["total_assets"], "current_assets > total_assets")

    # period_months/flows_derived/flows_defined visibility -- see
    # docs/US_QUARTERLY_BACKFILL_PLAN.md. Warn-only: a missing/odd value here
    # is informative (e.g. an old build predating this schema, or a fiscal
    # year item6 couldn't derive a Q4 for), never a reason to block the write.
    if "period_months" in cols:
        bad = ~df["period_months"].isin([3, 6, 9, 12]) & df["period_months"].notna()
        if bad.any():
            r.warn(f"{int(bad.sum())} row(s) with a period_months value outside {{3,6,9,12}}")
        n_missing = df["period_months"].isna().sum()
        if n_missing:
            r.warn(f"{int(n_missing)} row(s) missing period_months")
        mixed = set(df["period_months"].dropna().unique())
        if {3, 12} <= mixed:
            n3 = int((df["period_months"] == 3).sum())
            n12 = int((df["period_months"] == 12).sum())
            r.warn(f"mixes quarterly and annual periods in the same file "
                   f"({n3} row(s) at period_months=3, {n12} at =12)")
    if "net_revenue" in cols:
        neg = df["net_revenue"] < 0
        if neg.any():
            r.warn(f"{int(neg.sum())} row(s) with negative net_revenue")
    if "flows_defined" in cols:
        n_undefined = int((df["flows_defined"] == 0).sum())
        if n_undefined:
            r.warn(f"{n_undefined} row(s) with flows_defined=0 "
                   f"(YTD reconstruction was unsafe -- flows NaN'd, not guessed)")
    return r


def validate_company_info(df: pd.DataFrame) -> ValidationResult:
    r = ValidationResult()
    if df.empty:
        r.error("empty dataframe")
        return r
    required = ["ticker", "ticker_primary", "corporate_name", "cvm_code", "cnpj"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        r.error(f"missing columns: {missing}")
        return r
    if df["ticker"].duplicated().any():
        r.error(f"{df['ticker'].duplicated().sum()} duplicate tickers")
    return r


def validate_macro(df: pd.DataFrame, name: str) -> ValidationResult:
    r = _common(df, "reference_date", ["reference_date", name])
    if not r.passed:
        return r
    if df[name].isna().all():
        r.error("all values null")
    return r


def validate_dividends(df: pd.DataFrame) -> ValidationResult:
    r = _common(df, "ex_date", DIVIDEND_COLS)
    if not r.passed:
        return r
    if (df["value_per_share"] <= 0).any():
        r.error(f"{(df['value_per_share'] <= 0).sum()} rows with value_per_share <= 0")
    return r


def validate_corporate_events(df: pd.DataFrame) -> ValidationResult:
    r = _common(df, "date", CORP_EVENT_COLS)
    if not r.passed:
        return r
    if (df["factor"] <= 0).any():
        r.error(f"{(df['factor'] <= 0).sum()} rows with factor <= 0")
    return r


def validate_sectors(df: pd.DataFrame) -> ValidationResult:
    r = ValidationResult()
    if df.empty:
        r.error("empty dataframe")
        return r
    missing = [c for c in ("name", "count") if c not in df.columns]
    if missing:
        r.error(f"missing columns: {missing}")
    return r
