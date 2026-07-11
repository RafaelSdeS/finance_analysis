"""
validate.py — lightweight per-collector data quality gate (runs before write).

This is the *schema/sanity* gate, distinct from the cross-source check in
tests/data_collection/validate_vs_yfinance.py (the validation STAGE). Returns a
ValidationResult; collectors refuse to save on errors, log on warnings.
"""

from dataclasses import dataclass, field

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
    if df[date_col].duplicated().any():
        r.warn(f"duplicate {date_col} values present")
    return r


def validate_prices(df: pd.DataFrame) -> ValidationResult:
    r = _common(df, "trade_date", PRICE_COLS)
    if not r.passed:
        return r
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
        bracket_violation = (
            (df[open_c] < df[low_c]) | (df[open_c] > df[high_c])
            | (df[close_c] < df[low_c]) | (df[close_c] > df[high_c])
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
