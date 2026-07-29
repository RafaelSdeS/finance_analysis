"""
yf_collectors.py — yfinance-sourced collectors for prices/fundamentals/dividends.

Mirrors collectors.py's contract exactly: collect_X(tickers, mode) -> validate ->
_merge_save -> checkpoint. Reuses _merge_save, checkpoint.py, validate.py as-is —
yfinance is just another source feeding the same idempotent writer.

company_info and macro have no yfinance equivalent and stay BolsAI/BCB-only
(see collectors.py); not touched here.
"""

import logging
from time import sleep

import numpy as np
import pandas as pd
import yfinance as yf

from . import checkpoint, config, validate
from .collectors import _merge_save

log = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _yf_symbol(ticker: str, suffix: str | None = None) -> str:
    return config.TICKER_ALIASES.get(ticker, ticker) + (config.YF_SUFFIX if suffix is None else suffix)


def _retry(fn, label: str, retry_on_empty: bool = False):
    """yfinance is a scraper with no typed exceptions worth special-casing —
    a couple of doubling-backoff retries is enough, no need for client.py's
    full httpx retry machinery (different transport entirely).

    `retry_on_empty` also retries when fn() returns an empty-but-non-raising
    result -- confirmed 2026-07-28 at US-scale collection: QCOM (a decades-listed,
    fully active ticker with 8,714 rows of real history) returned an empty
    DataFrame on first attempt during a large batch run with no exception raised
    at all, then succeeded immediately on manual retry. Without this, a single
    transient hiccup gets silently recorded as permanent "no yfinance coverage"
    for an otherwise-fine ticker. Only used for price history, where an empty
    result this early is surprising; left False for callers where empty is
    often a legitimate answer (e.g. a ticker with no dividends/splits).
    Still returns the (possibly empty) result after exhausting retries rather
    than raising -- a genuinely-empty ticker must degrade gracefully, not error.
    """
    last_err = None
    for attempt in range(config.YF_RETRIES):
        try:
            result = fn()
            is_last = attempt == config.YF_RETRIES - 1
            if retry_on_empty and not is_last and hasattr(result, "empty") and result.empty:
                wait = config.YF_RETRY_SLEEP * 2 ** attempt
                log.warning("%s: empty result (possible transient issue), retry in %ds", label, wait)
                sleep(wait)
                continue
            return result
        except Exception as e:
            last_err = e
            wait = config.YF_RETRY_SLEEP * 2 ** attempt
            log.warning("%s: yfinance error (%s), retry in %ds", label, e, wait)
            sleep(wait)
    raise RuntimeError(f"max retries exceeded for {label}: {last_err}")


def _seed_last_date(cp: dict, ticker: str, path, col: str) -> str | None:
    """Own checkpoint wins; else fall back to the max date already on disk
    (BolsAI backfill or a prior update run), so the first update doesn't
    redownload full history. --mode update keeps its own checkpoint dir,
    decoupled from prototype/full_scale.

    A checkpoint entry is only trusted if the file it describes still
    exists. Real bug, found scaling to the full ~10,432-ticker US universe
    (2026-07-28): re-running under the SAME mode string after wiping
    data/raw/us/prices/*.parquet (to redo a smaller verification pass
    first) left 1,972 stale checkpoint entries claiming "already up to
    date" for tickers whose actual parquet files no longer existed --
    silently skipping real collection for every one of them, with no
    error and no log line distinguishing it from genuinely-nothing-new.
    By construction the checkpoint is only ever written right after a
    successful file write (`collect_prices_yf`'s loop), so this can only
    diverge when the file is deleted independently afterward -- exactly
    the scenario that bit us.
    """
    if ticker in cp and path.exists():
        return cp[ticker].get("last_date") or cp[ticker].get("last_quarter")
    if path.exists():
        return str(pd.read_parquet(path, columns=[col])[col].max().date())
    return None


TRUSTED_MIN_YF_ROWS = 10  # below this, a recorded yfinance span looks truncated, not genuine


def _prices_fetch_start(cp: dict, ticker: str, path, floor: str | None = None) -> str:
    """Where to start the prices fetch from.

    `floor` overrides config.START_DATE (BR's 2000-01-01 backfill floor) for the
    "no prior data at all" case — US collection passes an intentionally early
    floor so yfinance returns as far back as it actually has (verified: old
    NYSE names go back to 1962-01-02, Yahoo's own hard floor).

    yfinance's auto_adjust=True back-adjusts adj_close relative to whatever "now"
    is at fetch time. If each --mode update run only fetched rows after the last
    checkpoint (like every other collector here), each quarterly batch would be
    anchored to its own fetch date and never revisited — a dividend paid after one
    quarter's fetch would permanently fail to propagate back into that quarter's
    already-stored adj_close. So prices is the one collector that re-fetches its
    entire yfinance-sourced span every run: once any yfinance row exists on disk
    (marked by NaN num_trades, a BolsAI-only field), refetch from the EARLIEST
    such row (not the latest) so the whole yfinance era gets recomputed together
    and stays internally consistent. Before that (no yfinance rows yet), behave
    like every other collector: start the day after the last row on disk.

    Below TRUSTED_MIN_YF_ROWS, the recorded span itself is NOT trusted as "this
    is where history starts" — refetch from the floor instead. Real bug, found
    at US-scale during a rate-limited batch run (2026-07-29): GRTX (a real,
    actively-traded Nasdaq biotech listed since 2020) got truncated to 2 rows
    on its first-ever fetch, no exception raised. Anchoring on that thin span
    forever, as the un-guarded logic below does, would permanently prevent it
    from ever fetching its real multi-year history — every subsequent run
    would just re-confirm the same 2 rows are "where it starts." Harmless for
    a genuinely brand-new listing: re-fetching from the floor just returns the
    same few real rows again, and it stops being "thin" once enough real
    trading days accumulate.
    """
    if path.exists():
        yf_start = pd.read_parquet(path, columns=["trade_date", "num_trades"])
        yf_start = yf_start[yf_start["num_trades"].isna()]
        if len(yf_start) >= TRUSTED_MIN_YF_ROWS:
            return str(yf_start["trade_date"].min().date())
        if len(yf_start):
            log.warning("prices %s: only %d yfinance row(s) on disk (below the %d-row trust "
                        "floor) -- treating as a possibly-truncated fetch, retrying from the "
                        "deep floor instead of anchoring on it", ticker, len(yf_start), TRUSTED_MIN_YF_ROWS)
            return floor or config.START_DATE
    last = _seed_last_date(cp, ticker, path, "trade_date")
    return (pd.to_datetime(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") \
        if last else (floor or config.START_DATE)


def _bolsai_junction_date(path, fetch_start: str) -> pd.Timestamp | None:
    """The last BolsAI-sourced row's date immediately preceding a yfinance-era
    refetch, if `fetch_start` (from _prices_fetch_start) marks the start of
    the yfinance era -- i.e. there's a BolsAI row on disk right before it.
    None on a first-ever fetch (no yfinance era exists yet on disk) or when
    fetch_start isn't actually the yfinance era boundary (a plain
    incremental fetch with no BolsAI history to reconcile against).
    """
    if not path.exists():
        return None
    existing = pd.read_parquet(path, columns=["trade_date", "num_trades"])
    bolsai_rows = existing[existing["num_trades"].notna()]
    if bolsai_rows.empty:
        return None
    junction = bolsai_rows["trade_date"].max()
    if junction >= pd.Timestamp(fetch_start):
        return None
    return junction


def _reconcile_yfinance_junction(ticker: str, path, df: pd.DataFrame,
                                  junction_date: pd.Timestamp | None) -> pd.DataFrame:
    """Rescale a freshly-fetched yfinance-era batch to match the frozen
    BolsAI basis at the junction date.

    yfinance's auto_adjust=True recomputes the WHOLE fetched batch's
    adjustment basis relative to "now" every run (see _prices_fetch_start),
    but the BolsAI-era rows immediately before the junction stay frozen at
    whatever basis they were originally collected at. Every dividend paid
    after that freeze opens a growing, un-reconciled gap right at the
    junction -- one small discontinuity per --mode update run, forever
    (2026-07-23 audit finding). Reconciles by an empirical factor from the
    junction date's own values, same pattern as continuity.py's
    ADJ_RECONCILE_TOL splice reconciliation -- never rescales the frozen
    BolsAI side, only the newly-fetched yfinance side.

    `df` must include a row for `junction_date` itself (the caller fetches
    from that date, not from _prices_fetch_start's date, specifically so
    this reconciliation has an anchor) -- that row is dropped from the
    return value regardless, since the junction date's OHLCV belongs to
    BolsAI on disk and must not be overwritten by _merge_save's dedup.
    """
    if junction_date is None or df.empty:
        return df
    if junction_date not in set(df["trade_date"]):
        return df  # fetch didn't return the junction row (holiday/gap) -- nothing to anchor on

    existing = pd.read_parquet(path, columns=["trade_date", "num_trades", "adj_close"])
    bolsai_junction = existing[(existing["trade_date"] == junction_date) & existing["num_trades"].notna()]
    yf_junction = df.loc[df["trade_date"] == junction_date, "adj_close"]

    if not bolsai_junction.empty and len(yf_junction):
        bolsai_adj = bolsai_junction["adj_close"].iloc[0]
        yf_adj = yf_junction.iloc[0]
        if pd.notna(bolsai_adj) and pd.notna(yf_adj) and yf_adj != 0:
            factor = bolsai_adj / yf_adj
            if abs(factor - 1.0) > 1e-9:
                for col in ("adj_open", "adj_high", "adj_low", "adj_close"):
                    df[col] = df[col] * factor
                log.info("prices %s: reconciled yfinance-era adj_* to frozen BolsAI "
                          "junction basis at %s (factor=%.6f)",
                          ticker, junction_date.date(), factor)

    return df[df["trade_date"] != junction_date].reset_index(drop=True)


def _repair_bad_ohlc(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Collapse rows with internally-inconsistent Open/High/Low/Close to their Close.

    Known yfinance glitch class, two confirmed forms:
    - Non-positive Open/High/Low on an otherwise-valid trading day (e.g. BOVA11
      has 13 such rows from 2009).
    - Open or Close falling OUTSIDE that day's own [Low, High] bracket (or
      High < Low) -- confirmed 2026-07-28 at US-scale collection: SHEL had 4
      such rows across its history, e.g. 2023-01-24 Open=51.26 vs that same
      day's Low=56.26, a >5-point gap. An earlier version of this only checked
      non-positive values, so bracket violations like this one slipped through.
    Left alone, EITHER form permanently fails validate_prices for the WHOLE
    ticker, since the whole span gets re-fetched every run (see
    _prices_fetch_start) -- one bad historical day blocks all new data forever.
    """
    o, h, lo, c = raw["Open"], raw["High"], raw["Low"], raw["Close"]
    non_positive = (raw[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    bracket_violation = (o < lo) | (o > h) | (c < lo) | (c > h) | (h < lo)
    bad = (non_positive | bracket_violation) & (c > 0)
    if bad.any():
        log.warning("prices %s: %d rows with internally-inconsistent OHLC from yfinance — "
                    "collapsing to Close (known vendor glitch)", ticker, bad.sum())
        close_fill = raw.loc[bad, "Close"]
        for col in ("Open", "High", "Low", "Close"):
            raw.loc[bad, col] = close_fill
    return raw


# ---------------------------------------------------------------------------
# prices
# ---------------------------------------------------------------------------

def _fetch_and_shape_prices(ticker: str, fetch_start: str, suffix: str | None = None) -> pd.DataFrame | None:
    """Fetch one ticker's yfinance OHLCV from fetch_start through now and shape
    it into the on-disk raw-prices schema. Shared by collect_prices_yf (auto-
    computed incremental range) and backfill_price_gap (explicit historical
    range) so the split-boundary fix and non-positive-OHLC repair below live
    in exactly one place. Returns None if yfinance has no rows for this span.

    `suffix` overrides config.YF_SUFFIX (e.g. "" for US tickers, which need no
    exchange suffix) without touching the BR default for existing callers.
    """
    t = yf.Ticker(_yf_symbol(ticker, suffix))
    raw = _retry(lambda: t.history(start=fetch_start, auto_adjust=False), f"prices/{ticker}",
                 retry_on_empty=True)
    if raw.empty:
        return None

    raw = _repair_bad_ohlc(raw, ticker)

    adj_close = _retry(lambda: t.history(start=fetch_start, auto_adjust=True)["Close"],
                       f"prices/{ticker} adj_close", retry_on_empty=True)

    # Split-boundary fix: reverse-adjust any pre-split rows within THIS fetch
    # (which may re-span multiple quarters now, see _prices_fetch_start) back to
    # BolsAI's unadjusted convention. Always logged loudly so it can be spot-checked.
    splits = t.splits
    if len(splits):
        affected = splits[splits.index >= pd.Timestamp(fetch_start, tz=splits.index.tz)]
        if len(affected):
            log.warning("prices %s: split(s) in fetch window %s — reverse-adjusting "
                       "pre-split rows to BolsAI's unadjusted convention",
                       ticker, dict(affected))
            for split_date, ratio in affected.items():
                mask = raw.index < split_date
                raw.loc[mask, ["Open", "High", "Low", "Close"]] *= ratio

    close = raw["Close"]
    ratio = adj_close / close

    out = pd.DataFrame({
        "ticker": ticker,
        "trade_date": raw.index.tz_localize(None),
        "open": raw["Open"].values,
        "high": raw["High"].values,
        "low": raw["Low"].values,
        "close": close.values,
        "adj_open": (raw["Open"] * ratio).values,
        "adj_high": (raw["High"] * ratio).values,
        "adj_low": (raw["Low"] * ratio).values,
        "adj_close": adj_close.values,
        "volume": raw["Volume"].values,
        "volume_adjusted": raw["Volume"].values,  # ponytail: yfinance doesn't split-adjust
        # volume; BolsAI does. Documented divergence, not worth reconstructing from splits.
        "traded_amount": (close * raw["Volume"]).values,  # approximation, no yfinance equivalent
        "num_trades": np.nan,  # no yfinance equivalent at all; nan keeps it float64,
                                # matching the on-disk BolsAI dtype (None -> object dtype
                                # triggers pd.concat's all-NA FutureWarning)
    })

    return _drop_incomplete_today(out)


def _drop_incomplete_today(df: pd.DataFrame) -> pd.DataFrame:
    """Drop today's row if the fetch landed mid-session: a still-forming intraday
    bar can have low > open (open/close print immediately; high/low keep moving
    from a differently-lagged feed — confirmed 2026-07-28, XOM: low 15c above
    open). One bad row here fails validate_prices for the WHOLE batch via
    _merge_save, silently dropping thousands of otherwise-good historical rows.
    Safe to drop unconditionally: the next run re-fetches this ticker's full
    yfinance-sourced span (see _prices_fetch_start) and picks up the finalized
    close once the session ends.
    """
    today = pd.Timestamp.now().normalize()
    return df[df["trade_date"] < today].reset_index(drop=True)


def collect_prices_yf(tickers: list[str], mode: str, price_dir=None, suffix: str | None = None,
                       floor: str | None = None):
    """`price_dir`/`suffix`/`floor` default to the BR globals (config.PRICES_DIR /
    config.YF_SUFFIX / config.START_DATE). Pass price_dir=config.US_PRICES_DIR, suffix="",
    floor="1900-01-01" for the pure-yfinance US path — there is no BolsAI history to
    reconcile against, so _bolsai_junction_date/_reconcile_yfinance_junction below are
    no-ops for that case (confirmed: both short-circuit when no on-disk row has a non-NaN
    num_trades, which is true for every row in a US-only file)."""
    price_dir = price_dir or config.PRICES_DIR
    cp = checkpoint.load("yf_prices", mode)
    for ticker in tickers:
        try:
            path = price_dir / f"{ticker}.parquet"
            fetch_start = _prices_fetch_start(cp, ticker, path, floor)
            # Fetch from the BolsAI junction date itself (one row earlier than
            # fetch_start) when one exists, so _reconcile_yfinance_junction
            # has an anchor row to compute the reconciliation factor from.
            junction_date = _bolsai_junction_date(path, fetch_start)
            actual_fetch_start = str(junction_date.date()) if junction_date is not None else fetch_start

            df = _fetch_and_shape_prices(ticker, actual_fetch_start, suffix)
            if df is None:
                log.info("prices %s: no new rows (delisted/renamed/no yfinance coverage?)", ticker)
                continue
            df = _reconcile_yfinance_junction(ticker, path, df, junction_date)
            if df.empty:
                log.info("prices %s: no new rows past the reconciled junction", ticker)
                continue

            saved = _merge_save(df, path, "trade_date", validate.validate_prices, f"prices/{ticker}")
            if saved is not None:
                cp[ticker] = {"last_date": str(saved["trade_date"].max().date()), "rows": len(saved)}
                checkpoint.save("yf_prices", mode, cp)
                log.info("prices %s: %d total rows", ticker, len(saved))
        except Exception as e:
            log.warning("prices %s: skipping after error: %s", ticker, e)
        finally:
            sleep(config.RATE_LIMIT_SLEEP)


def _flat_run_fraction(close: pd.Series, min_run: int = 10) -> float:
    """Fraction of rows sitting inside a run of >= min_run consecutive
    identical values.

    yfinance was found (2026-07-14, see ANOMALY_INVESTIGATION.md) to pad
    holes in its OWN historical coverage with a carried-forward stale price
    instead of leaving the date absent — e.g. LREN3's 2002-2005 gap got
    "filled" with a dense, correctly-dated row count that was actually 98%
    a single repeated close, confirmed directly against yfinance's raw feed
    with zero transformation applied on our side. A dense row count alone
    is NOT evidence of real trading; this catches what a row-count check
    misses. 24 of the first 40 candidate tickers hit this before the guard
    below existed and had to be reverted from data/raw/prices/ by hand.
    """
    if len(close) == 0:
        return 0.0
    same = close.diff() == 0
    run = same.groupby((~same).cumsum()).cumsum()
    return float((run >= min_run).sum() / len(close))


# Above this fraction of the fetched batch sitting in a flat run, treat it as
# yfinance coverage-padding rather than real data. Calibrated against the
# 2026-07-14 audit: genuinely clean backfills topped out at 12.6% flat,
# contaminated ones started at 48% — 0.2 sits with margin on both sides.
_MAX_FLAT_RUN_FRACTION = 0.2


def backfill_price_gap(ticker: str, gap_start: str, gap_end: str) -> pd.DataFrame | None:
    """One-off historical backfill for a confirmed BolsAI vendor data gap
    (see ANOMALY_INVESTIGATION.md): fetch yfinance data spanning
    [gap_start, gap_end] and merge in ONLY the dates genuinely missing from
    the existing raw file. Never touches/overwrites an existing row —
    _merge_save's dedup keeps "last" on a date collision, which would let
    yfinance silently replace a good BolsAI row if the fetch window ever
    strayed past the gap's true edges; filtering to missing dates first
    makes that impossible regardless of how loosely gap_start/gap_end are
    specified. Also rejects the whole fetch if it looks like yfinance
    coverage-padding rather than real data — see _flat_run_fraction.
    """
    path = config.PRICES_DIR / f"{ticker}.parquet"
    df = _fetch_and_shape_prices(ticker, gap_start)
    if df is None:
        log.warning("backfill %s: no yfinance data for gap window [%s, %s]", ticker, gap_start, gap_end)
        return None

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df[df["trade_date"] <= pd.Timestamp(gap_end)]
    if path.exists():
        existing_dates = set(pd.read_parquet(path, columns=["trade_date"])["trade_date"])
        df = df[~df["trade_date"].isin(existing_dates)]
    if df.empty:
        log.info("backfill %s: no missing dates in [%s, %s] (already filled?)", ticker, gap_start, gap_end)
        return None

    flat_frac = _flat_run_fraction(df["close"])
    if flat_frac > _MAX_FLAT_RUN_FRACTION:
        log.error("backfill %s: REJECTED — %.0f%% of the %d fetched rows sit in runs of "
                  ">=10 identical closes (yfinance padding a coverage hole with a stale "
                  "carried-forward price, not real data). Not merged.",
                  ticker, flat_frac * 100, len(df))
        return None

    saved = _merge_save(df, path, "trade_date", validate.validate_prices, f"backfill/{ticker}")
    if saved is not None:
        log.info("backfill %s: filled %d missing rows in [%s, %s]", ticker, len(df), gap_start, gap_end)
    return saved


# ---------------------------------------------------------------------------
# fundamentals
# ---------------------------------------------------------------------------

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


def _shares_outstanding(bs: pd.DataFrame, path) -> pd.Series:
    if "Ordinary Shares Number" in bs.index:
        return bs.loc["Ordinary Shares Number"]
    # carry forward the latest value already on disk — avoids an extra, slower t.info call
    if path.exists():
        existing = pd.read_parquet(path)
        if len(existing):
            return pd.Series(existing.iloc[-1]["shares_outstanding"], index=bs.columns)
    return pd.Series(np.nan, index=bs.columns)


def collect_fundamentals_yf(tickers: list[str], mode: str):
    cp = checkpoint.load("yf_fundamentals", mode)
    for ticker in tickers:
        try:
            fund_path = config.FUND_DIR / f"{ticker}.parquet"
            price_path = config.PRICES_DIR / f"{ticker}.parquet"

            t = yf.Ticker(_yf_symbol(ticker))
            qf = _retry(lambda: t.quarterly_income_stmt, f"fundamentals/{ticker} income")
            bs = _retry(lambda: t.quarterly_balance_sheet, f"fundamentals/{ticker} balance")
            if qf.empty or bs.empty:
                log.info("fundamentals %s: no data (delisted/no yfinance coverage?)", ticker)
                continue

            dates = sorted(set(qf.columns) & set(bs.columns))
            last = _seed_last_date(cp, ticker, fund_path, "reference_date")
            if last:
                dates = [d for d in dates if d > pd.Timestamp(last)]
            if not dates:
                log.info("fundamentals %s: up to date", ticker)
                continue

            shares = _shares_outstanding(bs, fund_path)
            prices = pd.read_parquet(price_path)[["trade_date", "close"]].sort_values("trade_date") \
                if price_path.exists() else pd.DataFrame(columns=["trade_date", "close"])

            def ttm(row_name):
                if row_name not in qf.index:
                    return pd.Series(dtype=float)
                s = pd.Series(qf.loc[row_name], dtype=float).sort_index()
                return s.rolling(4).sum() / K

            def point(row_name):
                if row_name not in bs.index:
                    return pd.Series(dtype=float)
                return pd.Series(bs.loc[row_name], dtype=float) / K

            net_revenue, net_income = ttm("Total Revenue"), ttm("Net Income")
            ebitda, ebit = ttm("EBITDA"), ttm("EBIT")
            cost_of_revenue = ttm("Cost Of Revenue")
            equity, total_assets = point("Stockholders Equity"), point("Total Assets")
            total_debt, cash = point("Total Debt"), point("Cash And Cash Equivalents")
            current_assets, current_liabilities = point("Current Assets"), point("Current Liabilities")

            rows = []
            for d in dates:
                close_at_date = prices[prices["trade_date"] <= d]["close"]
                base = {
                    "ticker": ticker,
                    "reference_date": d,
                    "close_price": close_at_date.iloc[-1] if len(close_at_date) else np.nan,
                    "shares_outstanding": shares.get(d, np.nan),
                    "net_income": net_income.get(d, np.nan),
                    "equity": equity.get(d, np.nan),
                    "net_revenue": net_revenue.get(d, np.nan),
                    "total_debt": total_debt.get(d, np.nan),
                    "ebitda": ebitda.get(d, np.nan),
                    "ebit": ebit.get(d, np.nan),
                    "cash": cash.get(d, np.nan),
                    "total_assets": total_assets.get(d, np.nan),
                    "current_assets": current_assets.get(d, np.nan),
                    "current_liabilities": current_liabilities.get(d, np.nan),
                    "cost_of_revenue": cost_of_revenue.get(d, np.nan),
                }
                base["net_debt"] = base["total_debt"] - base["cash"]
                row = {**base, **compute_ratios(base)}
                row.pop("cost_of_revenue", None)  # not part of the on-disk schema
                rows.append(row)

            df = pd.DataFrame(rows)[FUND_FULL_COLS]
            saved = _merge_save(df, fund_path, "reference_date",
                                validate.validate_fundamentals, f"fundamentals/{ticker}")
            if saved is not None:
                cp[ticker] = {"last_quarter": str(saved["reference_date"].max().date()), "rows": len(saved)}
                checkpoint.save("yf_fundamentals", mode, cp)
                log.info("fundamentals %s: %d quarters", ticker, len(saved))
        except Exception as e:
            log.warning("fundamentals %s: skipping after error: %s", ticker, e)
        finally:
            sleep(config.RATE_LIMIT_SLEEP)


# ---------------------------------------------------------------------------
# dividends
# ---------------------------------------------------------------------------

def collect_dividends_yf(tickers: list[str], mode: str, dividend_dir=None,
                          suffix: str | None = None, floor: str | None = None):
    """`dividend_dir`/`suffix`/`floor` default to the BR globals (config.DIVIDENDS_DIR /
    config.YF_SUFFIX / config.START_DATE) -- mirrors collect_prices_yf's override pattern
    for the pure-yfinance US path (price_dir=config.US_DIVIDENDS_DIR, suffix="",
    floor="1900-01-01"). Verified back to 1994 against 5 real long-history payers
    (KO/GE/IBM/PG/XOM, 2026-07-28): plausible quarterly values, and KO's reconciles
    exactly to its known real 1994 dividend once un-split (0.04875 x 4 = $0.195/share,
    matching its real 1996 + 2012 2:1 splits)."""
    dividend_dir = dividend_dir or config.DIVIDENDS_DIR
    cp = checkpoint.load("yf_dividends", mode)
    for ticker in tickers:
        try:
            path = dividend_dir / f"{ticker}.parquet"
            start = _seed_last_date(cp, ticker, path, "ex_date")
            fetch_start = (pd.to_datetime(start) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") \
                if start else (floor or config.START_DATE)

            t = yf.Ticker(_yf_symbol(ticker, suffix))
            hist = _retry(lambda: t.history(start=fetch_start, actions=True), f"dividends/{ticker}")
            if hist.empty or "Dividends" not in hist.columns:
                log.info("dividends %s: no new rows", ticker)
                continue
            divs = hist[hist["Dividends"] > 0]["Dividends"]
            if divs.empty:
                log.info("dividends %s: no new rows", ticker)
                continue

            df = pd.DataFrame({
                "ex_date": divs.index.tz_localize(None),
                "payment_date": None,
                "type": "UNKNOWN",  # ponytail: yfinance can't distinguish JCP vs Dividendo
                "value_per_share": divs.values,
                "adjusted": False,
                "ticker": ticker,
            })

            saved = _merge_save(df, path, "ex_date", validate.validate_dividends, f"dividends/{ticker}")
            if saved is not None:
                cp[ticker] = {"last_date": str(saved["ex_date"].max().date()), "rows": len(saved)}
                checkpoint.save("yf_dividends", mode, cp)
                log.info("dividends %s: %d total payments", ticker, len(saved))
        except Exception as e:
            log.warning("dividends %s: skipping after error: %s", ticker, e)
        finally:
            sleep(config.RATE_LIMIT_SLEEP)


# ---------------------------------------------------------------------------
# self-check (no network)
# ---------------------------------------------------------------------------

def _demo():
    r = {
        "net_income": 100.0, "equity": 500.0, "net_revenue": 1000.0,
        "total_assets": 2000.0, "total_debt": 300.0, "ebitda": 200.0, "ebit": 150.0,
        "cash": 50.0, "current_assets": 400.0, "current_liabilities": 200.0,
        "shares_outstanding": 10.0, "close_price": 100.0, "cost_of_revenue": 600.0,
    }
    out = compute_ratios(r)
    assert out["market_cap"] == 1000.0
    assert abs(out["roe"] - 20.0) < 1e-9
    assert abs(out["roa"] - 5.0) < 1e-9
    assert abs(out["net_margin"] - 10.0) < 1e-9
    assert out["net_debt"] == 250.0
    assert abs(out["debt_equity"] - 0.6) < 1e-9
    assert abs(out["current_ratio"] - 2.0) < 1e-9
    assert np.isnan(compute_ratios({**r, "equity": 0.0})["roe"])  # 100/0 -> inf -> cleaned to NaN
    assert np.isnan(compute_ratios({k: v for k, v in r.items() if k != "ebitda"})["ev_ebitda"])
    print("compute_ratios: OK")

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        seed_path = Path(tmp) / "PETR4.parquet"
        cp = {"PETR4": {"last_date": "2026-01-01"}}
        # File exists: checkpoint is trusted (the normal case).
        pd.DataFrame({"trade_date": pd.to_datetime(["2026-01-01"])}).to_parquet(seed_path)
        assert _seed_last_date(cp, "PETR4", seed_path, "trade_date") == "2026-01-01"
        # File does NOT exist even though the checkpoint has an entry -- must NOT
        # be trusted. Real bug, found scaling to the full US universe
        # (2026-07-28): a data wipe/redo under the SAME mode string left stale
        # checkpoint entries pointing at deleted files, silently skipping real
        # collection for ~1,972 tickers with no error at all.
        ghost_path = Path(tmp) / "GHOST.parquet"
        assert _seed_last_date(cp, "PETR4", ghost_path, "trade_date") is None
    print("_seed_last_date: OK")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "TEST3.parquet"

        # Checkpoint has an entry but the file doesn't exist yet: the checkpoint
        # must NOT be trusted (see _seed_last_date) -- falls back to a full
        # refetch from the floor, not the stale checkpoint date.
        cp2 = {"TEST3": {"last_date": "2026-01-01"}}
        assert _prices_fetch_start(cp2, "TEST3", path, floor="2020-01-01") == "2020-01-01"

        # Once the file genuinely exists, the checkpoint IS trusted again.
        pd.DataFrame({"trade_date": pd.to_datetime(["2026-01-01"]),
                      "num_trades": [100.0]}).to_parquet(path)
        assert _prices_fetch_start(cp2, "TEST3", path) == "2026-01-02"

        # BolsAI-only rows on disk (num_trades populated): same fallback, day after
        # the last row — no yfinance era started yet.
        bolsai_only = pd.DataFrame({
            "trade_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "num_trades": [100.0, 120.0],
        })
        bolsai_only.to_parquet(path)
        assert _prices_fetch_start({}, "TEST3", path) == "2026-01-03"

        # A yfinance era already exists (NaN num_trades) with enough rows to be
        # TRUSTED: re-anchor to its EARLIEST date, not the latest — this is the
        # fix, re-fetching the whole yfinance span every run instead of only
        # appending past the last checkpoint.
        yf_dates = pd.date_range("2026-01-03", periods=TRUSTED_MIN_YF_ROWS, freq="D")
        mixed = pd.concat([
            pd.DataFrame({"trade_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                          "num_trades": [100.0, 120.0]}),
            pd.DataFrame({"trade_date": yf_dates, "num_trades": np.nan}),
        ], ignore_index=True)
        mixed.to_parquet(path)
        assert _prices_fetch_start({}, "TEST3", path) == "2026-01-03"

        # A yfinance era exists but is THIN (below TRUSTED_MIN_YF_ROWS): must NOT
        # be anchored on -- treated as a possibly-truncated first fetch (see
        # GRTX in the docstring) and retried from the floor instead. Without
        # this, a truncated first fetch anchors on itself forever.
        thin = pd.DataFrame({
            "trade_date": pd.to_datetime(["2026-07-17", "2026-07-20"]),
            "num_trades": [np.nan, np.nan],
        })
        thin.to_parquet(path)
        assert _prices_fetch_start({}, "GRTX", path, floor="1962-01-02") == "1962-01-02"
    print("_prices_fetch_start: OK")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "JUNC3.parquet"

        # No file on disk: no junction to reconcile against.
        assert _bolsai_junction_date(path, "2026-01-03") is None

        # BolsAI era only (last row 2026-01-02), fetch_start is the day after --
        # this IS the yfinance-era boundary: junction = 2026-01-02.
        bolsai_only = pd.DataFrame({
            "trade_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "num_trades": [100.0, 120.0], "adj_close": [10.0, 10.5],
        })
        bolsai_only.to_parquet(path)
        junction = _bolsai_junction_date(path, "2026-01-03")
        assert junction == pd.Timestamp("2026-01-02")

        # fetch_start earlier than or equal to the last BolsAI row: NOT a
        # yfinance-era boundary (e.g. a plain incremental fetch mid-BolsAI-era) --
        # no reconciliation anchor.
        assert _bolsai_junction_date(path, "2026-01-02") is None
        assert _bolsai_junction_date(path, "2026-01-01") is None
    print("_bolsai_junction_date: OK")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "RECON3.parquet"
        junction_date = pd.Timestamp("2026-01-02")
        bolsai_only = pd.DataFrame({
            "trade_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "num_trades": [100.0, 120.0], "adj_close": [10.0, 10.5],
        })
        bolsai_only.to_parquet(path)

        # Fetched batch includes the junction row (2026-01-02, adj_close=10.0
        # per yfinance's OWN fresh basis) plus one new day past it. yfinance's
        # implied adj_close at the junction (10.0) disagrees with BolsAI's
        # frozen value (10.5) -- factor = 10.5/10.0 = 1.05, must rescale
        # EVERY row (including the new day) by it, then drop the junction row.
        fetched = pd.DataFrame({
            "trade_date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "adj_open": [9.9, 10.6], "adj_high": [10.1, 10.8],
            "adj_low": [9.8, 10.5], "adj_close": [10.0, 10.7],
        })
        result = _reconcile_yfinance_junction("RECON3", path, fetched.copy(), junction_date)

        assert list(result["trade_date"]) == [pd.Timestamp("2026-01-05")]  # junction row dropped
        assert abs(result.iloc[0]["adj_close"] - 10.7 * 1.05) < 1e-9

        # No junction_date (first-ever fetch, e.g.) -> no-op, nothing dropped/rescaled.
        untouched = _reconcile_yfinance_junction("RECON3", path, fetched.copy(), None)
        pd.testing.assert_frame_equal(untouched, fetched)
    print("_reconcile_yfinance_junction: OK")

    raw = pd.DataFrame({
        "Open": [10.0, 0.0, 5.0], "High": [11.0, 6.0, 5.5],
        "Low": [9.5, 5.0, 4.5], "Close": [10.5, 5.5, 5.0],
    })
    fixed = _repair_bad_ohlc(raw.copy(), "TEST3")
    assert (fixed.loc[1, ["Open", "High", "Low", "Close"]] == 5.5).all()  # glitch row collapsed to Close
    assert list(fixed.loc[0]) == list(raw.loc[0])  # untouched otherwise
    assert list(fixed.loc[2]) == list(raw.loc[2])
    print("_repair_bad_ohlc: OK")

    # _flat_run_fraction must flag yfinance's coverage-padding signature
    # (mostly one repeated value) and pass real, varying data through clean.
    stale = pd.Series([5.0] * 100)
    assert abs(_flat_run_fraction(stale) - 0.9) < 1e-9  # 90 of 100 cross the >=10-run threshold
    varying = pd.Series([1.0, 2.0, 3.0, 2.0, 4.0, 1.0, 5.0, 2.0, 3.0, 6.0])
    assert _flat_run_fraction(varying) == 0.0  # no repeat ever forms a run at all
    mixed = pd.Series([5.0] * 100 + list(range(1, 11)))  # 100 flat + 10 varying
    assert abs(_flat_run_fraction(mixed) - (90 / 110)) < 1e-9
    assert _flat_run_fraction(stale) > _MAX_FLAT_RUN_FRACTION  # would trip the guard
    assert _flat_run_fraction(varying) < _MAX_FLAT_RUN_FRACTION  # would NOT trip the guard
    print("_flat_run_fraction: OK")

    # backfill_price_gap must never let a yfinance row replace an existing
    # on-disk row, even if the fetch window overlaps real data at the edges —
    # only genuinely-missing dates may be written. No network: monkeypatch
    # _fetch_and_shape_prices to return a synthetic fetch spanning both a
    # pre-existing date (should be dropped) and two real gap dates (should
    # be kept).
    import src.data_collection.yf_collectors as _mod
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "GAPTEST.parquet"
        existing = pd.DataFrame({
            "ticker": "GAPTEST",
            "trade_date": pd.to_datetime(["2002-01-01", "2002-01-10"]),
            "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [1.0, 1.0],
            "adj_open": [1.0, 1.0], "adj_high": [1.0, 1.0], "adj_low": [1.0, 1.0], "adj_close": [1.0, 1.0],
            "volume": [100, 100], "volume_adjusted": [100, 100], "traded_amount": [100.0, 100.0],
            "num_trades": [10.0, 10.0],
        })
        existing.to_parquet(path)
        _orig_prices_dir = config.PRICES_DIR
        config.PRICES_DIR = Path(tmp)  # redirect for this check only

        fetched = pd.DataFrame({
            "ticker": "GAPTEST",
            "trade_date": pd.to_datetime(["2002-01-01", "2002-01-05", "2002-01-08"]),
            "open": [999.0, 2.0, 3.0], "high": [999.0, 2.0, 3.0],
            "low": [999.0, 2.0, 3.0], "close": [999.0, 2.0, 3.0],
            "adj_open": [999.0, 2.0, 3.0], "adj_high": [999.0, 2.0, 3.0],
            "adj_low": [999.0, 2.0, 3.0], "adj_close": [999.0, 2.0, 3.0],
            "volume": [1, 1, 1], "volume_adjusted": [1, 1, 1], "traded_amount": [1.0, 1.0, 1.0],
            "num_trades": [np.nan, np.nan, np.nan],
        })
        _orig = _mod._fetch_and_shape_prices
        _mod._fetch_and_shape_prices = lambda ticker, fetch_start: fetched
        try:
            saved = _mod.backfill_price_gap("GAPTEST", "2002-01-01", "2002-01-10")
        finally:
            _mod._fetch_and_shape_prices = _orig
            config.PRICES_DIR = _orig_prices_dir
        assert len(saved) == 4  # 2 original + 2 new gap-fill dates (01-05, 01-08)
        assert saved.loc[saved["trade_date"] == "2002-01-01", "close"].iloc[0] == 1.0  # NOT overwritten by the 999.0 fetch row
        assert set(saved["trade_date"].dt.strftime("%Y-%m-%d")) == {"2002-01-01", "2002-01-05", "2002-01-08", "2002-01-10"}
    print("backfill_price_gap: OK")

    # US-market support: suffix/floor overrides default to the BR globals
    # (empty-string args must NOT fall back — only None means "use the default").
    assert _yf_symbol("PETR4") == "PETR4.SA"
    assert _yf_symbol("AAPL", suffix="") == "AAPL"
    assert _yf_symbol("AAPL", suffix=None) == "AAPL.SA"  # explicit None -> BR default, not ""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "USNEW.parquet"
        assert _prices_fetch_start({}, "USNEW", path) == config.START_DATE  # BR default floor
        assert _prices_fetch_start({}, "USNEW", path, floor="1900-01-01") == "1900-01-01"
    print("_yf_symbol/_prices_fetch_start overrides: OK")

    # Same-day guard: a still-forming intraday bar must never reach validation.
    today = pd.Timestamp.now().normalize()
    df = pd.DataFrame({"trade_date": [today - pd.Timedelta(days=1), today]})
    result = _drop_incomplete_today(df)
    assert list(result["trade_date"]) == [today - pd.Timedelta(days=1)]
    print("_drop_incomplete_today: OK")

    # retry_on_empty: a transient empty result (no exception) must be retried when
    # requested, but still degrade gracefully (return empty, not raise) once retries
    # are exhausted -- confirmed 2026-07-28: QCOM returned empty on first attempt
    # during a large batch run despite having 8,714 rows of real history.
    from unittest import mock
    calls = {"n": 0}
    def flaky_then_ok():
        calls["n"] += 1
        return pd.DataFrame() if calls["n"] == 1 else pd.DataFrame({"x": [1]})
    with mock.patch.object(config, "YF_RETRY_SLEEP", 0):
        out = _retry(flaky_then_ok, "test", retry_on_empty=True)
    assert not out.empty and calls["n"] == 2, "must retry past the first empty result"

    calls["n"] = 0
    with mock.patch.object(config, "YF_RETRY_SLEEP", 0):
        out = _retry(flaky_then_ok, "test", retry_on_empty=False)
    assert out.empty and calls["n"] == 1, "default (False) must NOT retry on empty -- some callers treat empty as legitimate"

    always_empty = lambda: pd.DataFrame()
    with mock.patch.object(config, "YF_RETRY_SLEEP", 0):
        out = _retry(always_empty, "test", retry_on_empty=True)
    assert out.empty, "must degrade gracefully (return empty) once retries are exhausted, not raise"
    print("_retry retry_on_empty: OK")

    # US-market support for dividends: dividend_dir/suffix/floor overrides mirror
    # collect_prices_yf's exact pattern -- verified end-to-end (mocked yf.Ticker),
    # not just checked as default values. Also pins the real KO reconciliation
    # from this function's docstring (0.195/share, KO's real un-split 1994 dividend).
    class _FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start, actions):
            assert self.symbol == "KO", f"suffix override must produce a bare symbol, got {self.symbol}"
            assert start == "1994-01-01", f"floor override must be honored when no prior data exists, got {start}"
            return pd.DataFrame({"Dividends": [0.195]}, index=pd.to_datetime(["1994-03-09"]))

    with tempfile.TemporaryDirectory() as tmp:
        div_dir = Path(tmp)
        with mock.patch.object(yf, "Ticker", _FakeTicker), \
             mock.patch.object(checkpoint, "load", return_value={}), \
             mock.patch.object(checkpoint, "save"):
            collect_dividends_yf(["KO"], mode="test", dividend_dir=div_dir, suffix="", floor="1994-01-01")
        saved = pd.read_parquet(div_dir / "KO.parquet")
        assert len(saved) == 1
        assert abs(saved["value_per_share"].iloc[0] - 0.195) < 1e-9
    print("collect_dividends_yf US overrides: OK")


if __name__ == "__main__":
    _demo()
