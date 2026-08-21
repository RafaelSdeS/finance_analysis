"""
yf/_common.py — shared fetch/repair helpers behind every yfinance collector.

Split out of yf_collectors.py (docs/DATA_LAYER_ORGANIZATION_PLAN.md §O3).
Two functions live here despite sitting under the old module's "prices"
banner: `_extract_dividends` and `_last_completed_trading_day` are both
called from collect_dividends_yf (yf/dividends.py) as well as the prices
path (yf/prices.py) -- putting either in one submodule would make the other
import across siblings for no reason.
"""

import logging
from time import sleep

import pandas as pd

from .. import config

log = logging.getLogger(__name__)


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

    Final fallback (2026-08-13): `cp[ticker]["checked_through"]` with no file
    on disk at all -- a ticker collect_dividends_yf confirmed pays no
    dividends (or has none newer than that date), which the on-disk-file
    branches above can never represent (no dividends -> no file, ever).
    Without this, a never-payer has no checkpoint entry either (the old
    early-return paths wrote nothing), so every run re-walked its FULL
    multi-decade history from `floor` just to reconfirm "still nothing" --
    measured at US scale: 5,376 of 9,593 priced tickers have no dividends
    file. Deliberately a SEPARATE key from `last_date`/`last_quarter`, not a
    fallback value for them: it only ever asserts "we looked through this
    date," never "rows are on disk," so it can't resurrect the wiped-file bug
    above -- a payer whose file gets deleted still correctly returns None
    here (its checkpoint entry holds `last_date`, not `checked_through`).
    """
    if ticker in cp and path.exists():
        return cp[ticker].get("last_date") or cp[ticker].get("last_quarter")
    if path.exists():
        return str(pd.read_parquet(path, columns=[col])[col].max().date())
    return cp.get(ticker, {}).get("checked_through")


TRUSTED_MIN_YF_ROWS = 10  # below this, a recorded yfinance span looks truncated, not genuine


def _prices_fetch_start(cp: dict, ticker: str, path, floor: str | None = None, tail_only: bool = False) -> str:
    """Where to start the prices fetch from.

    `tail_only` (default off): once the on-disk yfinance-era span is trusted
    (>= TRUSTED_MIN_YF_ROWS), fetch only since the last stored row instead of
    re-fetching the whole yfinance era every run. Used by the fast refresh
    path for tickers with no new dividend/split since the last collection --
    those are the only ones whose stored adj_close can't have gone stale (see
    collect_prices_yf's `full_refetch`). The thin-file and no-file guards
    below are untouched and still win: a truncated or missing file always
    falls through to the deep floor regardless of `tail_only`.

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
            if tail_only:
                last = yf_start["trade_date"].max()
                return (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
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


def _extract_dividends(ticker: str, raw: pd.DataFrame) -> pd.DataFrame:
    """Shape the Dividends column off an already-fetched prices `raw` response
    into the on-disk dividends schema -- the exact same shape
    collect_dividends_yf's own (separate) fetch produces, since actions=True
    already returns this column alongside OHLCV (2026-08-13: no need for a
    second, dividends-specific request just to re-derive it). Always returns
    a DataFrame (possibly empty, never None) so callers can check `len()`
    without a separate None-check.
    """
    divs = raw["Dividends"] if "Dividends" in raw.columns else pd.Series(dtype=float)
    divs = divs[divs > 0]
    return pd.DataFrame({
        "ex_date": divs.index.tz_localize(None),
        "payment_date": None,
        "type": "UNKNOWN",  # ponytail: yfinance can't distinguish JCP vs Dividendo
        "value_per_share": divs.values,
        "adjusted": False,
        "ticker": ticker,
    })


def _last_completed_trading_day() -> pd.Timestamp:
    """Most recent weekday strictly before today. `_drop_incomplete_today` always
    drops today's row, so the freshest a checkpoint can legitimately be is this
    date -- used by collect_prices_yf's same-day skip to avoid a network round
    trip for a ticker that's provably already current. A market holiday just
    means one wasted-but-cheap fetch that returns nothing new; not worth a
    market-calendar dependency to shave that off too.
    """
    d = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d -= pd.Timedelta(days=1)
    return d
