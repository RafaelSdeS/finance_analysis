"""
yf/prices.py — yfinance OHLCV collector.

Split out of yf_collectors.py (docs/DATA_LAYER_ORGANIZATION_PLAN.md §O3).
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

import numpy as np
import pandas as pd
import yfinance as yf

from .. import checkpoint, config, validate
from ..storage import _merge_save
from ._common import (
    _bolsai_junction_date,
    _extract_dividends,
    _last_completed_trading_day,
    _prices_fetch_start,
    _reconcile_yfinance_junction,
    _repair_bad_ohlc,
    _retry,
    _yf_symbol,
)

log = logging.getLogger(__name__)

# No real security has ever traded anywhere near this (Berkshire Hathaway
# Class A, the most expensive legitimate US stock ever, tops out around
# $700K) -- a generous ceiling with zero false-positive risk on genuine
# stocks, however expensive. Found scaling US price collection to the long
# tail of deeply-diluted penny stocks (2026-07-30): yfinance's OWN raw
# (auto_adjust=False) history for tickers with many extreme cumulative
# reverse splits is itself corrupted at the source -- confirmed on 9 tickers
# (ADTX, MRDN, XTIA, NXPL, JAGX, TOPS, PPCB, NUWE, BINI), all classic
# repeated-reverse-split penny stocks: 60-90% of EACH ticker's rows show a
# "Close" in the billions to quadrillions (ADTX max $3.71e12/share, BINI max
# $3.00e17/share) -- not something our split-repair introduces (confirmed by
# inspecting raw yfinance output directly, before any of our code touches
# it), and not something recoverable (there's no way to reconstruct the true
# historical price from data this corrupted). Left unguarded, this fed into
# the split-reverse-adjustment multiplication and failed validate_prices with
# a confusing "adj_open/adj_close outside [adj_low, adj_high]" error that
# gives no hint of the real cause. Same "unfixable vendor corruption ->
# quarantine, don't try to repair" precedent as BR's WDCN3, just detected
# rather than hardcoded, since the signature (implausible absolute magnitude)
# is mechanically checkable.
_MAX_PLAUSIBLE_PRICE = 10_000_000.0


def _fetch_and_shape_prices(ticker: str, fetch_start: str, suffix: str | None = None,
                             changes_out: dict | None = None) -> pd.DataFrame | None:
    """Fetch one ticker's yfinance OHLCV from fetch_start through now and shape
    it into the on-disk raw-prices schema. Shared by collect_prices_yf (auto-
    computed incremental range) and backfill_price_gap (explicit historical
    range) so the split-boundary fix and non-positive-OHLC repair below live
    in exactly one place. Returns None if yfinance has no rows for this span.

    `suffix` overrides config.YF_SUFFIX (e.g. "" for US tickers, which need no
    exchange suffix) without touching the BR default for existing callers.

    `changes_out` (default None, no effect on any existing caller): pass a
    dict to also collect this ticker's dividend/split activity from THIS SAME
    response instead of a second, dividends-specific fetch (see
    collect_prices_yf's `collect_dividends` param) -- `changes_out["dividends"]`
    gets a (possibly empty) DataFrame appended when there's a nonzero dividend
    row, `changes_out["split"] = True` is set when a split is found (mirrors
    the reverse-adjustment check below, which already reads this same column).
    """
    t = yf.Ticker(_yf_symbol(ticker, suffix))
    raw = _retry(lambda: t.history(start=fetch_start, auto_adjust=False, actions=True), f"prices/{ticker}",
                 retry_on_empty=True)
    if raw.empty:
        return None

    if (raw["Close"] > _MAX_PLAUSIBLE_PRICE).any():
        log.warning("prices %s: %d row(s) with an implausible Close (max $%.3g/share) -- "
                    "yfinance's own source data is corrupted for this ticker, not repairable, "
                    "skipping entirely", ticker, (raw["Close"] > _MAX_PLAUSIBLE_PRICE).sum(), raw["Close"].max())
        return None

    raw = _repair_bad_ohlc(raw, ticker)

    # auto_adjust=True issues NO separate request in yfinance 0.2.66 -- history()
    # builds one response and then applies utils.auto_adjust(df) as a pure local
    # post-process (ratio = Adj Close / Close, rescale OHL, rename Adj Close ->
    # Close). So auto_adjust=True's Close IS this same response's "Adj Close",
    # and reading it directly here is a strict improvement over the old second
    # request, not just a faster path: confirmed 2026-08-13 on the two tickers
    # that motivated the reindex/mask guards below -- DEC's documented 39-row
    # mismatch reproduces from two independent calls, but the 1,370 rows the two
    # responses DO share are bit-identical, so the mismatch was a request-count
    # artifact, never a data difference; SAFE's 1,024 non-positive values appear
    # identically in this same Adj Close column, confirming the mask guard is
    # about vendor corruption, not a computation artifact of auto_adjust.
    if "Adj Close" not in raw.columns:
        log.warning("prices %s: no 'Adj Close' column in yfinance response, skipping", ticker)
        return None
    adj_close = raw["Adj Close"].mask(raw["Adj Close"] <= 0)

    if changes_out is not None:
        divs = _extract_dividends(ticker, raw)
        if len(divs):
            changes_out.setdefault("dividends", []).append(divs)

    # Split-boundary fix: reverse-adjust any pre-split rows within THIS fetch
    # (which may re-span multiple quarters now, see _prices_fetch_start) back to
    # BolsAI's unadjusted convention. Always logged loudly so it can be spot-checked.
    # Read from the actions=True column on `raw` rather than a separate t.splits
    # call -- same window, one fewer yfinance request per ticker (2026-08-12).
    splits = raw["Stock Splits"]
    affected = splits[splits > 0]
    if len(affected):
        if changes_out is not None:
            changes_out["split"] = True
        log.warning("prices %s: split(s) in fetch window %s — reverse-adjusting "
                   "pre-split rows to BolsAI's unadjusted convention",
                   ticker, affected.to_dict())
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


MAX_CONSECUTIVE_FAILURES = 25  # a systemic problem, not a legitimate coverage-gap cluster --
# confirmed 2026-07-29: a stale yfinance session after a laptop suspend/resume failed EVERY
# subsequent ticker, each one individually logged as "no yfinance coverage" though a fresh
# process fetched every one of them instantly and correctly. Real coverage gaps do cluster
# (OTC/foreign/shell tickers grouped together in the crosswalk's roughly market-cap-sorted
# order) but not indefinitely -- this catches "everything is failing", not "this stretch of
# small caps has thin coverage", and fails loudly rather than silently mislabeling the rest
# of a multi-hour run.

MAX_CONSECUTIVE_FAILURES_RESUME = 300  # skip_existing=True gets a much looser bound -- real
# structural difference, found the hard way (2026-07-30): bumping the plain threshold above
# to 40 after one genuine cluster tripped it (individually verified: every ticker in the
# stretch was legitimately uncoverable, and a fresh session fetched AAPL/MSFT instantly right
# after) didn't hold -- the VERY NEXT resume attempt tripped the new 40 threshold too, on a
# COMPLETELY DIFFERENT set of tickers. Root cause: skip_existing permanently skips every
# ticker that already succeeded, so each successive resume pass draws its "still to fetch"
# pool from an increasingly concentrated remainder of exactly the tickers that failed
# LAST time -- the "coincidental clustering" assumption behind a tight threshold breaks down
# by construction in resume mode, and no fixed bump survives more than one more pass. Kept
# high rather than disabled outright so a genuinely catastrophic full-session failure (the
# ORIGINAL 2026-07-29 incident this guard exists for) still can't silently burn through an
# entire remaining pool unnoticed.

EMPTY_RUNS_SKIP_THRESHOLD = 3   # consecutive empty/no-coverage runs before going dark on a ticker
EMPTY_RUNS_REPROBE_INTERVAL = 10  # re-probe every Nth run even while dark (catches a re-listing)
# Negative cache for dead/delisted tickers (2026-08-13): a genuinely delisted ticker (128
# "possibly delisted" + 85 empty-result retries in one US prices log) gets re-probed with a
# full network request -- AND pays _retry's empty-result backoff sleep -- on EVERY run,
# forever, with no distinction from a ticker that's merely between updates. `empty_runs`
# (cp[ticker]) tracks consecutive empty results; once it reaches the threshold, the ticker is
# skipped outright (no request, no sleep) except on every Nth run, which still probes for real
# -- catches a genuine re-listing eventually instead of caching "dead" permanently. Any
# non-None fetch (real yfinance coverage, whether new rows or none since last time) resets the
# counter, so a briefly-flaky-but-real ticker never gets stuck in the negative cache.


def collect_prices_yf(tickers: list[str], mode: str, price_dir=None, suffix: str | None = None,
                       floor: str | None = None, skip_existing: bool = False, workers: int = 1,
                       full_refetch: set[str] | None = None,
                       dividend_dir=None, collect_dividends: bool = False) -> set[str] | None:
    """`price_dir`/`suffix`/`floor` default to the BR globals (config.PRICES_DIR /
    config.YF_SUFFIX / config.START_DATE). Pass price_dir=config.US_PRICES_DIR, suffix="",
    floor="1900-01-01" for the pure-yfinance US path — there is no BolsAI history to
    reconcile against, so _bolsai_junction_date/_reconcile_yfinance_junction below are
    no-ops for that case (confirmed: both short-circuit when no on-disk row has a non-NaN
    num_trades, which is true for every row in a US-only file).

    `skip_existing` (default off) skips a ticker outright if its file already exists --
    NOT a general-purpose flag, a narrow escape hatch for resuming an interrupted FIRST-TIME
    backfill within the same short window (hours, not months). Every ticker normally
    re-fetches its ENTIRE span every run on purpose (see _prices_fetch_start): a dividend
    paid after one collection would otherwise never get backward-adjusted into
    already-stored history. Setting this True skips that re-check entirely for whatever's
    already on disk -- fine when "already on disk" means "collected a few hours ago in this
    same backfill," wrong for --mode update's quarterly cadence, where months can pass and a
    real new dividend needs exactly the re-fetch this flag skips.

    `workers` defaults to 1 (unchanged, strictly-sequential behavior, same as before this
    parameter existed) -- NOT collect_dividends_yf's default-4, deliberately. Prices makes
    ONE yfinance request per ticker (since 2026-08-13; auto_adjust=True used to trigger a
    second, independent t.history() call -- redundant, see _fetch_and_shape_prices), same
    as dividends now, on the same vendor whose rate limit prices' OWN throttle
    (YF_RATE_LIMIT_SLEEP) already exists to protect (a real Yahoo 429 incident hit THIS
    specific collector earlier the same day this parameter was added) -- kept
    conservative rather than immediately matched to dividends' default-4, since the
    429 incident was about cumulative request volume over hours, not purely the old
    per-ticker request count; opt into a higher value explicitly for a large backfill,
    don't default to it. Each thread keeps its own YF_RATE_LIMIT_SLEEP pace, same
    reasoning as collect_dividends_yf's workers.

    The consecutive-failure guard (MAX_CONSECUTIVE_FAILURES/_RESUME) is now PER-WORKER
    (thread-local), not global -- a real design question, not an oversight: the original
    global counter's whole premise is "a strict, unbroken RUN of failures in submission
    order is implausible for genuine coverage gaps, so it signals a stuck session."
    That premise breaks under concurrency -- several workers hitting scattered bad
    tickers at once would look like one long "consecutive" streak despite being
    unrelated coincidences. A per-worker streak preserves the actual signal (THIS
    worker's own connection/session seems stuck) without being diluted or falsely
    tripped by unrelated concurrent workers. With workers=1 this is exactly the
    original global counter (one thread, so "per-worker" and "global" coincide).

    `full_refetch` (default None reproduces today's behavior exactly: every
    ticker gets the full re-fetch _prices_fetch_start otherwise always does).
    Pass a `set[str]` to fetch TAIL-ONLY (since the last stored row) for any
    ticker NOT in the set, and full-span only for tickers IN it. Meant to be
    fed the changed-ticker set collect_dividends_yf returns (dividends run
    first in the fast refresh path) -- a ticker with no new dividend or split
    can't have had its stored adj_close go stale, so re-fetching its whole
    history is pure waste. Tail-only tickers whose checkpoint already covers
    the last completed trading day are skipped outright (no request at all).

    Negative cache for dead tickers (EMPTY_RUNS_SKIP_THRESHOLD/
    _REPROBE_INTERVAL, 2026-08-13): once a ticker returns EMPTY_RUNS_SKIP_
    THRESHOLD consecutive empty/no-coverage results (delisted, renamed off
    this symbol, never had yfinance coverage), it's skipped outright -- no
    request, no retry/backoff sleep -- for every run except every
    EMPTY_RUNS_REPROBE_INTERVAL-th, which still probes for real so a genuine
    re-listing isn't cached as dead forever. Deliberately NOT routed through
    the consecutive-failure streak guard above: this is a per-TICKER history
    across runs (persisted in the checkpoint), not a per-RUN streak across
    tickers, and a skip here is an intentional decision, not a failure signal.

    `collect_dividends` (default off, no effect on any existing caller;
    `dividend_dir` defaults to config.DIVIDENDS_DIR, mirroring `price_dir`):
    fold dividends collection into THIS SAME price fetch instead of a
    separate collect_dividends_yf pass (2026-08-13) -- actions=True already
    returns the Dividends column alongside OHLCV (see _fetch_and_shape_prices'
    `changes_out`), so a ticker with nothing new costs ONE request total,
    not two. Returns the set of tickers with a new dividend or split THIS
    call found (empty set if collect_dividends=False was never actually
    checked; None if collect_dividends=False) -- feed straight into a second
    call's `full_refetch` for just that subset, refresh.py's actual usage:
    call once tail-only (full_refetch=set()) to detect + fetch tails +
    write whatever dividends the tail window covers, then call again with
    full_refetch=<the returned set> to force a full re-fetch (price AND
    dividend history) for exactly the tickers that need one. Does NOT touch
    collect_dividends_yf's own `yf_dividends` checkpoint -- only the
    dividends PARQUET FILE, which is the actual source of truth
    (_seed_last_date already falls back to reading it directly when the
    checkpoint has no entry). The one accepted gap: a never-payer processed
    only through this path never gets `checked_through` written to that
    checkpoint, so a LATER standalone collect_dividends_yf run (e.g.
    us/pipeline.py's full-backfill resume, a separate, infrequent
    entry point) would re-walk its full history once more for that ticker --
    a one-time cost on a rare path, not a correctness issue, not worth the
    extra plumbing to keep two independent checkpoints in lockstep.
    """
    price_dir = price_dir or config.PRICES_DIR
    dividend_dir = dividend_dir or config.DIVIDENDS_DIR
    cp = checkpoint.load("yf_prices", mode)
    cp_lock = Lock()
    tl = threading.local()
    last_trading_day = _last_completed_trading_day()
    changed: set[str] = set()

    def _one(ticker: str) -> None:
        if skip_existing and (price_dir / f"{ticker}.parquet").exists():
            return
        tail_only = full_refetch is not None and ticker not in full_refetch
        if tail_only:
            last_date = cp.get(ticker, {}).get("last_date")
            if last_date and pd.Timestamp(last_date) >= last_trading_day:
                log.info("prices %s: checkpoint already covers the last completed "
                          "trading day, skipping", ticker)
                return
        empty_runs = cp.get(ticker, {}).get("empty_runs", 0)
        if empty_runs >= EMPTY_RUNS_SKIP_THRESHOLD and empty_runs % EMPTY_RUNS_REPROBE_INTERVAL != 0:
            log.info("prices %s: skipping (negative cache, %d consecutive empty runs, "
                      "next probe at run %d)", ticker, empty_runs,
                      (empty_runs // EMPTY_RUNS_REPROBE_INTERVAL + 1) * EMPTY_RUNS_REPROBE_INTERVAL)
            with cp_lock:
                cp[ticker] = {**cp.get(ticker, {}), "empty_runs": empty_runs + 1}
                checkpoint.save("yf_prices", mode, cp)
            return
        ok = False
        try:
            path = price_dir / f"{ticker}.parquet"
            fetch_start = _prices_fetch_start(cp, ticker, path, floor, tail_only=tail_only)
            # Fetch from the BolsAI junction date itself (one row earlier than
            # fetch_start) when one exists, so _reconcile_yfinance_junction
            # has an anchor row to compute the reconciliation factor from.
            junction_date = _bolsai_junction_date(path, fetch_start)
            actual_fetch_start = str(junction_date.date()) if junction_date is not None else fetch_start

            changes: dict = {} if collect_dividends else None
            df = _fetch_and_shape_prices(ticker, actual_fetch_start, suffix, changes_out=changes)
            if df is None:
                log.info("prices %s: no new rows (delisted/renamed/no yfinance coverage?)", ticker)
                with cp_lock:
                    cp[ticker] = {**cp.get(ticker, {}), "empty_runs": empty_runs + 1}
                    checkpoint.save("yf_prices", mode, cp)
            else:
                if collect_dividends:
                    div_parts = changes.get("dividends", [])
                    if div_parts:
                        div_df = pd.concat(div_parts, ignore_index=True)
                        div_path = dividend_dir / f"{ticker}.parquet"
                        saved_divs = _merge_save(div_df, div_path, "ex_date",
                                                  validate.validate_dividends, f"dividends/{ticker}")
                        if saved_divs is not None:
                            changed.add(ticker)
                    if changes.get("split"):
                        changed.add(ticker)

                df = _reconcile_yfinance_junction(ticker, path, df, junction_date)
                if df.empty:
                    log.info("prices %s: no new rows past the reconciled junction", ticker)
                    ok = True  # a real fetch succeeded, just nothing new -- not a failure signal
                    if empty_runs:
                        with cp_lock:
                            cp[ticker] = {**cp[ticker], "empty_runs": 0}
                            checkpoint.save("yf_prices", mode, cp)
                else:
                    saved = _merge_save(df, path, "trade_date", validate.validate_prices, f"prices/{ticker}")
                    if saved is not None:
                        ok = True
                        with cp_lock:
                            cp[ticker] = {"last_date": str(saved["trade_date"].max().date()), "rows": len(saved)}
                            checkpoint.save("yf_prices", mode, cp)
                        log.info("prices %s: %d total rows", ticker, len(saved))
        except Exception as e:
            log.warning("prices %s: skipping after error: %s", ticker, e)
        finally:
            sleep(config.YF_RATE_LIMIT_SLEEP)

        streak = 0 if ok else getattr(tl, "consecutive_failures", 0) + 1
        tl.consecutive_failures = streak
        threshold = MAX_CONSECUTIVE_FAILURES_RESUME if skip_existing else MAX_CONSECUTIVE_FAILURES
        if streak >= threshold:
            raise RuntimeError(
                f"{streak} consecutive tickers failed (most recently {ticker}) -- "
                "genuine coverage gaps don't run this deep; this looks systemic (stale "
                "connection, real Yahoo throttling, an outage), not coincidence. Aborting "
                "rather than silently mislabeling the rest of the run as 'no coverage'."
            )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_one, tickers))

    return changed if collect_dividends else None


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
    below existed and had to be reverted from data/raw/br/prices/ by hand.
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
