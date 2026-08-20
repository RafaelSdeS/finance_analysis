#!/usr/bin/env python3
"""
Self-check for yf_collectors.py's pure helper functions -- most importantly
_prices_fetch_start(), the staleness-anchor fix for `--mode update` (CLAUDE.md:
without it, a dividend paid after one quarter's fetch would permanently fail
to propagate back into that quarter's already-stored adj_close -- one silent,
cumulative discontinuity per update, forever).

Moved here verbatim from yf_collectors._demo() (2026-08-19). It had grown to
462 lines -- a third of the module -- which is well past the size where an
inline `_demo()` self-check pays for itself; production code should not ship
its own test suite. The `_demo()` convention still stands for genuinely small
checks (see storage.py, ~30 lines).

Kept as ONE function rather than split into 23 `test_*` functions: the blocks
share local setup (tmpdir, mock scaffolding) and splitting them is a rewrite,
not a move. Split it if a specific block ever needs to run in isolation.

Run from project root: python tests/data_collection/test_yf_collectors.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection import checkpoint, config, validate  # noqa: E402
from src.data_collection.yf_collectors import (  # noqa: E402
    _MAX_FLAT_RUN_FRACTION,
    TRUSTED_MIN_YF_ROWS,
    _bolsai_junction_date,
    _drop_incomplete_today,
    _fetch_and_shape_prices,
    _flat_run_fraction,
    _prices_fetch_start,
    _reconcile_yfinance_junction,
    _repair_bad_ohlc,
    _retry,
    _seed_last_date,
    _yf_symbol,
    collect_dividends_yf,
    compute_ratios,
)


def main():
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

    # _fetch_and_shape_prices makes exactly ONE yfinance request per ticker
    # (2026-08-13): auto_adjust=True used to trigger a SECOND, independent
    # t.history() call. Redundant -- yfinance applies auto_adjust as a pure
    # local post-process of the SAME response (utils.auto_adjust: ratio =
    # Adj Close / Close, rescale OHL, rename). Reading "Adj Close" straight off
    # the single auto_adjust=False response is identical, and removes the
    # two-independent-requests failure mode entirely (a second call could
    # return a different row count or index than the first -- confirmed on DEC
    # historically; no longer possible with one call, since there's only one
    # response to disagree with).
    from unittest import mock
    _idx3 = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]).tz_localize("America/New_York")

    class _FakeTickerSingleRequest:
        call_count = 0

        def __init__(self, symbol):
            self.splits = pd.Series([], dtype=float)

        def history(self, start, auto_adjust, actions=False):
            _FakeTickerSingleRequest.call_count += 1
            assert not auto_adjust, "must only ever request auto_adjust=False -- a second, redundant request regressed"
            return pd.DataFrame({"Open": [1.0, 2.0, 3.0], "High": [1.0, 2.0, 3.0],
                                  "Low": [1.0, 2.0, 3.0], "Close": [1.0, 2.0, 3.0],
                                  "Adj Close": [1.0, 2.0, 3.0],
                                  "Volume": [100, 100, 100], "Stock Splits": [0.0, 0.0, 0.0]}, index=_idx3)

    with mock.patch.object(yf, "Ticker", _FakeTickerSingleRequest):
        out = _fetch_and_shape_prices("SINGLEREQ", "2020-01-01", suffix="")
    assert _FakeTickerSingleRequest.call_count == 1, \
        f"must call history() exactly once per ticker, got {_FakeTickerSingleRequest.call_count}"
    assert list(out["adj_close"]) == [1.0, 2.0, 3.0]
    print("_fetch_and_shape_prices single-request: OK")

    class _FakeTickerNoAdjClose:
        def __init__(self, symbol):
            self.splits = pd.Series([], dtype=float)

        def history(self, start, auto_adjust, actions=False):
            return pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0],
                                  "Volume": [100], "Stock Splits": [0.0]}, index=_idx3[:1])

    with mock.patch.object(yf, "Ticker", _FakeTickerNoAdjClose):
        out = _fetch_and_shape_prices("NOADJ", "2020-01-01", suffix="")
    assert out is None, "a response missing 'Adj Close' entirely must be skipped cleanly, not KeyError"
    print("_fetch_and_shape_prices missing Adj Close guard: OK")

    class _FakeTickerNegativeAdjClose:
        def __init__(self, symbol):
            self.splits = pd.Series([], dtype=float)

        def history(self, start, auto_adjust, actions=False):
            idx = _idx3[:2]
            return pd.DataFrame({"Open": [10.0, 10.0], "High": [10.0, 10.0],
                                  "Low": [10.0, 10.0], "Close": [10.0, 10.0],
                                  "Adj Close": [10.0, -5.0],  # impossible negative, like SAFE
                                  "Volume": [100, 100], "Stock Splits": [0.0, 0.0]}, index=idx)

    with mock.patch.object(yf, "Ticker", _FakeTickerNegativeAdjClose):
        out = _fetch_and_shape_prices("NEGADJ", "2020-01-01", suffix="")
    bad_row = out.loc[out["trade_date"] == "2020-01-02"].iloc[0]
    assert pd.isna(bad_row["adj_close"]), "a negative adj_close must be masked to NaN, not written as-is"
    assert pd.isna(bad_row["adj_open"]) and pd.isna(bad_row["adj_high"]) and pd.isna(bad_row["adj_low"]), \
        "adj_open/high/low are derived from adj_close's ratio -- must also come out NaN, not a corrupted negative"
    r = validate.validate_prices(out)
    assert r.passed, f"masked NaN must not fail validate_prices' non-positive-adj_* check, got {r.errors}"
    print("_fetch_and_shape_prices negative adj_close: OK")

    # Implausible-price guard: yfinance's OWN raw data can be corrupted at the
    # source for penny stocks with many extreme cumulative reverse splits
    # (confirmed on ADTX/MRDN/XTIA/NXPL/JAGX/TOPS/PPCB/NUWE/BINI, 2026-07-30) --
    # must skip the whole ticker cleanly (return None) rather than let it
    # cascade into a confusing validate_prices bracket-violation failure.
    class _FakeTickerCorruptedPrice:
        def __init__(self, symbol):
            self.splits = pd.Series([], dtype=float)

        def history(self, start, auto_adjust, actions=False):
            idx = _idx3[:2]
            close = [1.5, 5e12]  # one row wildly implausible
            return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                                  "Adj Close": close, "Volume": [100, 100]}, index=idx)

    with mock.patch.object(yf, "Ticker", _FakeTickerCorruptedPrice):
        out = _fetch_and_shape_prices("CORRUPT", "2020-01-01", suffix="")
    assert out is None, "a ticker with an implausible (multi-trillion) Close must be skipped entirely, not partially written"
    print("_fetch_and_shape_prices implausible-price guard: OK")

    # Split-boundary fix now reads the Stock Splits column off the SAME
    # actions=True history() call instead of a separate t.splits request
    # (2026-08-12, dropped the extra request) -- pin that the reverse-adjust
    # still fires correctly from that column.
    class _FakeTickerSplit:
        def __init__(self, symbol):
            pass

        def history(self, start, auto_adjust, actions=False):
            return pd.DataFrame({"Open": [10.0, 10.0, 20.0], "High": [10.0, 10.0, 20.0],
                                  "Low": [10.0, 10.0, 20.0], "Close": [10.0, 10.0, 20.0],
                                  "Adj Close": [10.0, 10.0, 20.0],
                                  "Volume": [100, 100, 100],
                                  "Stock Splits": [0.0, 2.0, 0.0]}, index=_idx3)  # 2:1 split on day 2

    with mock.patch.object(yf, "Ticker", _FakeTickerSplit):
        out = _fetch_and_shape_prices("SPLIT", "2020-01-01", suffix="")
    pre = out.loc[out["trade_date"] == "2020-01-01"].iloc[0]
    assert pre["open"] == 20.0 and pre["close"] == 20.0, \
        f"pre-split row must be reverse-adjusted by the 2:1 ratio read from Stock Splits, got {pre}"
    post = out.loc[out["trade_date"] == "2020-01-02"].iloc[0]
    assert post["close"] == 10.0, "split day itself and rows after it must be untouched"
    print("_fetch_and_shape_prices split-boundary fix via Stock Splits column: OK")

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

    # Threading speedup (2026-07-31): workers>1 runs multiple tickers
    # concurrently, sharing the SAME checkpoint dict `cp` across threads. Real
    # race found designing this: checkpoint.save()'s own lock only protects its
    # file write, not a caller mutating the same dict object from ANOTHER
    # thread while save() is mid-snapshot. 20 tickers, 8 workers, every one
    # must land in the final checkpoint state -- a lost update (the race this
    # guards against) would silently drop one or more tickers.
    n = 20
    tickers = [f"DIV{i}" for i in range(n)]

    class _FakeTickerMulti:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start, actions):
            return pd.DataFrame({"Dividends": [1.0]}, index=pd.to_datetime(["2020-01-01"]))

    with tempfile.TemporaryDirectory() as tmp:
        div_dir = Path(tmp)
        with mock.patch.object(yf, "Ticker", _FakeTickerMulti), \
             mock.patch.object(checkpoint, "load", return_value={}), \
             mock.patch.object(checkpoint, "save") as mock_save, \
             mock.patch.object(config, "YF_RATE_LIMIT_SLEEP", 0):
            collect_dividends_yf(tickers, mode="test", dividend_dir=div_dir, suffix="", floor="2020-01-01", workers=8)
        final_cp = mock_save.call_args_list[-1].args[2]
        missing = set(tickers) - set(final_cp)
        assert not missing, f"concurrent checkpoint writes lost {len(missing)} ticker(s): {missing}"
        assert len(mock_save.call_args_list) == n, (
            f"every successful ticker must persist its own checkpoint update, got {len(mock_save.call_args_list)}")
        for tk in tickers:
            assert (div_dir / f"{tk}.parquet").exists(), f"{tk}'s file must exist despite concurrent execution"
    print("collect_dividends_yf threading: OK (no lost checkpoint updates across 20 tickers/8 workers)")

    # checked_through fallback (2026-08-13): a never-payer gets no dividends
    # file (nothing to write), so before this fix it also got no checkpoint
    # entry -- every run re-walked its FULL history from `floor` to reconfirm
    # "still nothing." Confirm a no-file ticker still gets a checkpoint entry,
    # and that a second run resumes from checked_through+1 instead of floor.
    calls = []

    class _FakeTickerNeverPays:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start, actions):
            calls.append(start)
            return pd.DataFrame()  # no rows at all, ever

    with tempfile.TemporaryDirectory() as tmp:
        div_dir = Path(tmp)
        cp_store: dict = {}

        def fake_load(name, mode):
            return cp_store.get((name, mode), {})

        def fake_save(name, mode, data):
            cp_store[(name, mode)] = dict(data)

        with mock.patch.object(yf, "Ticker", _FakeTickerNeverPays), \
             mock.patch.object(checkpoint, "load", side_effect=fake_load), \
             mock.patch.object(checkpoint, "save", side_effect=fake_save), \
             mock.patch.object(config, "YF_RATE_LIMIT_SLEEP", 0):
            collect_dividends_yf(["NEVER"], mode="test_checked", dividend_dir=div_dir,
                                  suffix="", floor="2000-01-01")
        assert not (div_dir / "NEVER.parquet").exists(), "a never-payer must not get a file"
        saved_cp = cp_store[("yf_dividends", "test_checked")]
        assert "checked_through" in saved_cp.get("NEVER", {}), \
            "a never-payer must still get a checkpoint entry, or every run re-walks its full history"
        assert calls[0] == "2000-01-01", "first run has no checkpoint, must start from floor"

        with mock.patch.object(yf, "Ticker", _FakeTickerNeverPays), \
             mock.patch.object(checkpoint, "load", side_effect=fake_load), \
             mock.patch.object(checkpoint, "save", side_effect=fake_save), \
             mock.patch.object(config, "YF_RATE_LIMIT_SLEEP", 0):
            collect_dividends_yf(["NEVER"], mode="test_checked", dividend_dir=div_dir,
                                  suffix="", floor="2000-01-01")
        expected_start = (pd.Timestamp(saved_cp["NEVER"]["checked_through"])
                           + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        assert calls[1] == expected_start, \
            f"second run must resume from checked_through+1, not re-walk from floor -- got {calls[1]}"
    print("collect_dividends_yf checked_through fallback: OK")




if __name__ == "__main__":
    main()
