"""
Test P2: apply_ticker_continuity() splices renamed/merged tickers correctly.

Pure code — synthetic prices/fundamentals + a temp continuity map. Covers:
  rename: prices AND fundamentals spliced, no date overlap at the boundary
  merger: prices spliced with the exchange ratio, old fundamentals dropped
  guard:  duplicate ticker+date rows after a bad map must raise

Run from project root:
    python tests/build_dataset/test_ticker_continuity.py
"""

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.build_dataset.continuity import apply_ticker_continuity  # noqa: E402
from test_utils import print_check  # noqa: E402


def _prices(ticker, dates, close, volume=None):
    df = pd.DataFrame({
        "ticker": ticker,
        "trade_date": pd.to_datetime(dates),
        "open": close, "high": [c * 1.01 for c in close],
        "low": [c * 0.99 for c in close], "close": close,
        "adj_open": close, "adj_high": [c * 1.01 for c in close],
        "adj_low": [c * 0.99 for c in close], "adj_close": close,
    })
    if volume is not None:
        df["volume"] = volume
    return df


def _fund(ticker, dates, net_income):
    return pd.DataFrame({
        "ticker": ticker,
        "reference_date": pd.to_datetime(dates),
        "net_income": net_income,
    })


def _map(events):
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"events": events}, f)
    f.close()
    return Path(f.name)


def test_rename_splice():
    # OLD3 trades through Jan, NEW3 (same entity) starts Feb; one Jan date overlaps
    prices = pd.concat([
        _prices("OLD3", ["2021-01-04", "2021-01-05", "2021-02-01"], [10.0, 11.0, 99.0],
                volume=[1000, 1100, 1200]),
        _prices("NEW3", ["2021-02-01", "2021-02-02"], [12.0, 12.5], volume=[1300, 1400]),
    ], ignore_index=True)
    prices["volume"] = prices["volume"].astype("int64")
    fund = pd.concat([
        _fund("OLD3", ["2020-12-31"], [100.0]),
        _fund("NEW3", ["2021-03-31"], [110.0]),
    ], ignore_index=True)
    path = _map([{"old": "OLD3", "new": "NEW3", "date": "2021-02-01",
                  "type": "rename", "ratio": 1.0}])

    p, f = apply_ticker_continuity(prices, fund, path=path)

    assert "OLD3" not in set(p["ticker"]) | set(f["ticker"]), "old ticker must vanish"
    new_p = p[p["ticker"] == "NEW3"].sort_values("trade_date")
    assert len(new_p) == 4, new_p  # 2 old (pre-boundary) + 2 new; overlap row dropped
    assert not new_p.duplicated("trade_date").any(), "no duplicate dates at boundary"
    assert new_p.iloc[0]["close"] == 10.0, "rename must not rescale prices"
    assert new_p["volume"].dtype == "int64", \
        "rename (ratio=1.0) must not touch volume or upcast its dtype to float"
    assert list(new_p["volume"]) == [1000, 1100, 1300, 1400], "rename must not rescale volume"
    new_f = f[f["ticker"] == "NEW3"]
    assert len(new_f) == 2, "rename splices fundamentals too"
    print_check("rename splice", True)
    return True


def test_merger_splice():
    prices = pd.concat([
        _prices("ACQ3", ["2021-01-04", "2021-01-05"], [20.0, 22.0], volume=[1000.0, 1100.0]),
        _prices("SRV3", ["2021-02-01", "2021-02-02"], [11.0, 11.5], volume=[500.0, 550.0]),
    ], ignore_index=True)
    fund = pd.concat([
        _fund("ACQ3", ["2020-12-31"], [500.0]),
        _fund("SRV3", ["2021-03-31"], [700.0]),
    ], ignore_index=True)
    # 1 ACQ3 share -> 0.5 SRV3 shares
    path = _map([{"old": "ACQ3", "new": "SRV3", "date": "2021-02-01",
                  "type": "merger", "ratio": 0.5}])

    p, f = apply_ticker_continuity(prices, fund, path=path)

    srv = p[p["ticker"] == "SRV3"].sort_values("trade_date")
    assert len(srv) == 4 and "ACQ3" not in set(p["ticker"]), srv
    assert srv.iloc[0]["close"] == 10.0, "merger prices must scale by the ratio (20*0.5)"
    assert srv.iloc[0]["high"] == 20.0 * 1.01 * 0.5, "all OHLC columns must scale"
    # continuity at the boundary: 22*0.5=11 vs SRV3's 11 open — no artificial jump
    assert abs(srv.iloc[1]["close"] - 11.0) < 1e-9
    # dollar volume (volume*price) must stay invariant across the splice:
    # price*ratio and volume/ratio cancel out, matching the split-repair fix
    assert abs(srv.iloc[0]["volume"] - 1000.0 / 0.5) < 1e-9, "volume must scale inversely to ratio"
    assert abs(srv.iloc[0]["volume"] * srv.iloc[0]["close"] - 1000.0 * 20.0) < 1e-6, \
        "dollar volume must be preserved across the merger splice"
    f_srv = f[f["ticker"] == "SRV3"]
    assert len(f_srv) == 1 and f_srv.iloc[0]["net_income"] == 700.0, \
        "acquired entity's fundamentals must be dropped, survivor's kept"
    print_check("merger splice", True)
    return True


def test_adj_close_reconciliation():
    # OLD3's vendor series never dividend-adjusted (adj_close == close); NEW3's
    # starts already discounted -- same real-world pattern as BVMF3->B3SA3.
    prices = pd.concat([
        _prices("OLD3", ["2021-01-04", "2021-01-05"], [10.0, 20.0]),
    ], ignore_index=True)
    new_rows = _prices("NEW3", ["2021-02-01", "2021-02-02"], [21.0, 21.5])
    new_rows["adj_close"] = [5.25, 5.375]  # 4x below close, unlike OLD3
    prices = pd.concat([prices, new_rows], ignore_index=True)
    fund = pd.concat([
        _fund("OLD3", ["2020-12-31"], [100.0]),
        _fund("NEW3", ["2021-03-31"], [110.0]),
    ], ignore_index=True)
    path = _map([{"old": "OLD3", "new": "NEW3", "date": "2021-02-01",
                  "type": "rename", "ratio": 1.0}])

    p, _ = apply_ticker_continuity(prices, fund, path=path)
    new_p = p[p["ticker"] == "NEW3"].sort_values("trade_date")

    # factor = 5.25 / 20.0 = 0.2625; OLD3's adj_close rescaled, raw close untouched
    assert new_p.iloc[0]["close"] == 10.0, "raw close must never be touched by reconciliation"
    assert abs(new_p.iloc[0]["adj_close"] - 10.0 * 0.2625) < 1e-9, new_p.iloc[0]["adj_close"]
    assert new_p.iloc[1]["adj_close"] == 20.0 * 0.2625, "old ticker's every adj_close row rescaled"
    assert new_p.iloc[2]["adj_close"] == 5.25, "new ticker's own rows untouched"
    print_check("adj_close basis reconciliation", True)
    return True


def test_adj_close_reconciliation_skips_within_tolerance():
    # 5% boundary mismatch is normal 1-day return noise -- must not be "fixed"
    prices = pd.concat([
        _prices("OLD3", ["2021-01-04", "2021-01-05"], [10.0, 20.0]),
        _prices("NEW3", ["2021-02-01"], [20.9]),  # 4.5% above OLD3's last close
    ], ignore_index=True)
    fund = pd.concat([
        _fund("OLD3", ["2020-12-31"], [100.0]),
        _fund("NEW3", ["2021-03-31"], [110.0]),
    ], ignore_index=True)
    path = _map([{"old": "OLD3", "new": "NEW3", "date": "2021-02-01",
                  "type": "rename", "ratio": 1.0}])

    p, _ = apply_ticker_continuity(prices, fund, path=path)
    new_p = p[p["ticker"] == "NEW3"].sort_values("trade_date")
    assert new_p.iloc[1]["adj_close"] == 20.0, "within-tolerance mismatch must not be rescaled"
    print_check("adj_close reconciliation skips within-tolerance mismatch", True)
    return True


def test_duplicate_guard():
    # two old legs mapped onto the same surviving ticker with overlapping dates
    prices = pd.concat([
        _prices("LEGA3", ["2021-01-04"], [10.0]),
        _prices("LEGB3", ["2021-01-04"], [30.0]),
        _prices("SRV3", ["2021-02-01"], [11.0]),
    ], ignore_index=True)
    fund = _fund("SRV3", ["2021-03-31"], [1.0])
    path = _map([
        {"old": "LEGA3", "new": "SRV3", "date": "2021-02-01", "type": "merger", "ratio": 1.0},
        {"old": "LEGB3", "new": "SRV3", "date": "2021-02-01", "type": "merger", "ratio": 1.0},
    ])
    try:
        apply_ticker_continuity(prices, fund, path=path)
    except ValueError as e:
        assert "duplicate" in str(e), e
        print_check("duplicate guard", True)
        return True
    print_check("duplicate guard: bad map did not raise", False)
    return False


def test_old_last_date_drops_dead_stub_rows():
    # OLD3's real trading stops 01-04, but its raw feed keeps emitting
    # near-zero-volume noise through 01-06 before NEW3 starts trading 01-08
    # (mirrors CCRO3->MOTV3): without old_last_date, 01-05/01-06 would be
    # relabeled as NEW3 history instead of dropped.
    prices = pd.concat([
        _prices("OLD3", ["2021-01-04", "2021-01-05", "2021-01-06"], [10.0, 0.63, 0.77]),
        _prices("NEW3", ["2021-01-08", "2021-01-11"], [10.5, 10.6]),
    ], ignore_index=True)
    fund = pd.concat([
        _fund("OLD3", ["2020-12-31"], [100.0]),
        _fund("NEW3", ["2021-03-31"], [110.0]),
    ], ignore_index=True)
    path = _map([{"old": "OLD3", "new": "NEW3", "date": "2021-01-08",
                  "type": "rename", "ratio": 1.0, "old_last_date": "2021-01-04"}])

    p, _ = apply_ticker_continuity(prices, fund, path=path)

    new_p = p[p["ticker"] == "NEW3"].sort_values("trade_date")
    assert len(new_p) == 3, new_p  # OLD3's 01-04 + NEW3's own 2 rows; stub dropped, not relabeled
    assert list(new_p["close"]) == [10.0, 10.5, 10.6], "dead-stub rows (0.63, 0.77) must be dropped, not spliced"
    print_check("old_last_date drops dead-stub rows", True)
    return True


def test_missing_map_is_noop():
    prices = _prices("PETR4", ["2021-01-04"], [30.0])
    fund = _fund("PETR4", ["2020-12-31"], [1.0])
    p, f = apply_ticker_continuity(prices, fund, path=Path("/nonexistent/map.json"))
    assert len(p) == 1 and len(f) == 1
    print_check("missing map no-op", True)
    return True


def test_vendor_alias_rename_drops_duplicate():
    # Two files (vendor aliases) with identical closes, consolidated via rename.
    # The new ticker's file contains both entities' history under its own name.
    prices = pd.concat([
        _prices("OLD3", ["2021-01-04", "2021-01-05"], [10.0, 11.0]),
        _prices("NEW3", ["2021-01-04", "2021-01-05", "2021-02-01"], [10.0, 11.0, 12.0]),
    ], ignore_index=True)
    fund = _fund("NEW3", ["2020-12-31"], [100.0])
    path = _map([{"old": "OLD3", "new": "NEW3", "date": "2021-02-01",
                  "type": "rename", "ratio": 1.0}])

    p, _ = apply_ticker_continuity(prices, fund, path=path)

    assert "OLD3" not in p["ticker"], "vendor alias old leg must be dropped"
    new_p = p[p["ticker"] == "NEW3"].sort_values("trade_date")
    assert len(new_p) == 3, f"expected 3 rows (no dupes), got {len(new_p)}"
    assert list(new_p["close"]) == [10.0, 11.0, 12.0]
    assert not new_p.duplicated("trade_date").any(), "no duplicate dates after alias consolidation"
    print_check("vendor alias rename drops duplicate", True)
    return True


def test_keep_separate_ignores_merger():
    # Parallel-trading acquirer: both legs stay untouched, no splice.
    prices = pd.concat([
        _prices("ACQ3", ["2021-02-01", "2021-02-02"], [20.0, 22.0]),
        _prices("SRV3", ["2020-01-01", "2021-02-01", "2021-02-02"], [10.0, 11.0, 11.5]),
    ], ignore_index=True)
    fund = _fund("SRV3", ["2020-12-31"], [700.0])
    # SRV3 acquired by ACQ3, but ACQ3 traded before the acquisition — keep separate
    path = _map([{"old": "SRV3", "new": "ACQ3", "date": "2021-02-01",
                  "type": "keep_separate", "ratio": 0.5}])

    p, f = apply_ticker_continuity(prices, fund, path=path)

    acq = p[p["ticker"] == "ACQ3"].sort_values("trade_date")
    srv = p[p["ticker"] == "SRV3"].sort_values("trade_date")
    assert len(acq) == 2 and acq.iloc[0]["close"] == 20.0, "ACQ3 untouched"
    assert len(srv) == 3 and srv.iloc[0]["close"] == 10.0, "SRV3 untouched"
    assert not acq.duplicated("trade_date").any() and not srv.duplicated("trade_date").any()
    print_check("keep_separate ignores parallel-trading merger", True)
    return True


if __name__ == "__main__":
    ok = (test_rename_splice() & test_merger_splice()
          & test_adj_close_reconciliation() & test_adj_close_reconciliation_skips_within_tolerance()
          & test_old_last_date_drops_dead_stub_rows()
          & test_duplicate_guard() & test_missing_map_is_noop()
          & test_vendor_alias_rename_drops_duplicate() & test_keep_separate_ignores_merger())
    sys.exit(0 if ok else 1)
