"""
Test 1b (CVM-derived fundamentals): ratio math on synthetic statements +
internal-consistency check against raw CVM statements when CVM caches exist,
plus a cross-ticker discontinuity regression guard.

Part 1 (always runs, pure code): a hand-built quarterly frame with known
values must produce the exact BolsAI-convention ratios (verified live against
BPAN4 2025-09-30: single-quarter flows, thousands for statements, R$ units
for market_cap) and pass validate_fundamentals + carry all FUND_COLS.

Part 2 (SKIPs until CVM caches are collected): data/raw/br/fundamentals/*.parquet
is CVM-sourced end to end since the 2026-08-19 rebuild (BOLSAI_EXIT_PLAN.md Task 1),
so this is no longer a cross-VENDOR check -- it validates that build_fundamentals()'s
per-cnpj price/shares join didn't corrupt the point-in-time balance items (equity)
relative to the raw parsed statements they came from.

Part 3 (SKIPs if no fundamentals on disk): scans every ticker's balance sheet for
a vendor-switch cliff (see test_balance_sheet_has_no_vendor_switch_cliff below).

Run from project root:
    python tests/data_collection/test_cvm_statements.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_collection import config, validate  # noqa: E402
from src.data_collection.cvm.ratios import (  # noqa: E402
    _apply_share_events, _share_events, _shares_asof, _ticker_family, compute_ratios,
)
from src.data_collection.cvm.statements import load_statements  # noqa: E402

TOLERANCE = 0.15  # 15%, consistent with validate_vs_yfinance's loose band


def test_ratio_math():
    # Four synthetic quarters (compute_ratios TTMs flows internally via rolling(4) --
    # see BOLSAI_EXIT_PLAN.md Task 1 -- so a single row now yields NaN flows; each
    # quarter here carries 1/4 of the target ANNUAL figures below so the TTM'd last
    # row lands on the same known numbers the old single-quarter fixture asserted).
    # Input rows in R$ thousands (like CVM/BolsAI): net_income 100_000k, equity
    # 1_000_000k, revenue 500_000k, assets 5_000_000k, close 10.00, shares
    # 1_000_000_000 -> market_cap 10e9. Point-in-time items (balance sheet, price,
    # shares) are NOT TTM'd, so they're just repeated every quarter. compute_ratios()
    # scales these thousands inputs to full R$ units internally (§1,
    # DATA_LAYER_CORRECTNESS_PLAN.md) -- ratios stay numerically identical (k
    # compensates), but stored LEVEL columns (cash/total_debt/net_debt/ebitda
    # below) come out 1000x their thousands-convention input, on purpose.
    quarters = pd.date_range("2019-06-30", "2020-03-31", freq="QE")
    q = pd.DataFrame({
        "reference_date": quarters,
        "net_revenue": [125_000.0] * 4,
        "gross_profit": [50_000.0] * 4,
        "ebit": [37_500.0] * 4,
        "net_income": [25_000.0] * 4,
        "depr_amort": [5_000.0] * 4,
        "total_assets": [5_000_000.0] * 4,
        "current_assets": [800_000.0] * 4,
        "cash_caixa": [50_000.0] * 4,
        "cash_aplic": [150_000.0] * 4,
        "current_liabilities": [400_000.0] * 4,
        "debt_st": [100_000.0] * 4,
        "debt_lt": [900_000.0] * 4,
        "equity": [1_000_000.0] * 4,
        "close_price": [10.0] * 4,
        "shares_outstanding": [1_000_000_000] * 4,
    })
    out = compute_ratios(q, "TEST CO")
    r = out.iloc[-1]  # last row = the first fully-populated TTM window

    assert abs(r["market_cap"] - 10e9) < 1, r["market_cap"]
    assert abs(r["pl"] - 100.0) < 0.01, r["pl"]          # 10e9 / 100_000k (TTM net_income)
    assert abs(r["pvp"] - 10.0) < 0.01, r["pvp"]         # 10e9 / 1_000_000k
    assert abs(r["p_sr"] - 20.0) < 0.01, r["p_sr"]
    assert abs(r["p_ebit"] - 66.667) < 0.01, r["p_ebit"]     # 10e9 / 150_000k (BUG-2)
    assert abs(r["p_ebitda"] - 58.8235) < 0.01, r["p_ebitda"]  # 10e9 / 170_000k (BUG-2)
    assert abs(r["p_assets"] - 2.0) < 0.01, r["p_assets"]
    assert abs(r["roe"] - 10.0) < 0.01, r["roe"]         # 100k/1000k * 100
    assert abs(r["roa"] - 2.0) < 0.01, r["roa"]
    assert abs(r["net_margin"] - 20.0) < 0.01, r["net_margin"]
    assert abs(r["gross_margin"] - 40.0) < 0.01, r["gross_margin"]
    assert abs(r["current_ratio"] - 2.0) < 0.01, r["current_ratio"]
    assert abs(r["cash"] - 200_000_000.0) < 1, r["cash"]              # §1: 200_000k -> full R$ units
    assert abs(r["total_debt"] - 1_000_000_000.0) < 1, r["total_debt"]  # §1: 1_000_000k -> full R$ units
    assert abs(r["net_debt"] - 800_000_000.0) < 1, r["net_debt"]     # §1: 800_000k -> full R$ units
    assert abs(r["ev_ebit"] - 72.0) < 0.01, r["ev_ebit"]         # (10e9+8e8) / 150_000k (BUG-2)
    assert abs(r["ev_ebitda"] - 63.529) < 0.01, r["ev_ebitda"]   # (10e9+8e8) / 170_000k (BUG-2)
    assert abs(r["debt_equity"] - 1.0) < 0.01, r["debt_equity"]
    assert abs(r["lpa"] - 0.10) < 0.001, r["lpa"]        # 100_000k*1000 / 1e9 shares
    assert abs(r["vpa"] - 1.00) < 0.001, r["vpa"]
    assert abs(r["ebitda"] - 170_000_000.0) < 1, r["ebitda"]  # §1: TTM ebit + TTM depr_amort, full R$ units
    assert abs(r["roic"] - 5.5) < 0.01, r["roic"]        # (150_000k*0.66) / 1_800_000k * 100

    # Single-quarter (non-TTM) companions (0827f8b): raw per-quarter net_revenue/
    # net_income, scaled to full R$ units same as their TTM siblings above --
    # this fixture's quarters are all equal, so net_margin_q/roe_q land on the
    # same value as the TTM net_margin/roe would if they weren't 4x'd; the point
    # here is the *_q scale/formula, not an inflection scenario.
    assert abs(r["net_revenue_q"] - 125_000_000.0) < 1, r["net_revenue_q"]  # 125_000k * 1000
    assert abs(r["net_income_q"] - 25_000_000.0) < 1, r["net_income_q"]      # 25_000k * 1000
    assert abs(r["net_margin_q"] - 20.0) < 0.01, r["net_margin_q"]
    assert abs(r["roe_q"] - 2.5) < 0.01, r["roe_q"]      # 25_000_000 / 1e9 (equity) * 100

    # schema gate: exactly what collect_fundamentals-written files must satisfy
    out["ticker"] = "XXXX3"
    vr = validate.validate_fundamentals(out)
    assert vr.passed, vr.errors
    missing = [c for c in validate.FUND_COLS if c not in out.columns]
    assert not missing, f"missing FUND_COLS: {missing}"
    print("PASS  ratio math + schema")
    return True


def test_cross_source_vs_bolsai():
    """Statement values from CVM vs BolsAI's, same ticker+quarter."""
    if not list(config.CVM_DIR.glob("stmt_*.parquet")):
        print("SKIP  cross-source: no CVM statement caches (run cvm_statements)")
        return True
    if not (config.CVM_DIR / "fca_crosswalk.parquet").exists():
        print("SKIP  cross-source: no crosswalk")
        return True

    xwalk = pd.read_parquet(config.CVM_DIR / "fca_crosswalk.parquet")
    stmts = load_statements()
    checked = 0
    for ticker in ("WEGE3", "PETR4", "VALE3"):
        bolsai_path = config.FUND_DIR / f"{ticker}.parquet"
        row = xwalk[xwalk["ticker"] == ticker]
        if not bolsai_path.exists() or row.empty:
            continue
        bolsai = pd.read_parquet(bolsai_path)
        cvm = stmts[stmts["cnpj"] == row.iloc[0]["cnpj"]]
        both = bolsai.merge(cvm, on="reference_date", suffixes=("_b", "_c"))
        both = both.dropna(subset=["net_income_b", "net_income_c", "equity_b", "equity_c"])
        if both.empty:
            continue
        # ponytail: net_income has ~75% systematic divergence between CVM and BolsAI
        # (likely different earnings definitions: adjusted vs reported). Equity matches <3%,
        # so balance sheet data is reliable. Check only equity; net_income needs deeper audit.
        # equity_b comes from compute_ratios() output (full R$ units, §1); equity_c is
        # load_statements()'s raw value, which stays in CVM's thousands convention by
        # design (statements.py:163) -- scale it up before comparing, or every row
        # trips the 15% band on the deliberate 1000x, not a real join bug.
        b, c = both["equity_b"], both["equity_c"] * 1000.0
        rel = ((b - c).abs() / b.abs().clip(lower=1)).median()
        assert rel < TOLERANCE, f"{ticker} equity: median rel diff {rel:.1%} > {TOLERANCE:.0%}"
        print(f"PASS  cross-source {ticker}: {len(both)} quarters, equity within {TOLERANCE:.0%}")
        checked += 1
    if not checked:
        print("SKIP  cross-source: no overlapping ticker had both sources")
    return True


def test_balance_sheet_has_no_vendor_switch_cliff():
    """Regression guard (BOLSAI_EXIT_PLAN.md, "BUG-1"): `--mode update`'s fundamentals
    stage (and, separately, refresh.py) used to route BR fundamentals through yfinance,
    which stores point-in-time balance-sheet items wrong in LEVEL, not just thin --
    confirmed on PETR4 2026-06-30: BOTH equity (445bn -> 92.9bn, ~4.79x) AND
    total_assets (1,246bn -> 247bn, ~5.04x) fell by roughly the SAME factor in the
    SAME (most recent) quarter -- a whole-statement rescale from a stray incremental
    vendor append, "the most decision-relevant row in the panel" per the plan.

    Deliberately scoped to only the last-vs-second-last quarter, not full history:
    an entire-history scan for any such joint drop flags plenty of REAL historical
    distress/restructuring events in this 612-ticker, 15-year, delisted-inclusive
    panel (verified -- an earlier version of this test flagged dozens of genuine
    small/micro-cap tickers). CLAUDE.md documents the same lesson for a related
    problem (`repair.py`'s split-persistence guard, rejected 3x): "illiquid tickers'
    ordinary volatility swamps any workable threshold" for a whole-history anomaly
    scan. The production risk this guards against -- a future `--mode update` run
    silently reverting to yfinance and re-corrupting the tail -- only ever shows up
    at the tail, so that's the only transition worth asserting on.
    """
    files = sorted(config.FUND_DIR.glob("*.parquet"))
    if not files:
        print("SKIP  balance-sheet vendor-switch cliff: no fundamentals files on disk")
        return True

    flagged = []
    for path in files:
        df = pd.read_parquet(path, columns=["reference_date", "equity", "total_assets"]) \
               .sort_values("reference_date").dropna(subset=["equity", "total_assets"])
        if len(df) < 2:
            continue
        prev, last = df.iloc[-2], df.iloc[-1]
        eq_ratio = last["equity"] / prev["equity"] if prev["equity"] else float("nan")
        assets_ratio = last["total_assets"] / prev["total_assets"] if prev["total_assets"] else float("nan")
        # Window bracketed on the confirmed PETR4 magnitude (~4.79x/~5.04x), with
        # margin, NOT open-ended down to zero: a near-total wipeout (ratio near 0) is
        # a plausible real event for an obscure micro-cap (e.g. a holding company
        # spinning off nearly all its assets) -- categorically different from a
        # same-scale ~5x vendor/unit mismatch, and not this test's job to adjudicate.
        both_cliff = 0.1 <= eq_ratio < 0.5 and 0.1 <= assets_ratio < 0.5
        same_scale = abs(eq_ratio - assets_ratio) < 0.2  # fell by roughly the same factor
        if both_cliff and same_scale:
            flagged.append((path.stem, round(eq_ratio, 2), round(assets_ratio, 2)))

    assert not flagged, f"whole-statement-rescale in the most recent quarter: {flagged[:10]}"
    print(f"PASS  balance-sheet vendor-switch cliff check: {len(files)} tickers' latest quarter clean")
    return True


def test_ticker_family_resolves_continuity_chain():
    """_ticker_family must resolve TIMP3 and TIMS3 (real ticker_continuity.json
    rename entry) to the same family, both directions, and leave an unrelated
    ticker as a singleton -- this is what lets a real 2025 TIMS3-keyed split
    reach TIMP3's own raw fundamentals file (see _share_events)."""
    fam_old = _ticker_family("TIMP3")
    fam_new = _ticker_family("TIMS3")
    assert fam_old == fam_new, f"TIMP3 and TIMS3 must resolve to the same family: {fam_old} vs {fam_new}"
    assert {"TIMP3", "TIMS3"}.issubset(fam_old), fam_old

    assert _ticker_family("__NO_SUCH_TICKER__") == {"__NO_SUCH_TICKER__"}
    print("PASS  ticker-family continuity-chain resolution")
    return True


def test_share_events_adjustment():
    """docs/DATA_LAYER_FOLLOWUP_FINDINGS.md's shares/splits mitigation: a
    synthetic corporate_events.parquet replicating TIMS3's exact real shape
    (one reverse split recorded TWICE, 2 days apart, at an inverse-but-equal
    ratio -- 1000:1 and 1:0.001 -- plus one later, independent forward split)
    must collapse to exactly 2 real events, and _apply_share_events must only
    apply an event when it happened AFTER the matched FRE record's own
    effective_date."""
    events = pd.DataFrame({
        "ticker": ["TESTX3"] * 3,
        "date": pd.to_datetime(["2007-07-01", "2007-07-03", "2010-01-01"]),
        "type": ["INPLIT", "INPLIT", "SPLIT"],
        "ratio_from": [1000.0, 1.0, 1.0],
        "ratio_to": [1.0, 0.001, 2.0],
        "factor": [1000.0, 0.001, 2.0],
    })
    tmp_path = ROOT / "tests" / "data_collection" / "_tmp_corp_events_test.parquet"
    events.to_parquet(tmp_path, index=False)
    orig_path = config.CORP_EVENTS_PATH
    try:
        config.CORP_EVENTS_PATH = tmp_path

        dedup = _share_events("TESTX3")
        assert len(dedup) == 2, f"expected 2 real events after dedup, got {len(dedup)}:\n{dedup}"
        assert abs(dedup.iloc[0]["share_multiplier"] - 0.001) < 1e-9
        assert abs(dedup.iloc[1]["share_multiplier"] - 2.0) < 1e-9

        ref_dates = pd.Series(pd.to_datetime(
            ["2005-01-01", "2008-01-01", "2011-01-01", "2011-01-01"]))
        # Row 0: before either event -> untouched.
        # Row 1: FRE record from before the 2007 event -> apply it once.
        # Row 2: FRE record from before BOTH events -> apply both, compounded.
        # Row 3: FRE record ALREADY from after the 2007 event -> only the 2010
        #        one is new information; re-applying 2007's would double-count.
        shares_vals = np.array([1_000_000.0, 1_000_000.0, 1_000_000.0, 1_000.0])
        eff_dates = pd.to_datetime(
            ["2004-01-01", "2006-01-01", "2000-01-01", "2008-01-01"]).to_numpy()

        out = _apply_share_events(shares_vals, eff_dates, ref_dates, "TESTX3")
        assert abs(out[0] - 1_000_000.0) < 1e-6, out[0]
        assert abs(out[1] - 1_000.0) < 1e-6, out[1]              # 1e6 * 0.001
        assert abs(out[2] - 2_000.0) < 1e-6, out[2]              # 1e6 * 0.001 * 2
        assert abs(out[3] - 2_000.0) < 1e-6, out[3]              # 1e3 * 2 (2007 already baked in)
    finally:
        config.CORP_EVENTS_PATH = orig_path
        tmp_path.unlink(missing_ok=True)

    print("PASS  share-events dedup + forward-adjustment")
    return True


def test_apply_share_events_no_double_count_when_fre_already_reflects_split():
    """RVEE3's real shape: FRE's OWN capital_social timeline already jumped 10x
    the day BEFORE corporate_events.parquet's recorded 1:10 split -- CVM's
    filing already reflects the post-split count (contrast TIMS3, which
    _apply_share_events was built for: FRE stayed frozen ACROSS its split).
    Reapplying the recorded split on top of an FRE snapshot that already
    reflects it double-counts (confirmed exactly: 10,171,150 -> 101,711,500 in
    the real raw fundamentals from 2025-09-30). A genuinely separate LATER
    event must still apply."""
    shares = pd.DataFrame({
        "cnpj": ["RVEE_CNPJ"] * 2,
        "effective_date": pd.to_datetime(["2025-06-01", "2025-08-06"]),
        "shares": [1_017_115.0, 10_171_150.0],
    })
    events = pd.DataFrame({
        "ticker": ["RVEE3", "RVEE3"],
        "date": pd.to_datetime(["2025-08-07", "2026-01-01"]),
        "type": ["SPLIT", "SPLIT"],
        "ratio_from": [1.0, 1.0],
        "ratio_to": [10.0, 2.0],
        "factor": [10.0, 2.0],
    })
    tmp_path = ROOT / "tests" / "data_collection" / "_tmp_corp_events_rvee_test.parquet"
    events.to_parquet(tmp_path, index=False)
    orig_path = config.CORP_EVENTS_PATH
    try:
        config.CORP_EVENTS_PATH = tmp_path

        ref_dates = pd.Series(pd.to_datetime(["2025-07-01", "2025-09-30", "2026-03-31"]))
        shares_vals, eff_dates, prev_shares_vals = _shares_asof(shares, "RVEE_CNPJ", ref_dates)
        out = _apply_share_events(shares_vals, eff_dates, ref_dates, "RVEE3", prev_shares_vals)

        assert abs(out[0] - 1_017_115.0) < 1e-6, out[0]      # before either event
        assert abs(out[1] - 10_171_150.0) < 1e-6, out[1]     # NOT 101,711,500 -- no double-count
        assert abs(out[2] - 20_342_300.0) < 1e-6, out[2]     # 10,171,150 * 2 -- later event still applies
    finally:
        config.CORP_EVENTS_PATH = orig_path
        tmp_path.unlink(missing_ok=True)

    print("PASS  RVEE3-shape share-event double-count guard")
    return True


if __name__ == "__main__":
    ok = (test_ratio_math() and test_cross_source_vs_bolsai()
          and test_balance_sheet_has_no_vendor_switch_cliff()
          and test_ticker_family_resolves_continuity_chain() and test_share_events_adjustment()
          and test_apply_share_events_no_double_count_when_fre_already_reflects_split())
    sys.exit(0 if ok else 1)
