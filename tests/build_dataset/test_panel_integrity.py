"""
test_panel_integrity.py
=======================
Spec: invariants that must hold on a built panel regardless of market, checked
against facts the row cannot manufacture for itself.

Companion to test_instrument_basis.py. That file checks whether a row's price
and share count describe the same instrument; this one checks the panel's own
internal consistency -- across sibling tickers, across time, and between a
derived column and the raw column it is derived from.

Deliberately a mix of RED and GREEN. The green ones are not filler: each is a
real property that was measured, not assumed, and several of them are the only
thing standing between a silent pipeline regression and a retrained model. They
are marked with what they measured so a future reader can tell "checked, holds"
from "never looked".

Measured 2026-08-23 (BR dataset_v7 1,718,263 rows / 561 tickers;
us_ml_dataset.parquet 15,353,292 rows / 2,903 tickers):

  RED   company-level facts disagree across sibling tickers   BR 2,571 + 242 groups
  RED   equity changes by >1000x between adjacent quarters    BR 23, US 184
  green no duplicate (ticker, trade_date)                     BR 0, US 0
  green no lookahead in the two date columns                  BR 0, US 0
  green log_return reconstructs from adj_close                BR 0/1.7M, US 0/15.3M
  green adjusted OHLC brackets hold                           BR 0, US 0 (see note)
  green every non-positive adj_close is flagged               BR 288/288 flagged
  green percentile columns stay inside [0, 1]                 BR 0, US 0

NOTE on the OHLC check. At exact comparison it reports 27,431 BR and 96,183 US
"violations" -- every one of them is float noise from the adjustment multiply
(a bar that closed at its high has adj_close and adj_high differing in the last
ulp). At rtol 1e-9 and looser: zero, both markets, every threshold tested up to
1e-3. The tolerance is the check; an exact comparison here is a false-positive
generator, not a stricter test.

DATA group: needs a built panel.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_dataset.paths import OUTPUT_PATH, US_OUTPUT_PATH  # noqa: E402

MARKETS = [("BR", OUTPUT_PATH, "cnpj"), ("US", US_OUTPUT_PATH, "cik")]
_M = [(m, p) for m, p, _ in MARKETS]

# Adjusted OHLC bracket tolerance, relative to the bar's own high. See the
# module docstring: below 1e-9 this measures float noise, not data.
_OHLC_RTOL = 1e-9

# log_return must reproduce log(adj_close).diff() exactly enough that no
# downstream feature can drift from the price it claims to describe.
_RETURN_ATOL = 1e-6

# Adjacent-quarter equity ratio that no real corporate event produces. A 5x
# quarter is a large capital raise or writedown and happens (BR 1.03% of
# quarters, US 1.59%); 10x is rare but arguable (0.48% / 0.92%); 1000x is not
# a company doing anything, it is a broken figure. Zero tolerance at 1000x
# only -- the softer bands are printed as context, not asserted, because
# separating a real recapitalisation from a bad filing needs an event source
# the panel does not carry.
_EQUITY_MAX_QOQ_RATIO = 1_000.0
_EQUITY_CONTEXT_BANDS = (5, 10, 50, 100)


def _load(path: Path, cols: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    import pyarrow.parquet as pq
    have = set(pq.read_schema(path).names)
    missing = [c for c in cols if c not in have]
    if missing:
        pytest.skip(f"{path.name} lacks {missing}")
    return pd.read_parquet(path, columns=cols)


@pytest.mark.parametrize("market,path", _M)
def test_no_duplicate_rows(market, path):
    """One row per ticker per trading day. Measured: 0 both markets."""
    df = _load(path, ["ticker", "trade_date"])
    if df is None:
        pytest.skip(f"{market} panel not built")
    dup = int(df.duplicated(["ticker", "trade_date"]).sum())
    print(f"  {market}: {dup} duplicate (ticker, trade_date) rows of {len(df):,}")
    assert dup == 0, f"{market}: {dup} duplicate (ticker, trade_date) rows"


@pytest.mark.parametrize("market,path", _M)
def test_no_lookahead_in_date_columns(market, path):
    """A filing cannot be known before it was filed, or filed before its period ends.

    The whole no-lookahead claim rests on these two orderings surviving the
    merge. Measured: 0 violations of either, both markets.
    """
    df = _load(path, ["ticker", "trade_date", "reference_date", "fundamentals_available_date"])
    if df is None:
        pytest.skip(f"{market} panel not built")
    future = int((df["fundamentals_available_date"] > df["trade_date"]).sum())
    backwards = int((df["reference_date"] > df["fundamentals_available_date"]).sum())
    print(f"  {market}: available_date > trade_date = {future}, "
          f"reference_date > available_date = {backwards}")
    assert future == 0, f"{market}: {future} rows use a filing not yet available"
    assert backwards == 0, f"{market}: {backwards} filings predate the period they report"


@pytest.mark.parametrize("market,path", _M)
def test_log_return_reconstructs_from_adj_close(market, path):
    """The derived return column must still match the price column it came from.

    Cheap, and the only check that would catch a feature pipeline silently
    computing returns off a stale or differently-adjusted price series.
    Measured: 0 mismatches of 1,716,195 BR / 15,349,321 US.
    """
    df = _load(path, ["ticker", "trade_date", "adj_close", "log_return"])
    if df is None:
        pytest.skip(f"{market} panel not built")
    df = df.sort_values(["ticker", "trade_date"])
    rebuilt = np.log(df["adj_close"] / df.groupby("ticker")["adj_close"].shift(1))
    comparable = df["log_return"].notna() & rebuilt.notna()
    off = int(((df["log_return"] - rebuilt).abs()[comparable] > _RETURN_ATOL).sum())
    print(f"  {market}: {off} of {int(comparable.sum()):,} comparable rows disagree")
    assert off == 0, f"{market}: log_return disagrees with log(adj_close).diff() on {off} rows"


@pytest.mark.parametrize("market,path", _M)
def test_adjusted_ohlc_brackets_hold(market, path):
    """low <= open/close <= high on the adjusted series, within float tolerance."""
    df = _load(path, ["ticker", "adj_open", "adj_high", "adj_low", "adj_close"])
    if df is None:
        pytest.skip(f"{market} panel not built")
    o, h, l, c = (df[f"adj_{k}"].to_numpy(float) for k in ("open", "high", "low", "close"))
    tol = _OHLC_RTOL * np.maximum(np.abs(h), 1e-12)
    bad = (h < l - tol) | (c > h + tol) | (c < l - tol) | (o > h + tol) | (o < l - tol)
    bad = np.where(np.isnan(o + h + l + c), False, bad)
    n = int(bad.sum())
    print(f"  {market}: {n} bracket violations at rtol={_OHLC_RTOL:g} "
          f"({int(((h < l) | (c > h) | (c < l) | (o > h) | (o < l)).sum()):,} at exact equality, "
          f"all float noise)")
    assert n == 0, f"{market}: {n} adjusted OHLC bracket violations"


@pytest.mark.parametrize("market,path", _M)
def test_percentile_columns_stay_in_range(market, path):
    """Every *_percentile column is a percentile. Measured: 0 out of range."""
    import pyarrow.parquet as pq
    if not path.exists():
        pytest.skip(f"{market} panel not built")
    cols = [c for c in pq.read_schema(path).names if c.endswith("_percentile")]
    if not cols:
        pytest.skip(f"{market}: no percentile columns")
    df = pd.read_parquet(path, columns=cols)
    offenders = {c: int(((df[c] < 0) | (df[c] > 1)).sum()) for c in cols}
    offenders = {c: n for c, n in offenders.items() if n}
    print(f"  {market}: {len(cols)} percentile columns, {len(offenders)} out of range")
    assert not offenders, f"{market}: percentile columns outside [0, 1]: {offenders}"


def test_nonpositive_adj_close_is_always_flagged():
    """BR's documented 2-decimal underflow class must be flagged, never silent.

    CLAUDE.md accepts that a few deep-history microcaps underflow BolsAI's
    2-decimal adj_close floor; what makes that acceptable rather than a data
    bug is `adj_close_precision_degraded` marking every such row so a consumer
    can exclude them. That flag is the contract. Measured: 288 rows, all
    LUXM4, 288 flagged, 0 silent.
    """
    df = _load(OUTPUT_PATH, ["ticker", "trade_date", "adj_close", "adj_close_precision_degraded"])
    if df is None:
        pytest.skip("BR panel not built")
    bad = df[df["adj_close"] <= 0]
    silent = bad[bad["adj_close_precision_degraded"].fillna(0) == 0]
    print(f"  BR: {len(bad)} non-positive adj_close rows across "
          f"{bad['ticker'].nunique()} tickers ({sorted(bad['ticker'].unique())}), "
          f"{len(silent)} unflagged")
    assert silent.empty, (
        f"BR: {len(silent)} rows have adj_close <= 0 without "
        f"adj_close_precision_degraded set: {sorted(silent['ticker'].unique())}")


@pytest.mark.parametrize("market,path,key", MARKETS)
def test_company_level_facts_agree_across_siblings(market, path, key):
    """Two tickers of the SAME company on the SAME day must report one set of
    company-level facts.

    `equity`, `net_income` and `shares_outstanding` describe the issuer, not
    the share class -- ITUB3 and ITUB4 are one balance sheet. A disagreement
    means the fundamentals join produced two different answers for one company
    on one date, and at least one of them is wrong.

    This is the strongest available check on the join because the contradiction
    is internal: no external price, no plausibility judgement, no threshold to
    argue about.

    Measured 2026-08-23:
      US  0 / 81,337 equity groups, 0 / 30,968 share-count groups.  Clean.
      BR  242 / 41,918 equity groups (all one company, AXIA3/AXIA5/AXIA6,
          max ratio 1.34x) and 2,571 / 41,276 share-count groups -- whose
          MEDIAN disagreement ratio is exactly 2.00x, with the 90th percentile
          at 500x. The 2x median is the unit/share-class basis of
          test_instrument_basis.py showing up again from a completely
          different direction; the 500x tail is the cvm/shares.py scale defect.

    BR cnpj is normalised to bare digits first -- the panel stores 306 of 694
    punctuated ("42.771.949/0001-35") and 388 bare, so grouping on the raw
    string splits real companies in two and UNDER-counts the disagreement.
    """
    if not path.exists():
        pytest.skip(f"{market} panel not built")
    import pyarrow.parquet as pq
    if key not in set(pq.read_schema(path).names):
        pytest.skip(f"{market} panel has no `{key}` column")

    cols = ["ticker", "trade_date", key, "equity", "net_income", "shares_outstanding"]
    df = _load(path, cols)
    df = df.dropna(subset=[key])
    if key == "cnpj":
        df[key] = df[key].astype(str).str.replace(r"\D", "", regex=True).str.zfill(14)

    failures = {}
    for col in ("equity", "net_income", "shares_outstanding"):
        d = df.dropna(subset=[col])
        agg = d.groupby([key, "trade_date"])[col].agg(["nunique", "count", "min", "max"])
        shared = agg[agg["count"] > 1]
        if shared.empty:
            print(f"  {market} {col}: no multi-ticker company-days to compare")
            continue
        disagree = shared[shared["nunique"] > 1]
        ratio = (disagree["max"] / disagree["min"]).replace([np.inf, -np.inf], np.nan).dropna()
        med = float(ratio.median()) if len(ratio) else float("nan")
        worst = float(ratio.max()) if len(ratio) else float("nan")
        print(f"  {market} {col}: {len(disagree):,} of {len(shared):,} company-days disagree"
              + (f" (median {med:.2f}x, worst {worst:,.0f}x)" if len(ratio) else ""))
        if len(disagree):
            names = sorted(disagree.reset_index()[key].unique())
            failures[col] = (len(disagree), len(shared), med, worst, names[:5])

    assert not failures, (
        f"{market}: sibling tickers of the same company report different "
        f"company-level facts -- {failures}")


@pytest.mark.parametrize("market,path", _M)
def test_equity_has_no_impossible_quarter_jumps(market, path):
    """Book equity cannot change by 1000x between adjacent quarters.

    This is the BUG-1 shape CLAUDE.md documents for yfinance BR fundamentals
    ("point-in-time balance-sheet items dropping ~5x in a single quarter with
    no real event behind it"), checked here on what actually reached the panel
    from CVM and SEC.

    Only the 1000x line is asserted. Real recapitalisations, reverse mergers
    and going-concern writedowns genuinely produce 5-10x quarters, and the
    panel carries no corporate-event source that could separate those from bad
    filings -- so the softer bands are printed for context and left unasserted
    rather than turned into a threshold nobody can defend.
    Measured: BR 23 / 23,038 quarters beyond 1000x, US 184 / 111,947.
    """
    df = _load(path, ["ticker", "reference_date", "equity"])
    if df is None:
        pytest.skip(f"{market} panel not built")
    q = (df.dropna(subset=["equity"])
           .drop_duplicates(["ticker", "reference_date"])
           .sort_values(["ticker", "reference_date"]))
    q = q[q["equity"] != 0]
    ratio = q.groupby("ticker")["equity"].apply(lambda s: (s / s.shift(1)).abs())
    ratio = ratio[np.isfinite(ratio)]
    for band in _EQUITY_CONTEXT_BANDS:
        n = int(((ratio > band) | (ratio < 1 / band)).sum())
        print(f"  {market}: {n:,} of {len(ratio):,} quarter-pairs beyond {band}x "
              f"({100 * n / max(len(ratio), 1):.3f}%)  [context, not asserted]")
    beyond = ratio[(ratio > _EQUITY_MAX_QOQ_RATIO) | (ratio < 1 / _EQUITY_MAX_QOQ_RATIO)]
    tickers = sorted(set(beyond.index.get_level_values(0))) if len(beyond) else []
    print(f"  {market}: {len(beyond)} beyond {_EQUITY_MAX_QOQ_RATIO:.0f}x across "
          f"{len(tickers)} tickers -> {tickers[:15]}")
    assert beyond.empty, (
        f"{market}: {len(beyond)} adjacent quarters change book equity by more than "
        f"{_EQUITY_MAX_QOQ_RATIO:.0f}x across {len(tickers)} tickers: {tickers[:15]}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
