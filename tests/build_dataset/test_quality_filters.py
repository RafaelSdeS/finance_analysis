#!/usr/bin/env python3
"""
Coverage and filing-lag gates: filter_tickers_with_no_fundamentals,
attach_filing_dates (real CVM DT_RECEB vs statutory fallback), and
filter_excessive_filing_lag. Mirrors src/build_dataset/quality_filters.py.

Previously only a constant (FILING_LAG_DAYS_QUARTERLY) was imported
elsewhere for unrelated arithmetic -- none of these three functions had a
dedicated test, despite being the anti-lookahead machinery CLAUDE.md
repeatedly calls load-bearing.

Run from project root: python tests/build_dataset/test_quality_filters.py
or: pytest tests/build_dataset/test_quality_filters.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_dataset import quality_filters as qf


def test_statutory_available_date_quarterly_vs_annual() -> None:
    """Non-December quarter-ends get the 45-day ITR buffer; December
    (annual/DFP filing) gets the wider 90-day buffer."""
    dates = pd.Series(pd.to_datetime(["2026-03-31", "2026-06-30", "2026-12-31"]))

    result = qf._statutory_available_date(dates)

    assert result.iloc[0] == pd.Timestamp("2026-03-31") + pd.Timedelta(days=45)
    assert result.iloc[1] == pd.Timestamp("2026-06-30") + pd.Timedelta(days=45)
    assert result.iloc[2] == pd.Timestamp("2026-12-31") + pd.Timedelta(days=90)


def test_filter_tickers_with_no_fundamentals() -> None:
    """Drops quarantined tickers, tickers with zero fundamental rows, and
    tickers with too little price history -- keeps everything else."""
    prices = pd.concat([
        pd.DataFrame({"ticker": "WDCN3", "trade_date": pd.date_range("2026-01-01", periods=20)}),
        pd.DataFrame({"ticker": "NOFUND", "trade_date": pd.date_range("2026-01-01", periods=20)}),
        pd.DataFrame({"ticker": "SHORT", "trade_date": pd.date_range("2026-01-01", periods=5)}),
        pd.DataFrame({"ticker": "GOOD", "trade_date": pd.date_range("2026-01-01", periods=20)}),
    ], ignore_index=True)
    fundamentals = pd.DataFrame({
        "ticker": ["GOOD", "SHORT"],
        "reference_date": pd.to_datetime(["2026-03-31", "2026-03-31"]),
    })

    result = qf.filter_tickers_with_no_fundamentals(prices, fundamentals)

    assert set(result["ticker"].unique()) == {"GOOD"}, (
        "WDCN3 quarantined, NOFUND has no fundamentals, SHORT has <MIN_PRICE_ROWS rows"
    )


def test_filter_tickers_with_no_fundamentals_classifies_exclusions(capsys) -> None:
    """The no-fundamentals report splits exclusions into: known non-company
    (BOVA11-style), delisted/renamed (stale prices), redundant (a sibling
    share class already has fundamentals), and unexplained GAP -- so a real
    coverage hole (like BTG Pactual's BPAC11 never being collected) doesn't
    get silently lumped in with routine, safe exclusions."""
    prices = pd.concat([
        # GOOD has fundamentals and sets the dataset's "current" date
        pd.DataFrame({"ticker": "GOOD", "trade_date": pd.date_range("2026-01-01", periods=20)}),
        # ROOT is GOOD's covered sibling class (e.g. GOOD3 vs GOOD11) — redundant
        pd.DataFrame({"ticker": "GOOD4", "trade_date": pd.date_range("2026-01-01", periods=20)}),
        # DEAD stopped trading years before the dataset's last date
        pd.DataFrame({"ticker": "DEAD3", "trade_date": pd.date_range("2015-01-01", periods=20)}),
        # ETF is in KNOWN_NO_FUNDAMENTALS
        pd.DataFrame({"ticker": "BOVA11", "trade_date": pd.date_range("2026-01-01", periods=20)}),
        # GAP trades recently, has no covered sibling, isn't known -- a real hole
        pd.DataFrame({"ticker": "GAP3", "trade_date": pd.date_range("2026-01-01", periods=20)}),
    ], ignore_index=True)
    fundamentals = pd.DataFrame({
        "ticker": ["GOOD"],
        "reference_date": pd.to_datetime(["2026-03-31"]),
    })

    result = qf.filter_tickers_with_no_fundamentals(prices, fundamentals)
    out = capsys.readouterr().out

    assert set(result["ticker"].unique()) == {"GOOD"}
    assert "BOVA11: benchmark ETF" in out
    assert "delisted/renamed" in out and "['DEAD3']" in out
    assert "redundant" in out and "GOOD4 -> GOOD" in out
    assert "⚠ GAP" in out and "['GAP3']" in out


def test_attach_filing_dates_uses_real_cvm_date(tmp_path, monkeypatch) -> None:
    """When a (cnpj, quarter) pair exists in filing_dates.parquet, its real
    received_date is used, not the statutory fallback."""
    filing_dates_path = tmp_path / "filing_dates.parquet"
    pd.DataFrame({
        "cnpj": ["11111111000101"],
        "reference_date": pd.to_datetime(["2026-03-31"]),
        "received_date": pd.to_datetime(["2026-04-20"]),  # 20d, well inside the 45d statutory buffer
    }).to_parquet(filing_dates_path)
    monkeypatch.setattr(qf, "FILING_DATES_PATH", filing_dates_path)

    company_info = pd.DataFrame({"ticker": ["A"], "cnpj": ["11.111.111/0001-01"]})
    fundamentals = pd.DataFrame({"ticker": ["A"], "reference_date": pd.to_datetime(["2026-03-31"])})

    result = qf.attach_filing_dates(fundamentals, company_info)

    assert result.iloc[0]["fundamentals_available_date"] == pd.Timestamp("2026-04-20")
    assert result.iloc[0]["filing_lag_days"] == 20


def test_attach_filing_dates_falls_back_to_statutory_for_missing_quarter(tmp_path, monkeypatch) -> None:
    """A ticker/quarter absent from the CVM register gets the statutory
    deadline instead, not a missing/NaT availability date."""
    filing_dates_path = tmp_path / "filing_dates.parquet"
    pd.DataFrame({
        "cnpj": ["11111111000101"],
        "reference_date": pd.to_datetime(["2025-12-31"]),  # different quarter than fundamentals below
        "received_date": pd.to_datetime(["2026-02-01"]),
    }).to_parquet(filing_dates_path)
    monkeypatch.setattr(qf, "FILING_DATES_PATH", filing_dates_path)

    company_info = pd.DataFrame({"ticker": ["A"], "cnpj": ["11.111.111/0001-01"]})
    fundamentals = pd.DataFrame({"ticker": ["A"], "reference_date": pd.to_datetime(["2026-03-31"])})

    result = qf.attach_filing_dates(fundamentals, company_info)

    assert result.iloc[0]["fundamentals_available_date"] == (
        pd.Timestamp("2026-03-31") + pd.Timedelta(days=45)
    )
    assert pd.isna(result.iloc[0]["filing_lag_days"])


def test_attach_filing_dates_rejects_received_date_before_quarter_end(tmp_path, monkeypatch) -> None:
    """A filing can't precede its own quarter-end -- such a (data-error) row
    must be treated as unknown and fall back to the statutory deadline,
    never accepted as a too-good-to-be-true early filing."""
    filing_dates_path = tmp_path / "filing_dates.parquet"
    pd.DataFrame({
        "cnpj": ["11111111000101"],
        "reference_date": pd.to_datetime(["2026-03-31"]),
        "received_date": pd.to_datetime(["2026-01-15"]),  # before the quarter it reports on
    }).to_parquet(filing_dates_path)
    monkeypatch.setattr(qf, "FILING_DATES_PATH", filing_dates_path)

    company_info = pd.DataFrame({"ticker": ["A"], "cnpj": ["11.111.111/0001-01"]})
    fundamentals = pd.DataFrame({"ticker": ["A"], "reference_date": pd.to_datetime(["2026-03-31"])})

    result = qf.attach_filing_dates(fundamentals, company_info)

    assert result.iloc[0]["fundamentals_available_date"] == (
        pd.Timestamp("2026-03-31") + pd.Timedelta(days=45)
    )


def test_attach_filing_dates_no_file_uses_statutory_only(tmp_path, monkeypatch) -> None:
    """No filing_dates.parquet on disk at all -- every row gets the statutory
    fallback, no crash."""
    monkeypatch.setattr(qf, "FILING_DATES_PATH", tmp_path / "does_not_exist.parquet")

    company_info = pd.DataFrame({"ticker": ["A"], "cnpj": ["11.111.111/0001-01"]})
    fundamentals = pd.DataFrame({
        "ticker": ["A", "A"],
        "reference_date": pd.to_datetime(["2026-03-31", "2026-12-31"]),
    })

    result = qf.attach_filing_dates(fundamentals, company_info)

    assert result.iloc[0]["fundamentals_available_date"] == pd.Timestamp("2026-03-31") + pd.Timedelta(days=45)
    assert result.iloc[1]["fundamentals_available_date"] == pd.Timestamp("2026-12-31") + pd.Timedelta(days=90)
    assert "filing_lag_days" not in result.columns


def test_drop_orphan_prefix_rows() -> None:
    """Rows before ORPHAN_PREFIX_TICKERS[ticker]['drop_before'] are removed;
    everything else (other tickers, later rows of the same ticker) is
    untouched."""
    prices = pd.concat([
        pd.DataFrame({"ticker": "BRDT3", "trade_date": pd.to_datetime(
            ["2001-12-03", "2003-01-07", "2017-12-15", "2017-12-18"]
        )}),
        pd.DataFrame({"ticker": "OTHER3", "trade_date": pd.to_datetime(["2001-12-03"])}),
    ], ignore_index=True)

    result = qf.drop_orphan_prefix_rows(prices)

    brdt3_dates = set(result.loc[result["ticker"] == "BRDT3", "trade_date"])
    assert brdt3_dates == {pd.Timestamp("2017-12-15"), pd.Timestamp("2017-12-18")}
    assert (result["ticker"] == "OTHER3").sum() == 1, "unrelated ticker must be untouched"


def test_filter_excessive_filing_lag_drops_only_over_threshold() -> None:
    """Rows filed more than max_lag_days late are dropped; rows within the
    threshold and rows with unknown (NaN) lag -- statutory fallback, no real
    CVM date -- are both kept."""
    fundamentals = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "filing_lag_days": [10.0, 200.0, float("nan"), 180.0],
    })

    result = qf.filter_excessive_filing_lag(fundamentals, max_lag_days=180)

    assert set(result["ticker"]) == {"A", "C", "D"}, "only B (200d > 180d threshold) must be dropped"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
