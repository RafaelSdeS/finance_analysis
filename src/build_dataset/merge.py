"""merge.py — join prices with fundamentals (no lookahead), company info,
macro series, and dividends into one daily panel."""

import numpy as np
import pandas as pd

from .paths import CVM_CROSSWALK_PATH, MACRO_DIR
from .quality_filters import _statutory_available_date

# Tickers absent from company_info (no sibling either) get status inferred from
# price recency: traded within this many days of the dataset's last date = ATIVO.
STATUS_INFERENCE_WINDOW_DAYS = 180


# =============================================================================
# MERGE DAILY PRICES + QUARTERLY FUNDAMENTALS
# =============================================================================

def merge_prices_and_fundamentals(prices, fundamentals):

    print()
    print("=" * 80)
    print("MERGING PRICES + FUNDAMENTALS")
    print("=" * 80)

    merged_dfs = []

    # Split once via groupby instead of re-scanning the full frame per ticker
    # (was O(tickers * total_rows) of repeated boolean filtering).
    prices_by_ticker = dict(tuple(prices.groupby("ticker", sort=False)))
    fundamentals_by_ticker = dict(tuple(fundamentals.groupby("ticker", sort=False)))

    for ticker in sorted(prices["ticker"].unique()):

        print(f"Merging {ticker}")

        p = prices_by_ticker[ticker].sort_values("trade_date")

        f = fundamentals_by_ticker.get(ticker, fundamentals.iloc[0:0]).copy()
        f = f.sort_values("reference_date")

        # attach_filing_dates() normally sets this (real CVM receipt date with
        # statutory fallback); synthesize the fallback if called standalone
        if "fundamentals_available_date" not in f.columns:
            f["fundamentals_available_date"] = _statutory_available_date(f["reference_date"])
        f = f.sort_values("fundamentals_available_date")

        # merge_asof: uses the most recent fundamental whose filing date has
        # already passed as of each trade_date (no lookahead bias).
        merged = pd.merge_asof(
            p,
            f,
            left_on="trade_date",
            right_on="fundamentals_available_date",
            by="ticker",
            direction="backward",
        )

        # Replace close_price with actual price at fundamentals_available_date
        # (BolsAI's close_price is from reference_date, 45-90 days earlier; comparing
        # it to today's close gives false >50% jumps; use real close at filing instead).
        # Vectorized asof lookup (was a per-row full-frame scan — the main build bottleneck,
        # O(rows^2) per ticker).
        if "fundamentals_available_date" in merged.columns:
            has_filing = merged["fundamentals_available_date"].notna()
            if has_filing.any():
                filing_dates = merged.loc[has_filing, ["fundamentals_available_date"]].sort_values(
                    "fundamentals_available_date"
                )
                price_at_filing = pd.merge_asof(
                    filing_dates,
                    p[["trade_date", "close"]].rename(columns={"trade_date": "fundamentals_available_date"}),
                    on="fundamentals_available_date",
                    direction="backward",
                )["close"]
                price_at_filing.index = filing_dates.index
                merged.loc[has_filing, "close_price"] = price_at_filing

        merged_dfs.append(merged)

    final_df = pd.concat(merged_dfs, ignore_index=True)

    print(f"Merged rows: {len(final_df)}")

    return final_df


# =============================================================================
# ADD STATIC COMPANY INFO
# =============================================================================

def merge_company_info(df, company_info):

    print()
    print("=" * 80)
    print("ADDING COMPANY INFO")
    print("=" * 80)

    # ticker_primary duplicates ticker — drop before merging
    company_info = company_info.drop(
        columns=[c for c in ["ticker_primary"] if c in company_info.columns]
    )

    merged = df.merge(
        company_info,
        on="ticker",
        how="left",
    )

    # ponytail: fill missing company_info from sibling tickers (share classes,
    # e.g. ALPA3/ALPA4). BolsAI only lists one ticker per company, so a sibling
    # class can be entirely absent from company_info -- match by base ticker
    # (trailing share-class digits stripped) rather than cvm_code, since a row
    # missing company_info has no cvm_code of its own to match on.
    missing_mask = merged["cvm_code"].isna()
    if missing_mask.any():
        base = company_info["ticker"].str.replace(r"\d+$", "", regex=True)
        info_by_base = (
            company_info.assign(_base=base)
            .dropna(subset=["cvm_code"])
            .drop_duplicates("_base")
            .set_index("_base")
        )
        # Compare only the rows that actually need filling (not the whole
        # dataset) and only loop over base tickers that appear among them —
        # was O(companies * total_rows), rescanning everything per company
        # even when nothing of that company's was missing.
        missing_base = merged.loc[missing_mask, "ticker"].str.replace(r"\d+$", "", regex=True)
        info_by_base = info_by_base[info_by_base.index.isin(missing_base.unique())]

        for b, info_row in info_by_base.iterrows():
            sel = missing_base.index[missing_base == b]
            if len(sel) == 0:
                continue
            # fillna dict values must be non-null: a literal None (from missing
            # object/string columns in the parquet) makes fillna raise instead
            # of being treated as "nothing to fill".
            fill_values = {k: v for k, v in info_row.to_dict().items() if pd.notna(v)}
            merged.loc[sel] = merged.loc[sel].fillna(fill_values)

        still_missing = merged["cvm_code"].isna().sum()
        filled = missing_mask.sum() - still_missing
        print(f"Filled {filled} missing company_info rows from sibling tickers")
        if still_missing:
            print(f"  {still_missing} rows still missing (no cvm_code available)")

    # ponytail: fall back to CVM crosswalk for cvm_code when sibling fill misses
    # (BolsAI collection gap for some tickers; crosswalk is free/always available).
    still_missing_code = merged["cvm_code"].isna()
    if still_missing_code.any() and CVM_CROSSWALK_PATH.exists():
        cvm_xwalk = pd.read_parquet(CVM_CROSSWALK_PATH)[["ticker", "cvm_code"]].drop_duplicates("ticker")
        cvm_map = dict(zip(cvm_xwalk["ticker"], cvm_xwalk["cvm_code"]))
        merge_tickers = merged.loc[still_missing_code, "ticker"].unique()
        xwalk_fill = sum(1 for t in merge_tickers if t in cvm_map)
        if xwalk_fill:
            merged.loc[still_missing_code, "cvm_code"] = (
                merged.loc[still_missing_code, "ticker"].map(cvm_map)
            )
            print(f"Filled {xwalk_fill} additional cvm_code values from CVM crosswalk")

    # ponytail: tickers with no company_info at all (no sibling either) have
    # status=NaN even after the fill above. Sector/cnpj/etc. can't be guessed,
    # but status can be inferred from price recency: still trading near the
    # dataset's last date = ATIVO, otherwise CANCELADA.
    status_missing = merged["status"].isna()
    if status_missing.any():
        last_trade = merged.groupby("ticker")["trade_date"].transform("max")
        max_date = merged["trade_date"].max()
        is_recent = (max_date - last_trade).dt.days <= STATUS_INFERENCE_WINDOW_DAYS
        merged.loc[status_missing, "status"] = np.where(
            is_recent[status_missing], "ATIVO", "CANCELADA"
        )
        n_ativo = (status_missing & (merged["status"] == "ATIVO")).sum()
        n_cancelada = (status_missing & (merged["status"] == "CANCELADA")).sum()
        print(f"Inferred status from price recency ({STATUS_INFERENCE_WINDOW_DAYS}d window) "
              f"for {status_missing.sum()} rows: {n_ativo} ATIVO, {n_cancelada} CANCELADA "
              f"(sector/cnpj/etc. remain NaN — no data source to fill them from)")

    # ponytail: override stale CANCELADA status when price data is recent. BolsAI
    # company_info can lag behind actual trading (e.g. ITUB3 marked delisted but
    # still trading). If a ticker traded within 30 days of dataset end, it's ATIVO.
    dataset_end = merged["trade_date"].max()
    very_recent_cutoff = dataset_end - pd.Timedelta(days=30)
    last_trade_per_ticker = merged.groupby("ticker")["trade_date"].transform("max")
    still_trading = last_trade_per_ticker >= very_recent_cutoff
    incorrectly_delisted = (merged["status"] == "CANCELADA") & still_trading
    if incorrectly_delisted.any():
        merged.loc[incorrectly_delisted, "status"] = "ATIVO"
        n_corrected = merged.loc[incorrectly_delisted, "ticker"].nunique()
        print(f"✓ Corrected {n_corrected} tickers marked CANCELADA but trading in last 30d to ATIVO")

    return merged


# =============================================================================
# ADD MACRO SERIES (SELIC, CDI, IPCA)
# =============================================================================

def merge_macro(dataset):

    print()
    print("=" * 80)
    print("ADDING MACRO SERIES")
    print("=" * 80)

    # Macro is ticker-independent: one value per calendar date applies to all
    # tickers. Combine the (small, thousands-of-rows) macro series together
    # first, then do ONE sort + ONE merge_asof against the big dataset —
    # sorting/merging the full multi-million-row dataset separately per
    # series (3x full-frame copies) was enough to OOM the build.
    # Outer-join on the union of each series' own publication dates, then
    # ffill — chaining merge_asof series-onto-series instead would collapse
    # everything onto the first series' date grid, silently dropping any
    # other series' publication dates that fall between them.
    macro = None
    for name in ("selic", "cdi", "ipca"):
        print(f"Merging {name}")
        m = pd.read_parquet(MACRO_DIR / f"{name}.parquet")[["reference_date", name]]
        macro = m if macro is None else macro.merge(m, on="reference_date", how="outer")

    macro = macro.sort_values("reference_date").ffill().rename(columns={"reference_date": "macro_date"})

    # Chaining sort/merge/drop/sort as one expression keeps every intermediate
    # frame alive at once (old `dataset` stays bound until the whole RHS
    # finishes evaluating) — on a multi-million-row frame that's several full
    # copies resident simultaneously. Split into statements and `del` each as
    # soon as its replacement exists so the old one is freed before the next
    # copy is made; `ignore_index=True` also skips a separate reset_index copy.
    dataset = dataset.sort_values("trade_date")
    merged = pd.merge_asof(
        dataset, macro, left_on="trade_date", right_on="macro_date", direction="backward",
    )
    del dataset
    del merged["macro_date"]  # in-place, unlike .drop(columns=...)

    return merged.sort_values(["ticker", "trade_date"], ignore_index=True)


# =============================================================================
# MERGE DIVIDENDS
# =============================================================================

def merge_dividends(dataset, dividends):

    print()
    print("=" * 80)
    print("MERGING DIVIDENDS")
    print("=" * 80)

    merged_dfs = []
    count = 0

    # ponytail: split dividends by ticker once via groupby instead of
    # re-filtering the full table on every loop iteration
    dividends_by_ticker = dict(tuple(dividends.groupby("ticker", sort=False)))

    for ticker, d in dataset.groupby("ticker", sort=False):

        if count % 50 == 0:
            print(f"Processing dividends for ticker #{count}")
        count += 1

        div = dividends_by_ticker.get(ticker, dividends.iloc[0:0]).sort_values("ex_date")

        if len(div) == 0:
            # No dividends — set div_value_recent (used downstream); yield/count are
            # (re)computed for all tickers in compute_dividend_features.
            # has_dividends=0 marks "never collected", distinct from a ticker that
            # was collected but genuinely paid nothing in a given window — without
            # this, div_yield_12m==0 is ambiguous between "confirmed zero" and
            # "we don't have this ticker's dividends yet" (coverage isn't uniform).
            d = d.copy()
            d["div_value_recent"] = 0.0
            d["has_dividends"] = 0
            merged_dfs.append(d)
            continue

        # Merge most recent dividend (ex_date <= trade_date) for each price row
        merged = pd.merge_asof(
            d.sort_values("trade_date"),
            div[["ex_date", "value_per_share"]].rename(
                columns={"ex_date": "div_ex_date", "value_per_share": "div_value_recent"}
            ),
            left_on="trade_date",
            right_on="div_ex_date",
            direction="backward",
        ).drop(columns="div_ex_date")
        merged["has_dividends"] = 1

        merged_dfs.append(merged)

    result = pd.concat(merged_dfs, ignore_index=True)
    n_missing = result.loc[result["has_dividends"] == 0, "ticker"].nunique()
    print(f"Merged {len(dividends)} dividends into {len(result)} rows")
    print(f"  {n_missing} tickers have no dividend data collected (has_dividends=0)")

    return result
