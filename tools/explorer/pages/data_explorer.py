"""PAGE 1 — Data Explorer: inspect the datasets at every pipeline stage."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.agent.config import DEFAULT_CONFIG  # noqa: E402
from tools.explorer import data_access as da  # noqa: E402
from tools.explorer import ui  # noqa: E402

st.title("Data Explorer")
st.markdown(
    """
This page inspects the **data itself**, independent of any model. The pipeline has four
data stages, each the output of one processing step:

| Stage | File(s) | Produced by |
|---|---|---|
| `raw_prices` | `data/raw/prices/{ticker}.parquet` | Stage 1 collection (BolsAI/yfinance) |
| `raw_fundamentals` | `data/raw/fundamentals/{ticker}.parquet` | Stage 1 collection (quarterly) |
| `processed` | `data/processed/ml_dataset.parquet` | Stage 2 merge + feature engineering |
| `training` | `data/processed/ml_dataset_training.parquet` | Stage 3 cleaning (corrupt-row removal, split-safe returns) |

Use the sidebar to pick tickers, a date range, and optionally restrict to one
training window's train/val/test span.
"""
)

filters = ui.sidebar_filters()
tickers = filters["tickers"]
date_range = filters["date_range"]
spans = filters["spans"]

tab_health, tab_ticker, tab_diff, tab_dist = st.tabs(
    ["Health overview", "Ticker explorer", "Stage diff", "Distributions"]
)

# ------------------------------------------------------------------ Health overview
with tab_health:
    st.markdown(
        """
### What this shows
A per-stage scorecard for the selected tickers/date range: row counts, ticker counts,
date span, duplicate `(ticker, date)` pairs, and the overall share of missing cells.

### Why it matters
Row counts should *shrink or stay equal* moving down the pipeline (cleaning drops rows,
it never invents them). Duplicates must be **zero** at every stage — a duplicate
`(ticker, date)` pair silently double-weights that day in training. A jump in NaN%
between two stages points at the step in between.
"""
    )
    rows = []
    for name in da.STAGES:
        df = ui.stage_df(name, tuple(tickers), date_range)
        if len(df) == 0:
            continue
        rows.append({
            "stage": name, "rows": len(df), "tickers": df["ticker"].nunique(),
            "span": f"{df['date'].min().date()} → {df['date'].max().date()}",
            "duplicates": int(df.duplicated(["ticker", "date"]).sum()),
            "nan_%": round(df.isnull().mean().mean() * 100, 2),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Coverage heatmap (ticker × month)")
    st.markdown(
        """
Each row is a ticker, each column a month; cell color = number of trading days with data.
**How to read it:** rows starting late are IPOs, rows ending early are delistings — both
are normal. What is *not* normal: a horizontal gap in the middle of a row (a collection
hole for that ticker) or a vertical light stripe across many tickers at once (a
pipeline-wide collection failure on those dates).
"""
    )
    stage_for_coverage = st.selectbox("Stage", list(da.STAGES.keys()), key="coverage_stage")
    cov_df = ui.stage_df(stage_for_coverage, tuple(tickers), date_range)
    if len(cov_df) > 0:
        cov = cov_df.assign(month=cov_df["date"].dt.to_period("M").astype(str))
        pivot = cov.groupby(["ticker", "month"]).size().unstack(fill_value=0)
        st.plotly_chart(px.imshow(pivot, aspect="auto", labels=dict(color="rows")),
                        use_container_width=True)

    st.subheader("NaN % per feature")
    st.markdown(
        """
Share of missing values per column, sorted worst-first. Quarterly fundamentals are
*expected* to have some NaN before a company's first report; a **price or volume column
with any NaN**, or a feature that is mostly NaN, means the step that computes it is broken
for these tickers.
"""
    )
    nan_stage = st.selectbox("Stage", list(da.STAGES.keys()), key="nan_stage")
    nan_df = ui.stage_df(nan_stage, tuple(tickers), date_range)
    if len(nan_df) > 0:
        nan_rates = (nan_df.isnull().mean() * 100).sort_values(ascending=False)
        st.bar_chart(nan_rates[nan_rates > 0])

# ------------------------------------------------------------------ Ticker explorer
with tab_ticker:
    st.markdown(
        """
### What this shows
Time series of any feature, for any tickers, at any stage — prices, returns, technical
indicators, fundamentals, macro. Background shading marks the training windows:
<span style="color:green">green = train</span>,
<span style="color:orange">orange = validation</span>,
<span style="color:red">red = test</span>.

### How to interpret
Prices should look continuous; a cliff-edge drop of ~50% in `close` that does **not**
appear in `adj_close` is an unadjusted stock split (a bug this project has hit before).
Fundamental features move in quarterly staircases — that is correct (they only change
when a report lands); smooth daily drift in a fundamental means the merge is wrong.
Use *Rebase to 100* to compare tickers of very different price levels.
""",
        unsafe_allow_html=True,
    )
    stage_name = st.selectbox("Stage", list(da.STAGES.keys()) + ["env_tensors"], key="t2_stage")
    if stage_name == "env_tensors":
        df = ui.env_tensors_df(tuple(tickers), date_range)
    else:
        df = ui.stage_df(stage_name, tuple(tickers), date_range)
    df = ui.apply_split_filter(df, filters)

    if len(df) > 0:
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        feature_choice = st.multiselect("Features", numeric_cols, default=numeric_cols[:1])
        rebase = st.checkbox("Rebase to 100 at first date")
        logy = st.checkbox("Log scale")

        fig = go.Figure()
        for t in tickers:
            sub = df[df["ticker"] == t].sort_values("date")
            for feat in feature_choice:
                y = sub[feat]
                if rebase and len(y) > 0 and y.iloc[0] not in (0, np.nan):
                    y = y / y.iloc[0] * 100
                fig.add_trace(go.Scatter(x=sub["date"], y=y, name=f"{t}:{feat}"))
        ui.add_split_shading(fig, spans)
        if logy:
            fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for the current filters.")

# ------------------------------------------------------------------ Stage diff
with tab_diff:
    st.markdown(
        """
### What this shows
The same ticker at two different pipeline stages, joined on date: which rows exist only
in one stage (what cleaning dropped/added) and the maximum absolute difference per shared
numeric column (what processing changed).

### Why it matters
This is the fastest way to verify a processing step did *only* what it claims.
`processed` → `training` should show dropped corrupt rows and little else. Any column
with an unexpectedly large `max_abs_diff` is a transformation you didn't intend.
"""
    )
    if len(tickers) != 1:
        st.info("Select exactly one ticker in the sidebar to diff stages.")
    else:
        t = tickers[0]
        stage_options = list(da.STAGES.keys()) + ["env_tensors"]
        c1, c2 = st.columns(2)
        stage_a = c1.selectbox("Stage A", stage_options, index=stage_options.index("processed"))
        stage_b = c2.selectbox("Stage B", stage_options, index=stage_options.index("training"))

        def _load_one(stage: str, col, slot: str) -> pd.DataFrame:
            if stage == "env_tensors":
                cutoffs = sorted(da.load_scalers().keys())
                cutoff = col.selectbox("Scaler cutoff (env_tensors)", ["(raw)"] + cutoffs, key=f"cutoff_{slot}")
                return ui.env_tensors_df((t,), date_range, None if cutoff == "(raw)" else cutoff)
            return ui.stage_df(stage, (t,), date_range)

        df_a, df_b = _load_one(stage_a, c1, "A"), _load_one(stage_b, c2, "B")
        if len(df_a) == 0 or len(df_b) == 0:
            st.info("One of the selected stages has no rows for this ticker/range.")
        else:
            merged = df_a.merge(df_b, on="date", how="outer", suffixes=("_A", "_B"), indicator=True)
            st.write(f"Rows only in A: {(merged['_merge']=='left_only').sum()} | "
                     f"only in B: {(merged['_merge']=='right_only').sum()} | "
                     f"shared: {(merged['_merge']=='both').sum()}")

            numeric_a = df_a.select_dtypes(include=[np.number]).columns
            shared_cols = [c for c in numeric_a if c in df_b.columns and c not in ("ticker", "date")]
            diff_rows = []
            for col in shared_cols:
                if f"{col}_A" in merged.columns and f"{col}_B" in merged.columns:
                    diff = (merged[f"{col}_A"] - merged[f"{col}_B"]).abs()
                    diff_rows.append({"column": col, "max_abs_diff": diff.max()})
            if diff_rows:
                st.dataframe(pd.DataFrame(diff_rows).sort_values("max_abs_diff", ascending=False),
                             use_container_width=True)

            overlay_col = st.selectbox("Overlay column", shared_cols) if shared_cols else None
            if overlay_col:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_a["date"], y=df_a[overlay_col], name=f"A: {stage_a}"))
                fig.add_trace(go.Scatter(x=df_b["date"], y=df_b[overlay_col], name=f"B: {stage_b}"))
                st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ Distributions
with tab_dist:
    st.markdown(
        """
### What this shows
The distribution of any feature, split by the train/val/test period it falls into,
plus a per-feature distribution-shift ranking and a correlation matrix of the 23
features the agent actually sees.

### How to interpret
- **Histogram/box by split**: the three splits should look broadly similar. A test
  distribution wildly unlike train means the model is being evaluated on data it never
  saw the likes of — either a genuine regime change or a preprocessing bug.
- **KS table**: Kolmogorov–Smirnov distance (0 = identical, 1 = disjoint) between the
  train and test distribution of each feature, sorted worst-first. The features at the
  top are where train/test shift lives.
- **Correlation heatmap**: pairs near ±1.0 are redundant features (the agent gains
  nothing from both); a feature correlated ~1.0 with next-day returns would indicate
  lookahead leakage.
"""
    )
    stage_name = st.selectbox("Stage", list(da.STAGES.keys()), key="t4_stage")
    df = ui.stage_df(stage_name, tuple(tickers), date_range)
    if len(df) > 0:
        df = df.assign(split=None)
        for row in spans.itertuples():
            m = (df["date"] >= row.start) & (df["date"] <= row.end)
            df.loc[m, "split"] = row.split
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        feat = st.selectbox("Feature", numeric_cols)
        chart_kind = st.radio("Chart", ["Histogram", "Box"], horizontal=True)
        if chart_kind == "Histogram":
            fig = px.histogram(df, x=feat, color="split", barmode="overlay", opacity=0.6)
        else:
            fig = px.box(df, x="split", y=feat)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("KS statistic: train vs test (per feature)")
        ks_rows = []
        train_df, test_df = df[df["split"] == "train"], df[df["split"] == "test"]
        for col in numeric_cols:
            a, b = train_df[col].dropna(), test_df[col].dropna()
            if len(a) > 5 and len(b) > 5:
                stat, _ = ks_2samp(a, b)
                ks_rows.append({"feature": col, "ks_stat": stat})
        if ks_rows:
            st.dataframe(pd.DataFrame(ks_rows).sort_values("ks_stat", ascending=False),
                         use_container_width=True)

        st.subheader("Correlation heatmap (agent state features)")
        present = [c for c in DEFAULT_CONFIG.state_features if c in df.columns]
        if present:
            corr = df[present].corr()
            st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto"), use_container_width=True)
