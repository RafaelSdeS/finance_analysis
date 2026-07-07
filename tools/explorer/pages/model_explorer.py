"""PAGE 3 — Model Explorer: what the model sees (inputs) and what it does (allocations)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.agent.config import DEFAULT_CONFIG  # noqa: E402
from tools.explorer import data_access as da  # noqa: E402
from tools.explorer import ui  # noqa: E402

st.title("Model Explorer")
st.markdown(
    """
This page inspects the **model side** of the pipeline: the exact observations the RL
agent receives, the portfolio weights it produced in backtests, and live weight
predictions from the trained model. If the Data Explorer answers *"is the data right?"*,
this page answers *"is the model seeing and doing what we think?"*
"""
)

filters = ui.sidebar_filters()
tickers = filters["tickers"]
date_range = filters["date_range"]

tab_inputs, tab_alloc, tab_infer = st.tabs(["Model inputs", "Allocations", "Live inference"])

# ------------------------------------------------------------------ Model inputs
with tab_inputs:
    st.markdown(
        """
### What this shows
The agent does not see the dataset — it sees a dense tensor of **23 normalized features**
per ticker per day (prices, technicals, fundamentals, macro), standardized by a scaler
fitted only on training data (no lookahead). This tab shows those features either *raw*
(pre-scaler) or *scaled* (exactly what enters the neural network).

### How to interpret
With a scaler applied, features restricted to that window's **train span should look
≈ N(0,1)**: mean near 0, most values within ±3. If a scaled feature sits far from 0 on
average, or spans ±50, the scaler is wrong or the feature is corrupted — and the network
will be dominated by that one input. On the *test* span some drift away from N(0,1) is
expected (the future differs from the past); wild blowups are not.
"""
    )
    cutoffs = sorted(da.load_scalers().keys())
    scale_choice = st.selectbox("Scaler (window train-end cutoff)", ["(raw — no scaling)"] + cutoffs)
    cutoff = None if scale_choice.startswith("(raw") else scale_choice
    df = ui.env_tensors_df(tuple(tickers), date_range, cutoff)

    if len(df) > 0:
        feats = st.multiselect("Features", DEFAULT_CONFIG.state_features,
                               default=DEFAULT_CONFIG.state_features[:1])
        fig = go.Figure()
        for t in tickers:
            sub = df[df["ticker"] == t].sort_values("date")
            for feat in feats:
                fig.add_trace(go.Scatter(x=sub["date"], y=sub[feat], name=f"{t}:{feat}"))
        ui.add_split_shading(fig, filters["spans"])
        st.plotly_chart(fig, use_container_width=True)

        if cutoff is not None and feats:
            st.markdown("**Sanity table** — mean/std of the scaled features over the "
                        "selected range (train-span values should be ≈ 0 / ≈ 1):")
            stats = df[feats].agg(["mean", "std"]).T.round(3)
            st.dataframe(stats, use_container_width=True)
    else:
        st.info("Selected tickers are not in the tensor universe (need ≥252 rows of history).")

# ------------------------------------------------------------------ Allocations
with tab_alloc:
    st.markdown(
        """
### What this shows
The daily portfolio weights the agent chose during a backtest — as a composition over
time, by sector, and as concentration/turnover diagnostics.

### Why it matters
The July 2026 reward change (excess return instead of absolute return) exists to give the
agent **conviction** — concentrated bets instead of closet-indexing. These charts are how
you verify it worked: effective-N should sit well below the ~240-name universe, max weight
above the ~0.4% equal-weight level, and turnover low enough that transaction costs don't
eat the alpha.
"""
    )
    source_label = st.selectbox("Backtest source", list(da.BACKTEST_SOURCES.keys()), key="alloc_source")
    source_file = da.BACKTEST_SOURCES[source_label]
    bt = ui.backtest_df(source_file)
    w_cols = [c for c in bt.columns if c.startswith("w_")]
    w_tickers = [c[2:] for c in w_cols]
    secs = ui.sector_series()

    st.subheader("Weights over time")
    st.markdown("Stacked composition of the portfolio, largest average holdings shown "
                "individually, the rest bucketed as *Other*. The CASH band is the agent "
                "choosing to sit out — a persistent 100% CASH means the model collapsed "
                "to the risk-free corner.")
    top_n = st.slider("Top N holdings", 3, 20, 10)
    mean_w = bt[w_cols].mean().sort_values(ascending=False)
    top_cols = [c for c in mean_w.index[:top_n] if c != "w_CASH"]
    if "w_CASH" in w_cols:
        top_cols = ["w_CASH"] + top_cols
    fig = go.Figure()
    for c in top_cols:
        fig.add_trace(go.Scatter(x=bt["date"], y=bt[c] * 100, name=c[2:], stackgroup="w"))
    other_cols = [c for c in w_cols if c not in top_cols]
    if other_cols:
        fig.add_trace(go.Scatter(x=bt["date"], y=bt[other_cols].sum(axis=1) * 100,
                                 name="Other", stackgroup="w"))
    fig.update_layout(yaxis_title="Weight (%)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sector allocation over time")
    st.markdown("The same weights aggregated by sector. A portfolio that is one sector in "
                "disguise (e.g. 80% banks) carries concentrated sector risk the per-ticker "
                "view hides.")
    sector_w = pd.DataFrame({t: bt[f"w_{t}"] for t in w_tickers if t != "CASH"})
    sector_w.index = bt["date"]
    sector_alloc = sector_w.T.groupby(secs.reindex(sector_w.columns)).sum().T
    fig = go.Figure()
    for sec in sector_alloc.columns:
        fig.add_trace(go.Scatter(x=sector_alloc.index, y=sector_alloc[sec] * 100,
                                 name=str(sec), stackgroup="s"))
    fig.update_layout(yaxis_title="Weight (%)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Conviction diagnostics")
    st.markdown(
        """
- **Effective N** = 1/HHI, the number of *meaningful* positions. Equal weight over ~240
  names → ~240; a 10-stock conviction portfolio → ~10.
- **Max weight**: the largest single position. Equal weight ≈ 0.4%; conviction shows >1–5%.
- **CASH weight**: risk appetite over time; spikes should coincide with drawdowns, not calm markets.
- **Cost drag**: cumulative transaction cost implied by turnover × 10 bps — what trading
  activity costs before any alpha. If cost drag grows faster than outperformance, the
  agent is churning.
"""
    )
    w_matrix = bt[w_cols].to_numpy()
    effective_n = 1.0 / np.maximum((w_matrix ** 2).sum(axis=1), 1e-12)
    max_w = w_matrix.max(axis=1)
    cash_w = bt["w_CASH"] if "w_CASH" in bt.columns else pd.Series(0.0, index=bt.index)
    turnover = np.abs(np.diff(w_matrix, axis=0, prepend=w_matrix[:1])).sum(axis=1) * 0.5
    cost_drag = (turnover * DEFAULT_CONFIG.transaction_cost_bps / 10_000).cumsum()

    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(x=bt["date"], y=effective_n,
                            labels={"x": "date", "y": "effective N (1/HHI)"}), use_container_width=True)
    c2.plotly_chart(px.line(x=bt["date"], y=max_w * 100,
                            labels={"x": "date", "y": "max weight %"}), use_container_width=True)
    c1.plotly_chart(px.line(x=bt["date"], y=cash_w * 100,
                            labels={"x": "date", "y": "CASH weight %"}), use_container_width=True)
    c2.plotly_chart(px.line(x=bt["date"], y=cost_drag * 100,
                            labels={"x": "date", "y": "cumulative cost drag %"}), use_container_width=True)

    st.subheader("Portfolio snapshot")
    st.markdown("The full portfolio on one day. The **pie** shows relative composition at "
                "a glance (top holdings + *Other*); the **bar chart** and table carry the "
                "exact numbers and sector colors.")
    snap_dates = sorted(bt["date"].dt.date.unique())
    snap_date = st.select_slider("Date", snap_dates, value=snap_dates[-1])
    snap = bt[bt["date"].dt.date == snap_date][w_cols].iloc[0]
    snap_df = pd.DataFrame({"ticker": [c[2:] for c in w_cols], "weight": snap.values})
    snap_df = snap_df[snap_df["weight"] > 0.001].sort_values("weight", ascending=False)
    snap_df["sector"] = snap_df["ticker"].map(secs).fillna("CASH")

    pie_top = snap_df.head(top_n).copy()
    other_weight = snap_df["weight"][top_n:].sum()
    if other_weight > 0:
        pie_top = pd.concat([pie_top, pd.DataFrame([{"ticker": "Other", "weight": other_weight,
                                                     "sector": "Other"}])], ignore_index=True)
    c1, c2 = st.columns(2)
    pie = px.pie(pie_top, names="ticker", values="weight", hole=0.35)
    pie.update_traces(textinfo="label+percent", sort=False)
    c1.plotly_chart(pie, use_container_width=True)
    c2.plotly_chart(px.bar(snap_df, x="ticker", y="weight", color="sector"), use_container_width=True)
    st.dataframe(snap_df, use_container_width=True)

    st.subheader("Two-date diff")
    st.markdown("What changed between two dates: positions entered, exited, or resized. "
                "Useful for auditing a specific rebalance the timeline made look suspicious.")
    d1, d2 = st.select_slider("Compare dates", snap_dates, value=(snap_dates[0], snap_dates[-1]))
    row1 = bt[bt["date"].dt.date == d1][w_cols].iloc[0]
    row2 = bt[bt["date"].dt.date == d2][w_cols].iloc[0]
    diff = (row2 - row1).sort_values()
    diff = diff[diff.abs() > 0.001]
    st.dataframe(pd.DataFrame({"ticker": [c[2:] for c in diff.index], "weight_change": diff.values}),
                 use_container_width=True)

    st.subheader("Holding drill-down")
    st.markdown("One ticker's weight over the backtest, with its price below on a shared "
                "time axis. Answers *\"was the agent buying strength or catching knives?\"* — "
                "weight rising while price falls means it was averaging down.")
    drill_ticker = st.selectbox("Ticker", sorted(t for t in w_tickers if t != "CASH"))
    price = da.load_raw_prices([drill_ticker],
                               (str(bt["date"].min().date()), str(bt["date"].max().date())))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=(f"{drill_ticker} weight (%)", f"{drill_ticker} adj. close"))
    fig.add_trace(go.Scatter(x=bt["date"], y=bt[f"w_{drill_ticker}"] * 100, name="weight %"), row=1, col=1)
    fig.add_trace(go.Scatter(x=price["date"], y=price["adj_close"], name="adj_close"), row=2, col=1)
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ Live inference
with tab_infer:
    st.markdown(
        """
### What this shows
Runs the **trained production model** (`agent_best.zip`) on the observation for a chosen
date and shows the weights it would output today — the same code path
(`src/agent/infer.py`) the daily allocation CLI uses.

### Why it matters
Backtest parquets are historical artifacts; this is the live model. If these weights look
degenerate (all CASH, or one ticker at 99%) while the backtest looked fine, the deployed
model file and the backtested model have diverged. Dates are restricted to the most recent
window's test span, where the model has observations it was never trained on.
"""
    )
    test_start = pd.Timestamp(DEFAULT_CONFIG.test_start).date()
    test_end = pd.Timestamp(DEFAULT_CONFIG.test_end).date()
    infer_date = st.date_input("Date", test_end, min_value=test_start, max_value=test_end)

    models = sorted((ROOT / "data/models").glob("*.zip"))
    if not models:
        st.error("No trained models in data/models/. Train first: `python -m src.agent.trainer`")
        model_path = None
    else:
        # agent_best.zip is the production model when it exists; otherwise pick manually
        default_ix = next((i for i, p in enumerate(models) if p.name == "agent_best.zip"), 0)
        model_path = models[st.selectbox("Model", range(len(models)),
                                         index=default_ix, format_func=lambda i: models[i].name)]

    if model_path is not None and st.button("Predict weights"):
        with st.spinner("Loading model and building observation..."):
            try:
                from src.agent.infer import predict_weights  # lazy: pulls in torch/SB3
                weights = predict_weights(date=str(infer_date), model_path=model_path)
            except Exception as e:  # surface, don't crash the dashboard
                st.error(f"Inference failed: {e}")
                weights = None
        if weights is not None and len(weights) > 0:
            st.write(f"**{len(weights)} positions**, weights sum = {weights['weight'].sum():.4f}")
            top = weights.head(15).copy()
            other = weights["weight"][15:].sum()
            if other > 0:
                top = pd.concat([top, pd.DataFrame([{"ticker": "Other", "weight": other}])],
                                ignore_index=True)
            c1, c2 = st.columns(2)
            pie = px.pie(top, names="ticker", values="weight", hole=0.35)
            pie.update_traces(textinfo="label+percent", sort=False)
            c1.plotly_chart(pie, use_container_width=True)
            c2.plotly_chart(px.bar(weights.head(30), x="ticker", y="weight"),
                            use_container_width=True)
            st.dataframe(weights, use_container_width=True)
