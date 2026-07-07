"""PAGE 4 — Training Analysis: learning curves, backtest performance, benchmark comparison."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.agent import metrics as agent_metrics  # noqa: E402
from tools.explorer import data_access as da  # noqa: E402
from tools.explorer import ui  # noqa: E402

st.title("Training Analysis")
st.markdown(
    """
This page evaluates the **training process and its outcome**: did training improve the
model (learning curves), and does the trained agent beat naive baselines out of sample
(backtest performance)? The three baselines matter because they are free — an RL agent
that can't beat *equal weight* (buy everything equally), *market cap* (buy the index),
or *inverse volatility* (overweight calm stocks) is not worth deploying.
"""
)

tab_curves, tab_perf = st.tabs(["Training curves", "Backtest performance"])

# ------------------------------------------------------------------ Training curves
with tab_curves:
    st.markdown(
        """
### What this shows
Validation metrics logged during PPO training, one point per evaluation checkpoint
(every ~40k timesteps). Each *run* is one invocation of the trainer for one rolling
window (`agent` = the production/most-recent window, `window_N` = earlier windows).

### How to interpret
- **val_sharpe rising then plateauing** is healthy learning; the trainer's early stopping
  keeps the checkpoint at the peak (`*_best.zip`), so a decline after the peak is fine.
- **val_sharpe flat from the start** means the agent isn't learning — check the reward
  signal and feature scaling before burning more GPU hours.
- **val_sharpe falling from the start** usually means the learning rate is too high or
  the reward is misconfigured.
- Compare runs of the same window across dates to see if a code change actually helped.
"""
    )
    logs = ui.training_logs_df()
    if len(logs) == 0:
        st.info("No training logs found in data/logs/agent_training_*.jsonl — train first.")
    else:
        runs = sorted(logs["run"].unique())
        default_runs = [r for r in runs if r.startswith("agent")][-2:] or runs[-2:]
        chosen = st.multiselect("Runs", runs, default=default_runs)
        metric = st.selectbox("Metric", ["val_sharpe", "val_final_value", "val_max_drawdown"])
        sub = logs[logs["run"].isin(chosen)]
        fig = px.line(sub, x="timesteps", y=metric, color="run", markers=True)
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ Backtest performance
with tab_perf:
    source_label = st.selectbox("Backtest source", list(da.BACKTEST_SOURCES.keys()), key="perf_source")
    st.markdown(
        """
Three backtest flavors: **Last window** = the production model on its held-out ~2y test
span; **Walk-forward** = all 8 windows' test spans stitched into one out-of-sample
history (the most honest long-run view); **Online rollout** = continuous deployment
simulation with periodic fine-tuning.
"""
    )
    source_file = da.BACKTEST_SOURCES[source_label]
    m = ui.metrics_dict(source_file)
    bt = ui.backtest_df(source_file)
    value_cols = [c for c in bt.columns if c.startswith("value_")]

    st.subheader("Metrics: agent vs baselines")
    st.markdown(
        """
Read Sharpe first (risk-adjusted return; >1 is good, <0.5 weak for equities), then max
drawdown (worst peak-to-trough loss — the number that gets strategies abandoned in real
life). Cumulative return alone is misleading: a strategy can win by taking absurd risk.
Best value per column is highlighted (lowest for drawdown).
"""
    )
    mtable = pd.DataFrame(m).T
    st.dataframe(
        mtable.style
        .highlight_max(axis=0, subset=[c for c in mtable.columns if c != "max_drawdown"])
        .highlight_min(axis=0, subset=["max_drawdown"] if "max_drawdown" in mtable.columns else []),
        use_container_width=True,
    )

    st.subheader("Cumulative value")
    st.markdown("Growth of the initial R$100k under each strategy, plus **BOVA11** (the "
                "IBOV ETF) rebased to the same start — the real-world 'just buy the index' "
                "alternative. Log scale recommended for multi-year spans: equal vertical "
                "distances = equal percentage gains.")
    logy = st.checkbox("Log scale", key="perf_logy", value=True)
    bova = ui.bova11_df((str(bt["date"].min().date()), str(bt["date"].max().date())))
    fig = go.Figure()
    for c in value_cols:
        fig.add_trace(go.Scatter(x=bt["date"], y=bt[c], name=c.removeprefix("value_")))
    if len(bova) > 0:
        rebased = bova.set_index("date")["close"].reindex(bt["date"]).ffill()
        rebased = rebased / rebased.dropna().iloc[0] * bt[value_cols[0]].iloc[0]
        fig.add_trace(go.Scatter(x=bt["date"], y=rebased.values, name="BOVA11 (rebased)",
                                 line=dict(dash="dot")))
    if logy:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Drawdown")
    st.markdown("Percentage below the running peak, per strategy. Depth is pain; *duration* "
                "(how long until a new peak) is what actually breaks conviction — a strategy "
                "3 years underwater gets turned off before it recovers.")
    fig = go.Figure()
    for c in value_cols:
        v = bt[c].to_numpy()
        dd = (np.maximum.accumulate(v) - v) / np.maximum.accumulate(v)
        fig.add_trace(go.Scatter(x=bt["date"], y=-dd * 100, name=c.removeprefix("value_")))
    fig.update_layout(yaxis_title="Drawdown (%)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rolling Sharpe")
    st.markdown("The headline Sharpe is an average that can hide one lucky quarter. This "
                "shows *when* the agent's risk-adjusted return was earned: consistently "
                "positive = a real edge; one spike surrounded by noise = luck.")
    window = st.selectbox("Window (trading days)", [63, 252], index=0)
    rolling_sharpe = bt["log_return"].rolling(window).apply(
        lambda x: agent_metrics.sharpe_ratio(x.to_numpy()), raw=False)
    st.plotly_chart(px.line(x=bt["date"], y=rolling_sharpe,
                            labels={"x": "date", "y": f"rolling {window}d Sharpe"}),
                    use_container_width=True)

    st.subheader("Alpha / beta vs benchmark")
    st.markdown("Each dot is one day: benchmark return (x) vs agent return (y). The fitted "
                "line's **slope (beta)** is market exposure — beta ≈ 1 with alpha ≈ 0 means "
                "the agent is just the market in disguise. **Intercept (alpha)** is the "
                "daily return earned independent of the market — the part that is actual skill.")
    bench_col = st.selectbox("Benchmark", [c for c in value_cols if c != "value_agent"])
    bench_ret = np.log(bt[bench_col]).diff()
    agent_ret = bt["log_return"]
    valid = bench_ret.notna() & agent_ret.notna()
    if valid.sum() > 2:
        beta, alpha = np.polyfit(bench_ret[valid], agent_ret[valid], 1)
        fig = px.scatter(x=bench_ret[valid], y=agent_ret[valid],
                         labels={"x": "benchmark daily return", "y": "agent daily return"},
                         opacity=0.4)
        xs = np.linspace(bench_ret[valid].min(), bench_ret[valid].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=alpha + beta * xs, mode="lines",
                                 name=f"alpha={alpha:.5f}/day, beta={beta:.2f}"))
        st.plotly_chart(fig, use_container_width=True)

    rolling = da.load_rolling_eval()
    if rolling and "windows" in rolling:
        st.subheader("Per-window results (walk-forward)")
        st.markdown("The agent's performance on each rolling window's test span, next to the "
                    "baselines on the same span. This is the robustness check the stitched "
                    "curve hides: winning 7 of 8 windows is an edge; winning 1 big window and "
                    "losing 7 is curve-fit luck.")
        st.dataframe(pd.DataFrame(rolling["windows"]), use_container_width=True)
