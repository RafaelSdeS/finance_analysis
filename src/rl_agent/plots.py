"""
plots.py — one self-contained plotly HTML report per experiment
(docs/eiie_agent/EIIE_AGENT_PLAN.md "Evaluation" section: portfolio value vs every
baseline, reward curves, allocation evolution, turnover/cost, weight
distribution, metrics+CI table). plotly.js is embedded once in the first
figure and reused by the rest -- the file opens standalone, no network
request needed.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go

from .data import CASH_GIDX, GlobalAssetIndex
from .environment import BacktestResult
from .metrics import MetricsSummary

_METRIC_FIELDS = [
    "total_return", "annualized_return", "cagr", "volatility", "sharpe", "sortino", "calmar",
    "max_drawdown", "var", "cvar", "mean_daily_turnover", "annualized_turnover",
    "transaction_cost_drag", "win_rate", "information_ratio", "final_apv",
    "allocation_entropy", "effective_n_holdings", "mean_cash_weight",
    "frac_days_cash_gt90", "frac_days_single_name_gt70", "argmax_switches",
    "mean_position_lifetime",
]


def _pv_figure(agent_result: BacktestResult, baseline_results: dict, agent_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agent_result.dates, y=agent_result.portfolio_value[1:],
                              mode="lines", name=agent_name, line=dict(width=3)))
    for name, result in baseline_results.items():
        fig.add_trace(go.Scatter(x=result.dates, y=result.portfolio_value[1:],
                                  mode="lines", name=name, line=dict(dash="dot")))
    fig.update_layout(title="Portfolio Value (log scale)", yaxis_type="log",
                       xaxis_title="Date", yaxis_title="APV (p_t / p_0)", template="plotly_white")
    return fig


def _reward_curve_figure(train_losses: list, eval_log_returns: Optional[np.ndarray]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=-np.asarray(train_losses), mode="lines", name="training reward (-loss)"))
    if eval_log_returns is not None and len(eval_log_returns) > 0:
        fig.add_trace(go.Scatter(y=np.cumsum(eval_log_returns), mode="lines",
                                  name="backtest cumulative log-return", yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="cumulative log-return"))
    fig.update_layout(title="Training / Evaluation Reward Curve", xaxis_title="step / period",
                       yaxis_title="training reward (-loss)", template="plotly_white")
    return fig


def _allocation_figure(result: BacktestResult, asset_index: GlobalAssetIndex, top_k: int = 9) -> go.Figure:
    weights = result.weights  # (T, n_global)
    mean_weight = weights.mean(axis=0)
    top_gidx = np.argsort(mean_weight[1:])[::-1][:top_k] + 1
    other_gidx = np.setdiff1d(np.arange(1, weights.shape[1]), top_gidx)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.dates, y=weights[:, CASH_GIDX], stackgroup="one", name="cash"))
    for gidx in top_gidx:
        fig.add_trace(go.Scatter(x=result.dates, y=weights[:, gidx], stackgroup="one",
                                  name=asset_index.tickers[gidx - 1]))
    if len(other_gidx) > 0:
        fig.add_trace(go.Scatter(x=result.dates, y=weights[:, other_gidx].sum(axis=1),
                                  stackgroup="one", name="other"))
    fig.update_layout(title="Allocation Evolution (top holdings)", xaxis_title="Date",
                       yaxis_title="weight", template="plotly_white")
    return fig


def _turnover_cost_figure(result: BacktestResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.dates, y=result.turnover, mode="lines", name="turnover"))
    fig.add_trace(go.Scatter(x=result.dates, y=np.cumsum(1.0 - result.mu), mode="lines",
                              name="cumulative cost drag", yaxis="y2"))
    fig.update_layout(title="Turnover & Transaction Costs", xaxis_title="Date", yaxis_title="turnover",
                       yaxis2=dict(overlaying="y", side="right", title="cumulative cost drag"),
                       template="plotly_white")
    return fig


def _weight_distribution_figure(result: BacktestResult) -> go.Figure:
    non_cash = result.weights[:, 1:].flatten()
    non_trivial = non_cash[non_cash > 1e-4]
    fig = go.Figure(go.Histogram(x=non_trivial, nbinsx=50))
    fig.update_layout(title="Position-Size Distribution (non-cash, non-trivial holdings)",
                       xaxis_title="weight", yaxis_title="count", template="plotly_white")
    return fig


def _ci_str(summary: MetricsSummary, attr: str) -> str:
    _, lo, hi = getattr(summary, attr)
    return f"[{lo:.4f}, {hi:.4f}]"


def _metrics_table_figure(agent_summary: MetricsSummary, agent_name: str, baseline_summaries: dict) -> go.Figure:
    row_names = [f.replace("_", " ") for f in _METRIC_FIELDS] + ["total return 95% CI", "sharpe 95% CI"]

    def _column(summary: MetricsSummary) -> list:
        return [f"{getattr(summary, f):.4f}" for f in _METRIC_FIELDS] + [
            _ci_str(summary, "total_return_ci"), _ci_str(summary, "sharpe_ci"),
        ]

    header = ["metric", agent_name] + list(baseline_summaries.keys())
    values = [row_names, _column(agent_summary)] + [_column(s) for s in baseline_summaries.values()]

    fig = go.Figure(go.Table(
        header=dict(values=header, fill_color="#2c3e50", font=dict(color="white"), align="left"),
        cells=dict(values=values, align="left"),
    ))
    fig.update_layout(title="Performance Metrics", template="plotly_white")
    return fig


def write_report(path, agent_result: BacktestResult, agent_summary: MetricsSummary,
                  baseline_results: dict, baseline_summaries: dict,
                  train_losses: list, asset_index: GlobalAssetIndex,
                  eval_log_returns: Optional[np.ndarray] = None,
                  agent_name: str = "EIIE Agent", title: str = "EIIE Agent -- Experiment Report") -> None:
    """Write one self-contained HTML report combining every required chart
    plus the metrics table, agent vs. every baseline side by side."""
    figures = [
        _pv_figure(agent_result, baseline_results, agent_name),
        _reward_curve_figure(train_losses, eval_log_returns),
        _allocation_figure(agent_result, asset_index),
        _turnover_cost_figure(agent_result),
        _weight_distribution_figure(agent_result),
        _metrics_table_figure(agent_summary, agent_name, baseline_summaries),
    ]

    parts = [f"<html><head><title>{title}</title></head><body><h1>{title}</h1>"]
    for i, fig in enumerate(figures):
        parts.append(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))
    parts.append("</body></html>")

    Path(path).write_text("\n".join(parts))
