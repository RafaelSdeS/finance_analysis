"""
visualize_portfolio.py -- interactive dashboard for the Phase 1 winning
candidate (the no-trade-band alpha sort from run_alpha_diagnostic.py):
equity curve vs baselines, drawdown, position weights over time (incl.
cash), and turnover per rebalance. For eyeballing what the strategy is
actually doing, not just its scoreboard numbers.

build_dashboard()/save_dashboard() are called automatically at the end of
`run_alpha_diagnostic.py` -- every training run saves its own timestamped
dashboard for free. This file's own `main()` is only for re-plotting with
different --top-holdings/--top-frac/--hold-frac without paying for a fresh
walk-forward retrain.

Run: python -m src.portfolio.visualize_portfolio [--top-n 50] [--horizon-td 252]
    [--top-frac 0.6] [--hold-frac 0.75] [--top-holdings 15]
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.build_dataset.paths import OUTPUT_PATH, PRICES_DIR
from src.portfolio import alpha, universe
from src.portfolio.backtest import buy_and_hold_curve, cdi_curve, equal_weight_fn, run_backtest
from src.portfolio.features import feature_columns
from src.portfolio.labels import forward_excess_return

REPORT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "reports"


def positions_over_time(log: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Wide DataFrame (rebalance date x ticker) of target weights, collapsed
    to the top_n tickers by average weight + an 'Other' bucket + 'Cash'
    (the residual run_backtest already tracks as cash_weight)."""
    wide = pd.DataFrame(log["weights"].tolist(), index=log["date"]).fillna(0.0)
    top = wide.mean().sort_values(ascending=False).head(top_n).index.tolist()
    out = wide[top].copy()
    out["Other"] = wide.drop(columns=top).sum(axis=1)
    out["Cash"] = 1.0 - wide.sum(axis=1)
    return out


def build_dashboard(alpha_curve: pd.Series, alpha_log: pd.DataFrame, eq_curve: pd.Series,
                     bova_curve: pd.Series, cdi_only_curve: pd.Series,
                     top_frac: float, hold_frac: float, top_holdings: int = 15) -> go.Figure:
    positions = positions_over_time(alpha_log, top_n=top_holdings)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, row_heights=[0.35, 0.15, 0.35, 0.15],
        vertical_spacing=0.04,
        subplot_titles=(
            "Equity curve (log scale)", "Drawdown",
            f"Positions over time (top {top_holdings} + Other + Cash)", "Turnover per rebalance",
        ),
    )

    for name, curve in [
        (f"Alpha-sort (top {top_frac:.0%}/hold {hold_frac:.0%})", alpha_curve),
        ("Equal-weight", eq_curve),
        ("BOVA11 (buy & hold)", bova_curve),
        ("100% CDI", cdi_only_curve),
    ]:
        fig.add_trace(go.Scatter(x=curve.index, y=curve.values, name=name, mode="lines"), row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)

    for name, curve in [("Alpha-sort", alpha_curve), ("Equal-weight", eq_curve)]:
        dd = curve / curve.cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=f"{name} drawdown",
                                  mode="lines", showlegend=True), row=2, col=1)

    for col in positions.columns:
        fig.add_trace(go.Scatter(x=positions.index, y=positions[col], name=col,
                                  mode="lines", stackgroup="positions"), row=3, col=1)

    fig.add_trace(go.Bar(x=alpha_log["date"], y=alpha_log["turnover"], name="Turnover"), row=4, col=1)

    fig.update_layout(height=1400, title="Portfolio Dashboard -- Alpha-Sort vs Baselines",
                       hovermode="x unified", legend=dict(groupclick="togglegroup"))
    return fig


def save_dashboard(fig: go.Figure, tag: str = "") -> Path:
    """Timestamped, non-overwriting save -- each training run keeps its own
    dashboard on disk instead of clobbering the last one."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    path = REPORT_DIR / f"portfolio_dashboard_{stamp}{suffix}.html"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(path)
    return path


def main(top_n: int = 50, horizon_td: int = 252, top_frac: float = 0.6,
         hold_frac: float = 0.75, top_holdings: int = 15):
    from src.portfolio.run_alpha_diagnostic import make_alpha_weighted_fn

    print("Loading dataset (full feature set)...")
    base_cols = ["ticker", "trade_date", "adj_close", "traded_amount"]
    all_cols = sorted(set(base_cols) | set(feature_columns(include_sector=False)))
    df = pd.read_parquet(OUTPUT_PATH, columns=all_cols)

    print(f"Building point-in-time liquid universe (top_n={top_n})...")
    membership = universe.liquid_universe(df[["ticker", "trade_date", "traded_amount"]], top_n=top_n)
    df = universe.restrict_to_universe(df, membership)

    print(f"Building the {horizon_td}-day forward-excess-return label...")
    df["label"] = forward_excess_return(df, horizon_td=horizon_td)

    reb_dates = universe.rebalance_dates(membership)
    print(f"Walk-forward training/prediction across {len(reb_dates)} rebalance dates...")
    preds = alpha.walk_forward_predict(df, reb_dates, horizon_td=horizon_td)
    preds_by_date = {d: g.set_index("ticker")["alpha"].to_dict() for d, g in preds.groupby("date")}
    alpha_fn = make_alpha_weighted_fn(preds_by_date, top_frac=top_frac, hold_frac=hold_frac)

    prices = df[["ticker", "trade_date", "adj_close"]]
    cdi = pd.read_parquet(OUTPUT_PATH, columns=["trade_date", "cdi"]).drop_duplicates().sort_values("trade_date")

    print("Running alpha-sort and equal-weight baseline...")
    alpha_curve, alpha_log = run_backtest(prices, cdi, membership, alpha_fn)
    eq_curve, _ = run_backtest(prices, cdi, membership, equal_weight_fn)

    print("Loading BOVA11 / 100% CDI baselines...")
    bova = pd.read_parquet(PRICES_DIR / "BOVA11.parquet", columns=["trade_date", "adj_close"])
    bova_curve = buy_and_hold_curve(bova.set_index("trade_date")["adj_close"].reindex(alpha_curve.index).ffill())
    cdi_only_curve = cdi_curve(cdi[cdi["trade_date"].isin(alpha_curve.index)])

    print("Building dashboard...")
    fig = build_dashboard(alpha_curve, alpha_log, eq_curve, bova_curve, cdi_only_curve,
                           top_frac, hold_frac, top_holdings)
    path = save_dashboard(fig, tag="manual")
    print(f"Wrote {path}")
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--horizon-td", type=int, default=252)
    parser.add_argument("--top-frac", type=float, default=0.6)
    parser.add_argument("--hold-frac", type=float, default=0.75)
    parser.add_argument("--top-holdings", type=int, default=15, help="how many tickers to show individually")
    args = parser.parse_args()
    main(top_n=args.top_n, horizon_td=args.horizon_td, top_frac=args.top_frac,
         hold_frac=args.hold_frac, top_holdings=args.top_holdings)
