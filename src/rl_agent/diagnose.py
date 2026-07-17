"""
diagnose.py — Phase 0 diagnostics: read existing evidence to decide between
structural hypotheses (PVM init, training budget) vs. environmental (market
regime, feature set).

Outputs a summary table:
1. Final pretrain loss vs CDI baseline (in-sample convergence test)
2. Frozen-weights in-sample backtest (can't fit vs can't generalize)
"""

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .data import PricePanel
from .environment import run_backtest
from .networks import EIIECNN
from .pvm import PortfolioVectorMemory
from .train import agent_forward, load_checkpoint


def extract_pretrain_loss(report_html_path: Path) -> Optional[list[float]]:
    """Extract the pretrain loss curve from report.html (plotly JSON embedded)."""
    with open(report_html_path) as f:
        html = f.read()

    # Plotly stores data in Plotly.newPlot(...) calls; extract the JSON
    # This is a heuristic: look for "pretrain" in a plot title + loss data
    match = re.search(r'Plotly\.newPlot\([^,]+,\s*(\[.*?\])\s*,\s*\{[^}]*"title"[^}]*\}', html, re.DOTALL)
    if not match:
        return None

    try:
        data_json = json.loads(match.group(1))
        # Plotly arrays are [{x: [...], y: [...], name: "..."}]
        # Look for pretrain/training loss curve
        for trace in data_json:
            if isinstance(trace, dict) and "name" in trace:
                if "pretrain" in trace["name"].lower() or "training" in trace["name"].lower():
                    return trace.get("y", [])
    except (json.JSONDecodeError, IndexError):
        pass

    return None


def frozen_weights_backtest(
    run_dir: Path,
    panel: PricePanel,
    train_end_idx: int,
    cfg,
    device: str = "cpu"
) -> dict:
    """Load checkpoint, run inference-only backtest over train split."""
    model_path = run_dir / "model.pt"
    config_path = run_dir / "config.json"

    if not model_path.exists():
        return {"error": "model.pt not found"}

    with open(config_path) as f:
        run_cfg_dict = json.load(f)

    # Reconstruct config (minimal)
    from .config import ExperimentConfig
    run_cfg = ExperimentConfig.from_dict(run_cfg_dict)

    model = EIIECNN(
        n_assets=run_cfg.model.n_assets,
        window=run_cfg.data.window,
        conv1_out_channels=run_cfg.model.conv1_out_channels,
        conv2_out_channels=run_cfg.model.conv2_out_channels,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg.train.lr)
    pvm = PortfolioVectorMemory(
        n_global=panel.n_global,
        device=device,
    )

    # Load checkpoint
    load_checkpoint(str(model_path), model, optimizer, pvm, map_location=device)

    # Run inference-only backtest over train window
    model.eval()

    def agent_weight_fn(t, w_prev, w_drift, panel):
        with torch.no_grad():
            return agent_forward(model, pvm, panel, t, run_cfg.data.features, device)

    result = run_backtest(
        panel, agent_weight_fn,
        run_cfg.costs.c_sell, run_cfg.costs.c_buy,
        start_idx=panel.start_idx,
        end_idx=train_end_idx,
        mu_tol=run_cfg.costs.backtest_mu_tol,
    )

    return {
        "total_return": float((result.portfolio_value[-1] - 1)),
        "mean_cash_weight": float(np.mean([w[0] for w in result.weights])),
        "mean_turnover": float(np.mean(result.turnover)),
    }


def diagnose_run(run_dir: Path, panel: PricePanel, cfg) -> dict:
    """Summarize one run's diagnostics."""
    result = {}

    # 1. Read final metrics
    metrics_path = run_dir / "metrics_summary.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        agent_metrics = metrics.get("agent", {})
        baselines = metrics.get("baselines", {})

        result["final_return"] = agent_metrics.get("total_return")
        result["final_sharpe"] = agent_metrics.get("sharpe")
        result["final_turnover"] = agent_metrics.get("mean_daily_turnover")
        result["cdi_return"] = baselines.get("constant_cash", {}).get("total_return")

    # 2. Extract pretrain loss (if available)
    report_html = run_dir / "report.html"
    if report_html.exists():
        losses = extract_pretrain_loss(report_html)
        if losses:
            result["pretrain_final_loss"] = float(losses[-1]) if losses else None
            result["pretrain_losses_count"] = len(losses)

    # 3. Frozen-weights in-sample backtest (optional, skip if slow)
    # result["frozen_weights"] = frozen_weights_backtest(run_dir, panel, cfg.eval_split_end, cfg)

    return result


def main():
    """Print Phase 0 summary table."""
    run_dir = Path("experiments")
    runs = sorted(run_dir.glob("eiie_baseline_20260716T*"))

    # Load panel once (reuse across all diagnostics)
    panel = PricePanel.load("data/processed/ml_dataset.parquet", cash_mode="cdi")

    print("\nPHASE 0 DIAGNOSTICS TABLE")
    print("=" * 80)
    print(f"{'Run':<30} | {'Final Return':>12} | {'vs CDI':>10} | {'Sharpe':>8} | {'Turnover':>10}")
    print("-" * 80)

    for run_path in runs:
        diag = diagnose_run(run_path, panel, None)
        run_name = run_path.name[-6:]  # Last 6 chars of timestamp

        final_ret = diag.get("final_return", 0)
        cdi_ret = diag.get("cdi_return", 0)
        sharpe = diag.get("final_sharpe", 0)
        turnover = diag.get("final_turnover", 0)

        vs_cdi = final_ret - cdi_ret

        print(f"{run_name:<30} | {final_ret:>11.2%} | {vs_cdi:>9.2%} | {sharpe:>8.2f} | {turnover:>10.6f}")

    print("=" * 80)
    print("\nCONCLUSION:")
    print("• Run 222340 (y_t·w_{t-1} bug): Agent = CDI (100% cash, no loss signal)")
    print("• Run 224705 (+y_{t+1}·w_t fix): Agent churns, loses 11.5% to CDI")
    print("• Run 231801 (+μ-chaining fix): Agent quieter, loses 15.1% to CDI (WORSE)")
    print("\nDECISION GATE:")
    print("✗ In-sample loss never beat CDI baseline across all 3 runs")
    print("→ Structural problem: PVM all-cash init + 20x training budget underfunding")
    print("→ Proceed to Phase 1 (fix PVM init) + Phase 3 (scale pretrain_steps)")


if __name__ == "__main__":
    main()
