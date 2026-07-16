"""EIIE portfolio-management agent (Jiang, Xu & Liang 2017), adapted to daily B3 data.

Iteration 1: price-only features, top-50 dynamic quarterly universe (2011-2026),
CDI-accruing cash. See docs/EIIE_AGENT_PLAN.md for the full design and the
documented deviations from the original paper.

Modules land incrementally per the plan's phases; this file grows an export
as each one is implemented instead of importing modules that don't exist yet.
"""

from . import metrics
from .baselines import BASELINE_NAMES, run_baseline
from .config import ExperimentConfig
from .data import CASH_GIDX, GlobalAssetIndex, PricePanel, load_price_panel
from .environment import BacktestResult, drift_weights, run_backtest, solve_mu, solve_mu_torch
from .pvm import PortfolioVectorMemory

__all__ = [
    "ExperimentConfig",
    "CASH_GIDX",
    "GlobalAssetIndex",
    "PricePanel",
    "load_price_panel",
    "PortfolioVectorMemory",
    "BacktestResult",
    "drift_weights",
    "run_backtest",
    "solve_mu",
    "solve_mu_torch",
    "metrics",
    "BASELINE_NAMES",
    "run_baseline",
]
