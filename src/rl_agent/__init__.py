"""EIIE portfolio-management agent (Jiang, Xu & Liang 2017), adapted to daily B3 data.

Iteration 1: price-only features, top-50 dynamic quarterly universe (2011-2026),
CDI-accruing cash. See docs/EIIE_AGENT_PLAN.md for the full design and the
documented deviations from the original paper.

Modules land incrementally per the plan's phases; this file grows an export
as each one is implemented instead of importing modules that don't exist yet.
"""

from .config import ExperimentConfig
from .data import CASH_GIDX, GlobalAssetIndex, PricePanel, load_price_panel

__all__ = [
    "ExperimentConfig",
    "CASH_GIDX",
    "GlobalAssetIndex",
    "PricePanel",
    "load_price_panel",
]
