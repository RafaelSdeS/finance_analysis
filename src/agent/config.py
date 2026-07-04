"""
Agent Configuration

Immutable configuration for RL agent training, validation, and inference.
All hyperparameters and paths defined here (zero hardcoding).

Training is always via anchored rolling windows (see `generate_windows()`):
`DEFAULT_CONFIG` is the config for the most recent window (train/val/test
carved from it), so downstream evaluation/inference modules that read
`config.test_start`/`test_end` automatically target the newest held-out
period without any special-casing.
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# Absolute project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AgentConfig:
    """Immutable configuration for ML agent."""

    # ===== Data & Paths =====
    data_dir: Path = _PROJECT_ROOT / "data/processed"
    model_dir: Path = _PROJECT_ROOT / "data/models"
    backtest_dir: Path = _PROJECT_ROOT / "data/backtest"
    log_dir: Path = _PROJECT_ROOT / "data/logs"

    dataset_path: Path = _PROJECT_ROOT / "data/processed/ml_dataset_training.parquet"

    # ===== Temporal Splits (by date, not rows) =====
    # Template defaults only — never used directly. Real instances are built
    # per rolling window via `window_to_config()`; see `DEFAULT_CONFIG` below.
    train_start: str = "2000-01-03"
    train_end: str = "2015-11-25"

    val_start: str = "2015-11-26"
    val_end: str = "2021-03-13"

    test_start: str = "2021-03-14"
    test_end: str = "2026-06-30"

    # ===== Anchored Rolling Windows (the sole training strategy) =====
    dataset_start: str = "2000-01-03"
    dataset_end: str = "2026-06-30"
    window_train_years: int = 10  # each window's train span, always anchored at dataset_start
    window_test_years: int = 2    # each window's held-out test span
    window_val_fraction: float = 0.15  # tail fraction of a window's train span carved out for early-stopping val

    # ===== Feature Configuration =====
    # Price & technical features
    price_features: list[str] = field(
        default_factory=lambda: [
            "open", "high", "low", "close", "volume",
            "returns",  # Computed from prices
        ]
    )

    # Fundamental features (quarterly, forward-filled)
    fundamental_features: list[str] = field(
        default_factory=lambda: [
            "pl",  # Price-to-Earnings (P/L in Portuguese)
            "pvp",  # Price-to-Book (P/VP in Portuguese)
            "roe", "debt_equity", "roic", "roa",
            "net_margin", "gross_margin", "ebitda_margin",
            "current_ratio", "cash_ratio",
            # Growth features (YoY, not CAGR to avoid synthetic data)
            "earnings_growth_yoy", "revenue_growth_yoy", "ebitda_growth_yoy",
        ]
    )

    # Macro features (daily, from BCB SGS)
    macro_features: list[str] = field(
        default_factory=lambda: [
            "selic", "cdi", "ipca",  # Interest rates, inflation
        ]
    )

    # Combined state features (price + fundamental + macro, excluding meta)
    @property
    def state_features(self) -> list[str]:
        """All features that go into agent state (normalized)."""
        return self.price_features + self.fundamental_features + self.macro_features

    # ===== Agent Hyperparameters =====
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE smoothing
    entropy_coef: float = 0.0  # No entropy bonus; excess reward removes market noise (variance reduction)
    log_std_init: float = -2.0  # Initial exploration noise: σ ≈ 0.135 so per-ticker credit assignment is learnable

    # ===== Training Configuration =====
    total_timesteps: int = 1_000_000
    n_envs: int = 8  # Parallel rollout workers (SubprocVecEnv); 1 = single-process
    n_steps: int = 2048  # Trajectory length per update, PER env
    batch_size: int = 64
    n_epochs: int = 10  # Gradient updates per rollout

    # ===== Checkpointing & Early Stopping =====
    eval_freq: int = 20  # Evaluate on val set every N episodes (20 * n_steps = 40,960 timesteps)
    early_stopping_patience: int = 3  # Stop if val Sharpe degrades 3x in a row

    # ===== Logging =====
    log_file_prefix: str = "agent_training"

    # ===== Portfolio Constraints =====
    initial_capital: float = 100_000.0  # R$ (Brazilian Real)

    # ===== Transaction Costs =====
    transaction_cost_bps: float = 10.0  # cost per unit of traded notional (B3 fees ~3bps + slippage margin)

    # ===== Misc =====
    seed: int = 42
    device: str = "cuda"  # or "cpu"
    verbose: int = 0  # 0=silent, 1=progress, 2=debug

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Ensure paths exist
        for path in [self.data_dir, self.model_dir, self.backtest_dir, self.log_dir]:
            path.mkdir(parents=True, exist_ok=True)

        # Validate dataset exists
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}\n"
                f"Run: python src/agent/feature_engineering.py"
            )

        # Validate date ranges
        if self.train_end >= self.val_start:
            raise ValueError(f"train_end ({self.train_end}) >= val_start ({self.val_start})")
        if self.val_end >= self.test_start:
            raise ValueError(f"val_end ({self.val_end}) >= test_start ({self.test_start})")

    def log_summary(self) -> None:
        """Print configuration summary."""
        print(
            f"AgentConfig | {self.dataset_path.name} | "
            f"{len(self.state_features)} features "
            f"({len(self.price_features)}p+{len(self.fundamental_features)}f+{len(self.macro_features)}m)\n"
            f"  splits: train {self.train_start}→{self.train_end} | "
            f"val {self.val_start}→{self.val_end} | test {self.test_start}→{self.test_end}\n"
            f"  ppo: lr={self.learning_rate} γ={self.gamma} λ={self.gae_lambda} "
            f"steps={self.total_timesteps:,} batch={self.batch_size} epochs={self.n_epochs}"
        )


class RollingWindow(NamedTuple):
    """A single anchored train/test window."""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def generate_windows(
    dataset_start: str, dataset_end: str, train_years: int, test_years: int,
) -> list[RollingWindow]:
    """
    Generate anchored rolling windows.

    Train always starts at dataset_start, test window slides forward.
    Non-overlapping windows preserve temporal integrity (no lookahead).
    """
    start_date = pd.Timestamp(dataset_start)
    end_date = pd.Timestamp(dataset_end)

    windows = []
    window_id = 0
    test_start = start_date + pd.DateOffset(years=train_years)

    while test_start + pd.DateOffset(years=test_years) <= end_date:
        test_end = test_start + pd.DateOffset(years=test_years)
        train_end = test_start - pd.Timedelta(days=1)

        windows.append(RollingWindow(
            window_id=window_id,
            train_start=start_date.strftime("%Y-%m-%d"),
            train_end=train_end.strftime("%Y-%m-%d"),
            test_start=test_start.strftime("%Y-%m-%d"),
            test_end=test_end.strftime("%Y-%m-%d"),
        ))

        window_id += 1
        test_start = test_end + pd.Timedelta(days=1)

    return windows


def _carve_val_tail(train_start: str, train_end: str, val_fraction: float) -> tuple[str, str]:
    """Tail-slice a window's train span: returns (new_train_end, val_start); val_end stays train_end."""
    ts, te = pd.Timestamp(train_start), pd.Timestamp(train_end)
    val_days = max(int((te - ts).days * val_fraction), 1)
    val_start = te - pd.Timedelta(days=val_days - 1)
    new_train_end = val_start - pd.Timedelta(days=1)
    return new_train_end.strftime("%Y-%m-%d"), val_start.strftime("%Y-%m-%d")


def window_to_config(window: RollingWindow, base: "AgentConfig") -> "AgentConfig":
    """Build a self-consistent per-window AgentConfig (train tail carved out for early-stopping val)."""
    train_end, val_start = _carve_val_tail(window.train_start, window.train_end, base.window_val_fraction)
    return dataclasses.replace(
        base,
        train_start=window.train_start, train_end=train_end,
        val_start=val_start, val_end=window.train_end,
        test_start=window.test_start, test_end=window.test_end,
    )


# Default configuration: the AgentConfig for the MOST RECENT rolling window.
# Downstream modules (evaluate.py, infer.py, run_allocation.py) read
# config.test_start/test_end and therefore automatically target the newest
# held-out period, with zero special-casing.
_template = AgentConfig()
_windows = generate_windows(
    _template.dataset_start, _template.dataset_end,
    _template.window_train_years, _template.window_test_years,
)
DEFAULT_CONFIG = window_to_config(_windows[-1], _template)


if __name__ == "__main__":
    DEFAULT_CONFIG.log_summary()
