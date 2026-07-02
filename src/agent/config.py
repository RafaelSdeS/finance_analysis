"""
Agent Configuration

Immutable configuration for RL agent training, validation, and inference.
All hyperparameters and paths defined here (zero hardcoding).
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AgentConfig:
    """Immutable configuration for ML agent."""

    # ===== Data & Paths =====
    data_dir: Path = Path("data/processed")
    model_dir: Path = Path("data/models")
    backtest_dir: Path = Path("data/backtest")
    log_dir: Path = Path("data/logs")

    dataset_path: Path = Path("data/processed/ml_dataset_training.parquet")

    # ===== Temporal Splits (by date, not rows) =====
    # Based on dataset verification: 26.49 years (2000-01-03 to 2026-06-30)
    # Recommended split (60/20/20): 15.89y / 5.30y / 5.30y
    train_start: str = "2000-01-03"
    train_end: str = "2015-11-25"

    val_start: str = "2015-11-26"
    val_end: str = "2021-03-13"

    test_start: str = "2021-03-14"
    test_end: str = "2026-06-30"

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
    entropy_coef: float = 0.01  # Exploration bonus

    # ===== Training Configuration =====
    total_timesteps: int = 1_000_000
    n_steps: int = 2048  # Trajectory length per update
    batch_size: int = 64
    n_epochs: int = 10  # Gradient updates per rollout

    # ===== Checkpointing & Early Stopping =====
    eval_freq: int = 100  # Evaluate on val set every N episodes
    early_stopping_patience: int = 3  # Stop if val Sharpe degrades 3x in a row

    # ===== Logging =====
    log_file_prefix: str = "agent_training"

    # ===== Portfolio Constraints =====
    initial_capital: float = 100_000.0  # R$ (Brazilian Real)

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


# Default configuration (can be overridden via CLI args in trainer.py)
DEFAULT_CONFIG = AgentConfig()


if __name__ == "__main__":
    config = AgentConfig()
    config.log_summary()
