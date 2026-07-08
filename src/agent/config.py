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
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# Absolute project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def configure_logging(log_dir: Path, run_id: str, tag: str = "session") -> Path:
    """Set up root logger once: console (as before) + a persisted per-session file.

    Safe to call from multiple entry points in the same process — a second
    call is a no-op and just returns the (deterministic) path again.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{tag}_{run_id}.log"
    root = logging.getLogger()
    if root.handlers:
        return log_path

    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(file_handler)

    return log_path


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
    window_id: int = -1  # stamped by window_to_config(); -1 = template default, never used directly

    # ===== Universe Filter (optional) =====
    universe_size: int | None = None  # top-N tickers by mean market cap; None = no filtering (all tickers with >=252 rows)

    # ===== Feature Configuration =====
    # 2026-07: pruned to the top-15 by Random Forest importance (R²=0.233 on
    # cleaned data; linear-regression ranking was discarded as unreliable —
    # R²≈0 there, so its "top" picks were scale artifacts, not signal).
    # Lower-ranked features are commented out, not deleted — uncomment to
    # bring any of them back into state_features with no other code changes.
    # Price & technical features (stationary, relative; raw OHLC dropped)
    price_features: list[str] = field(
        default_factory=lambda: [
            "returns",  # Computed from prices; stationary. Hard dependency: data_pipeline.py needs this exact name to build the returns tensor — never remove.
            # "volume",
            # "rsi_14",
            # "volatility_20d",
            # "volatility_60d",
            # "drawdown",
            # "momentum_vs_market_1m",
            "momentum_vs_market_3m",  # RF rank 3
            # "momentum_vs_market_12m",
            # "momentum_vs_sector_1m",
            "momentum_vs_sector_3m",  # RF rank 1
            "momentum_vs_sector_12m",  # RF rank 14
            "real_return",  # RF rank 4
            # "excess_return",  # RF rank 7 — REMOVED: this is regime (market return), not alpha
            # "ma_20",  # RF rank 10 — REMOVED: negative IC in ranker (mean reversion noise)
            # "ma_60",  # RF rank 5 — REMOVED: negative IC in ranker (mean reversion noise)
            "price_percentile_5y",  # RF rank 6
            "drawdown_percentile",  # RF rank 8
            "return_6m",  # RF rank 12
        ]
    )

    # Fundamental features (quarterly, forward-filled + sector-relative + quality trends)
    fundamental_features: list[str] = field(
        default_factory=lambda: [
            # "pl",  # Price-to-Earnings (P/L in Portuguese)
            # "pvp",  # Price-to-Book (P/VP in Portuguese)
            # "roe",
            # "debt_equity",
            # "roic",
            "roa",  # Return on Assets — HIGH IC in ranker (0.0968 at 21d!)
            # "net_margin",
            # "gross_margin",
            # "ebitda_margin",
            # "current_ratio",
            # "cash_ratio",
            # Growth features (YoY, not CAGR to avoid synthetic data)
            # "earnings_growth_yoy",
            # "revenue_growth_yoy",
            # "ebitda_growth_yoy",
            # Sector-relative valuation & quality
            "pl_zscore_sector",  # Valuation relative to sector — helps identify cheap/expensive
            # "pvp_zscore_sector",
            # "roe_zscore_sector",
            # "debt_equity_zscore_sector",
            # Quality trends & signals
            "f_score",  # Piotroski F-Score — quality metric, high IC in ranker baseline
            # "roe_trend_4q",
            # "margin_trend_4q",
            # "earnings_yield_vs_selic",
            # Dividend signals
            "div_yield_12m",  # RF rank 2
            # "payout_ratio",
            # 1.0 once the ticker's first filing exists, else 0.0 — lets the
            # model tell "no data yet" apart from "average company" after
            # the env's NaN→0 (post-scaling mean) imputation. Not individually
            # RF-ranked (it's a flag, not a continuous signal) — commented out
            # under the "top-15 only" rule; uncomment if this distinction
            # turns out to matter for young/newly-listed tickers.
            # "has_fundamentals",
            "days_since_fundamental",  # RF rank 11
            "debt_trend_4q",  # RF rank 15
        ]
    )

    # Macro features (daily, from BCB SGS)
    macro_features: list[str] = field(
        default_factory=lambda: [
            # "selic",
            # "cdi",
            "ipca",  # RF rank 13
            # "selic_trend_20d",
        ]
    )

    # Combined state features (price + fundamental + macro, excluding meta)
    @property
    def state_features(self) -> list[str]:
        """All features that go into agent state (normalized)."""
        return self.price_features + self.fundamental_features + self.macro_features

    @property
    def effective_gamma(self) -> float:
        """Per-decision discount factor, accounting for N-day aggregation.

        gamma is per-day (0.997); with rebalance_interval_days=N, one env.step()
        spans N days, so the effective discount per step is gamma^N.
        """
        return self.gamma ** self.rebalance_interval_days

    # ===== Agent Hyperparameters =====
    learning_rate: float = 3e-4
    gamma: float = 0.997  # Discount factor; effective horizon ~333 trading days (~1.3y)
    gae_lambda: float = 0.95  # GAE smoothing
    entropy_coef: float = 0.001  # Small entropy bonus to encourage exploration
    log_std_init: float = -2.0  # Exploration noise σ ≈ 0.135; combined with logit_scale → effective logit noise ~1.35
    # Softmax temperature applied to actions inside the env (weights = softmax(action * logit_scale)).
    # Why: PPO's trust region (target_kl) bounds movement in *Gaussian* action space; at scale 1 the
    # softmax over 280 assets needs O(1) logit spreads to concentrate, which takes millions of steps
    # of ~0.007-logit updates (empirically the policy stays frozen at uniform = equal-weight).
    # Scaling by 10 gives 10x weight-space movement and exploration per unit of trust region.
    logit_scale: float = 10.0

    # ===== Training Configuration =====
    total_timesteps: int = 1_000_000
    n_envs: int = 8  # In-process envs (DummyVecEnv) batched N-wide through the policy
    n_steps: int = 2048  # Trajectory length per update, PER env
    batch_size: int = 64
    n_epochs: int = 10  # Gradient updates per rollout

    # ===== Checkpointing & Early Stopping =====
    eval_freq: int = 20  # Evaluate on val set every N episodes (20 * n_steps = 40,960 timesteps)
    early_stopping_patience: int = 8  # Stop if val Sharpe degrades 8x in a row; generous because concentration initially costs turnover before alpha shows (the "paying to learn" valley)

    # ===== Portfolio Constraints =====
    initial_capital: float = 100_000.0  # R$ (Brazilian Real)

    # ===== Rebalancing Interval =====
    rebalance_interval_days: int = 21  # N-day aggregate steps (1 = legacy daily, 21 = monthly); one step = N days

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

        # Validate rebalance interval
        if self.rebalance_interval_days < 1:
            raise ValueError(f"rebalance_interval_days must be >= 1, got {self.rebalance_interval_days}")

        # Validate date ranges
        if self.train_end >= self.val_start:
            raise ValueError(f"train_end ({self.train_end}) >= val_start ({self.val_start})")
        if self.val_end >= self.test_start:
            raise ValueError(f"val_end ({self.val_end}) >= test_start ({self.test_start})")

    def log_summary(self) -> None:
        """Log configuration summary."""
        logger.info(
            f"AgentConfig | {self.dataset_path.name} | "
            f"{len(self.state_features)} features "
            f"({len(self.price_features)}p+{len(self.fundamental_features)}f+{len(self.macro_features)}m) | "
            f"splits: train {self.train_start}→{self.train_end} | "
            f"val {self.val_start}→{self.val_end} | test {self.test_start}→{self.test_end} | "
            f"ppo: lr={self.learning_rate} γ={self.gamma:.4f} (eff={self.effective_gamma:.4f} per {self.rebalance_interval_days}d step) "
            f"λ={self.gae_lambda} "
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
        window_id=window.window_id,
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


def _selfcheck_logging() -> None:
    """Verify configure_logging: idempotence and file persistence."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "nested" / "run_dir"  # does not exist yet — exercises the mkdir fix
        run_id = "test_run"

        handler_count_before = len(logging.getLogger().handlers)
        path1 = configure_logging(log_dir, run_id, tag="test")
        handler_count_after_1 = len(logging.getLogger().handlers)

        path2 = configure_logging(log_dir, run_id, tag="test")
        handler_count_after_2 = len(logging.getLogger().handlers)

        assert handler_count_after_1 > handler_count_before, "no handlers added on first call"
        assert handler_count_after_2 == handler_count_after_1, "handlers added on second call (not idempotent)"

        assert path1 == path2, "paths differ on second call"
        assert path1.exists(), "log file not created"

        logger.info("test_sentinel_message_12345")
        with open(path1) as f:
            content = f.read()
        assert "test_sentinel_message_12345" in content, "message not persisted to file"

        print("✓ configure_logging self-check passed")


if __name__ == "__main__":
    _selfcheck_logging()
    DEFAULT_CONFIG.log_summary()
