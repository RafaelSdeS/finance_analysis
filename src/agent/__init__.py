"""
ML Agent module for portfolio allocation using reinforcement learning.

Submodules:
  - config: Hyperparameters, paths, feature list (AgentConfig dataclass)
  - feature_engineering: Data preparation (compute_returns, prepare_training_dataset)
  - env: PortfolioEnv gymnasium environment
  - trainer: PPO training loop (SB3 MlpPolicy)
  - evaluate: Backtesting and metrics
  - infer: Inference for live allocation
  - run_allocation: Daily entry point
"""

__all__ = [
    "compute_returns",
    "prepare_training_dataset",
]


def __getattr__(name: str):
    if name in __all__:
        from .feature_engineering import compute_returns, prepare_training_dataset
        return {
            "compute_returns": compute_returns,
            "prepare_training_dataset": prepare_training_dataset,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
