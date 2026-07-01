"""
PortfolioEnv: gymnasium environment for daily portfolio allocation.

State:  normalized features for all tickers + activity mask
        obs = [n_tickers * n_features + n_tickers]  (flattened, float32)
Action: raw logits [n_tickers]; env applies masked softmax so that
        inactive tickers get exactly 0 weight and active weights sum to 1.
Reward: daily log return of the portfolio (next trading day).

Prerequisite: run `python -m src.agent.data_pipeline` once to build
data/processed/agent_tensors.npz and data/models/feature_scaler.pkl.
"""

import logging
import pickle

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.agent.config import AgentConfig, DEFAULT_CONFIG
from src.agent.data_pipeline import SCALER_PATH, TENSORS_PATH

logger = logging.getLogger(__name__)


class PortfolioEnv(gym.Env):
    """Daily rebalancing over a masked, time-varying ticker universe."""

    metadata = {"render_modes": []}

    def __init__(self, config: AgentConfig = DEFAULT_CONFIG, date_range: str = "train"):
        """
        Args:
            config: AgentConfig with paths and split dates.
            date_range: "train", "val", or "test" (selects date slice).
        """
        super().__init__()
        self.config = config
        self.date_range = date_range

        data = np.load(TENSORS_PATH, allow_pickle=True)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        dates = pd.to_datetime(data["dates"])
        self.tickers: np.ndarray = data["tickers"]
        n_tickers = len(self.tickers)
        n_features = data["features"].shape[2]

        # Slice the requested date range
        bounds = {
            "train": (config.train_start, config.train_end),
            "val": (config.val_start, config.val_end),
            "test": (config.test_start, config.test_end),
        }
        if date_range not in bounds:
            raise ValueError(f"date_range must be one of {list(bounds)}, got '{date_range}'")
        start, end = (pd.Timestamp(b) for b in bounds[date_range])
        sel = (dates >= start) & (dates <= end)

        self.dates = dates[sel]
        self.mask = data["mask"][sel]                       # [T, N] bool
        self.returns = data["returns"][sel]                 # [T, N] log returns, NaN if inactive

        # Normalize once with the train-only scaler; NaN and inactive cells → 0 (mean-imputed)
        feats = (data["features"][sel] - scaler.mean_) / scaler.scale_
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        feats[~self.mask] = 0.0
        self.features = feats.astype(np.float32)            # [T, N, F]

        self.n_steps = len(self.dates) - 1  # last date has no next-day return
        if self.n_steps < 2:
            raise ValueError(f"Date range '{date_range}' has too few days: {len(self.dates)}")

        obs_dim = n_tickers * n_features + n_tickers
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-10.0, 10.0, shape=(n_tickers,), dtype=np.float32)

        self._t = 0
        self.portfolio_value = config.initial_capital
        logger.info(
            "PortfolioEnv[%s]: %d days (%s → %s), %d tickers, obs_dim=%d",
            date_range, len(self.dates), self.dates[0].date(), self.dates[-1].date(),
            n_tickers, obs_dim,
        )

    # ------------------------------------------------------------------ API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._t = 0
        self.portfolio_value = self.config.initial_capital
        return self._obs(), {"date": self.dates[0], "portfolio_value": self.portfolio_value}

    def step(self, action: np.ndarray):
        weights = self._masked_softmax(action, self.mask[self._t])

        # Next-day log returns; inactive-tomorrow or missing → 0 (position carried flat)
        next_returns = np.nan_to_num(self.returns[self._t + 1], nan=0.0)
        next_returns = np.where(self.mask[self._t + 1], next_returns, 0.0)

        simple_return = float(np.dot(weights, np.expm1(next_returns)))
        # Clip at -99.99% to keep log finite even on catastrophic days
        reward = float(np.log1p(max(simple_return, -0.9999)))
        self.portfolio_value *= 1.0 + max(simple_return, -0.9999)

        self._t += 1
        terminated = self._t >= self.n_steps
        info = {
            "date": self.dates[self._t],
            "portfolio_value": self.portfolio_value,
            "weights": weights,
            "n_active": int(self.mask[self._t].sum()),
        }
        return self._obs(), reward, terminated, False, info

    # -------------------------------------------------------------- helpers

    def _obs(self) -> np.ndarray:
        t = min(self._t, self.n_steps)
        return np.concatenate(
            [self.features[t].ravel(), self.mask[t].astype(np.float32)]
        )

    @staticmethod
    def _masked_softmax(logits: np.ndarray, active: np.ndarray) -> np.ndarray:
        """Softmax over active tickers only; inactive get exactly 0 weight."""
        z = np.where(active, logits.astype(np.float64), -np.inf)
        z -= z.max()  # numerical stability
        e = np.exp(z)
        total = e.sum()
        if total == 0 or not np.isfinite(total):  # no active tickers (shouldn't happen)
            return active.astype(np.float64) / max(active.sum(), 1)
        return e / total
