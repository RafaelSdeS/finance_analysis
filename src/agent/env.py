"""
PortfolioEnv: gymnasium environment for daily portfolio allocation.

State:  normalized features for all tickers + activity mask + prev weights
        obs = [n_tickers * n_features + 2*n_tickers]  (flattened, float32)
Action: raw logits [n_tickers]; env applies masked softmax so that
        inactive tickers get exactly 0 weight and active weights sum to 1.
Reward: daily excess log return vs equal-weight (variance-reduced training signal).
        info["log_return"] carries absolute portfolio log return for backtest metrics.

Prerequisite: run `python -m src.agent.data_pipeline` once to build
data/processed/agent_tensors.npz and data/models/feature_scaler.pkl.
"""

import logging
import pickle
from functools import lru_cache

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.agent.config import AgentConfig, DEFAULT_CONFIG
from src.agent.data_pipeline import SCALER_PATH, TENSORS_PATH

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _load_slice(start: str, end: str):
    """Slice + mask the shared tensors once per (start, end) and cache the result.

    The online-backtest retrain loop builds a 16-worker DummyVecEnv per chunk where
    every worker uses the SAME date bounds; without this cache each of the 16 identical
    PortfolioEnv instances independently copies the (growing, anchored-at-2000) train
    slice, i.e. 16x redundant multi-hundred-MB arrays rebuilt every chunk. Safe to share
    by reference since PortfolioEnv never mutates features/mask/returns after __init__.
    """
    dates, tickers, mask, returns, feats_all = _load_normalized_tensors()
    sel = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    sliced_mask = mask[sel]
    feats = feats_all[sel].copy()
    feats[~sliced_mask] = 0.0
    return dates[sel], tickers, sliced_mask, returns[sel], feats.astype(np.float32)


@lru_cache(maxsize=1)
def _load_normalized_tensors():
    """Load + normalize the full tensor dataset once per process.

    TENSORS_PATH/SCALER_PATH are fixed constants, so every PortfolioEnv
    construction (e.g. the 16 DummyVecEnv workers rebuilt per online-backtest
    chunk) reuses this instead of re-reading a 21.5MB npz + rescaling from
    scratch each time.
    """
    data = np.load(TENSORS_PATH, allow_pickle=True)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    dates = pd.to_datetime(data["dates"])
    tickers = data["tickers"]
    mask = data["mask"]
    returns = data["returns"]
    feats = (data["features"] - scaler.mean_) / scaler.scale_
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return dates, tickers, mask, returns, feats


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

        bounds = {
            "train": (config.train_start, config.train_end),
            "val": (config.val_start, config.val_end),
            "test": (config.test_start, config.test_end),
        }
        if date_range not in bounds:
            raise ValueError(f"date_range must be one of {list(bounds)}, got '{date_range}'")
        start, end = bounds[date_range]
        self.dates, self.tickers, self.mask, self.returns, self.features = _load_slice(start, end)
        n_tickers = len(self.tickers)
        n_features = self.features.shape[2]

        self.n_steps = len(self.dates) - 1  # last date has no next-day return
        if self.n_steps < 2:
            raise ValueError(f"Date range '{date_range}' has too few days: {len(self.dates)}")

        self._is_cash_mask = self.tickers == "CASH"  # precomputed, avoid per-step string compare

        obs_dim = n_tickers * n_features + 2 * n_tickers  # features + mask + prev_weights
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-10.0, 10.0, shape=(n_tickers,), dtype=np.float32)

        self._t = 0
        self.portfolio_value = config.initial_capital
        self._prev_weights = self._is_cash_mask.astype(np.float32)  # start 100% CASH (new investor)
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
        self._prev_weights[:] = self._is_cash_mask  # fresh start: 100% CASH; first allocation incurs full deployment cost
        return self._obs(), {"date": self.dates[0], "portfolio_value": self.portfolio_value}

    def step(self, action: np.ndarray):
        weights = self._masked_softmax(action, self.mask[self._t])

        # Transaction cost: cost per unit of traded notional, excluding CASH leg (which trades free)
        # ponytail: prev weights not drift-adjusted for returns; model sees absolute weight deltas,
        # acceptable since agent plans daily rebalance anyway (one-step horizon)
        traded = np.abs(weights - self._prev_weights)[~self._is_cash_mask].sum()
        transaction_cost = traded * (self.config.transaction_cost_bps / 10_000)

        # Next-day log returns; inactive-tomorrow or missing → 0 (position carried flat)
        next_returns = np.nan_to_num(self.returns[self._t + 1], nan=0.0)
        next_returns = np.where(self.mask[self._t + 1], next_returns, 0.0)

        simple_return = float(np.dot(weights, np.expm1(next_returns))) - transaction_cost
        # Clip at -99.99% to keep log finite even on catastrophic days
        portfolio_log_return = float(np.log1p(max(simple_return, -0.9999)))

        # Equal-weight baseline for variance-reduced training signal
        # (excludes CASH, so baseline = 1/n_active_noncash of active non-cash tickers)
        active_noncash = self.mask[self._t + 1] & ~self._is_cash_mask
        n_active_noncash = int(active_noncash.sum())
        if n_active_noncash > 0:
            ew_return = float(np.expm1(next_returns[active_noncash]).mean())
        else:
            ew_return = 0.0
        ew_log_return = float(np.log1p(max(ew_return, -0.9999)))

        # Reward: excess log return (agent vs equal-weight) for credit assignment
        reward = portfolio_log_return - ew_log_return

        self.portfolio_value *= 1.0 + max(simple_return, -0.9999)
        self._prev_weights = weights.copy()

        self._t += 1
        terminated = self._t >= self.n_steps
        info = {
            "date": self.dates[self._t],
            "portfolio_value": self.portfolio_value,
            "weights": weights,
            "log_return": portfolio_log_return,  # absolute return for backtest metrics
            "n_active": int(self.mask[self._t].sum()),
            "turnover": float(traded),
        }
        return self._obs(), reward, terminated, False, info

    # -------------------------------------------------------------- helpers

    def _obs(self) -> np.ndarray:
        t = min(self._t, self.n_steps)
        return np.concatenate(
            [self.features[t].ravel(), self.mask[t].astype(np.float32), self._prev_weights]
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
