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
from src.agent.data_pipeline import SCALER_PATH, TENSORS_PATH, fit_train_scaler

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_raw_tensors():
    """Load raw tensors and per-window scalers once per process.

    TENSORS_PATH contains raw (unscaled) features. SCALER_PATH contains a dict
    of scalers indexed by train_end cutoff date, allowing each window to
    normalize using its own distribution (fitted on that window's train span,
    no lookahead).
    """
    data = np.load(TENSORS_PATH, allow_pickle=True)
    with open(SCALER_PATH, "rb") as f:
        scalers = pickle.load(f)
    dates = pd.to_datetime(data["dates"])
    tickers = data["tickers"]
    mask = data["mask"]
    returns = data["returns"]
    feats = data["features"]
    return dates, tickers, mask, returns, feats, scalers


@lru_cache(maxsize=1)  # windows train sequentially; unbounded cache leaked ~170MB per window
def _load_normalized_tensors(train_end_cutoff: str):
    """Load, normalize, and clip features for a specific window.

    Each window has its own scaler (fitted on that window's train span), keyed
    by its train_end cutoff date — stable regardless of how many windows a
    given (window_train_years, window_test_years) choice produces. If a config
    uses a window layout that `data_pipeline.py` never precomputed a scaler
    for (e.g. an ad-hoc --train-years/--test-years smoke run), fit one now
    from the shared raw tensors: same fit_train_scaler() used to build
    SCALER_PATH, so results are identical to a precomputed run, just paid at
    call time instead of upfront.
    Clipping at ±10 std devs provides defense-in-depth against outliers.
    """
    dates, tickers, mask, returns, feats, scalers = _load_raw_tensors()
    scaler = scalers.get(train_end_cutoff)
    if scaler is None:
        logger.warning(
            "No precomputed scaler for cutoff=%s (available: %s) — fitting on demand. "
            "Run `python -m src.agent.data_pipeline` with matching --train-years/--test-years "
            "to precompute this once and avoid refitting on every process start.",
            train_end_cutoff, sorted(scalers.keys()),
        )
        scaler = fit_train_scaler({"dates": dates, "features": feats, "mask": mask}, train_end_cutoff)
    feats = (feats - scaler.mean_) / scaler.scale_
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats = np.clip(feats, -10.0, 10.0)  # ponytail: outlier defense-in-depth; ±10 std is generous
    return dates, tickers, mask, returns, feats


@lru_cache(maxsize=4)
def _load_slice(start: str, end: str, train_end_cutoff: str):
    """Slice + mask the shared tensors once per (start, end, train_end_cutoff) and cache the result.

    All PortfolioEnv instances with the same bounds (e.g. the N envs of a
    DummyVecEnv) share these arrays by reference — safe since PortfolioEnv
    never mutates them after __init__.

    Besides the raw slices, precomputes everything step() needs that depends
    only on the day index t, never on the action (computed once here instead
    of ~1M+ times in the hot loop):
      simple_rets    [D,N] next-day simple returns, 0 where inactive/NaN
      ew_log_returns [D]   equal-weight (active non-cash) log return per day
      mask_f32       [D,N] mask as float32 for the observation vector
      n_active       [D]   active-ticker count per day
    """
    dates, tickers, mask, returns, feats_all = _load_normalized_tensors(train_end_cutoff)
    sel = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    sliced_mask = mask[sel]
    feats = feats_all[sel].copy()
    feats[~sliced_mask] = 0.0

    sliced_returns = returns[sel]
    simple_rets = np.where(sliced_mask, np.expm1(np.nan_to_num(sliced_returns, nan=0.0)), 0.0)

    is_cash = tickers == "CASH"
    active_noncash = sliced_mask & ~is_cash                       # [D, N]
    n_active_noncash = active_noncash.sum(axis=1)                 # [D]
    ew_sum = np.where(active_noncash, simple_rets, 0.0).sum(axis=1)
    ew_return = np.where(n_active_noncash > 0, ew_sum / np.maximum(n_active_noncash, 1), 0.0)
    ew_log_returns = np.log1p(np.maximum(ew_return, -0.9999))     # [D]

    derived = {
        "simple_rets": simple_rets,
        "ew_log_returns": ew_log_returns,
        "mask_f32": sliced_mask.astype(np.float32),
        "n_active": sliced_mask.sum(axis=1),
    }
    return dates[sel], tickers, sliced_mask, sliced_returns, feats.astype(np.float32), derived


class PortfolioEnv(gym.Env):
    """Daily rebalancing over a masked, time-varying ticker universe."""

    metadata = {"render_modes": []}

    def __init__(self, config: AgentConfig = DEFAULT_CONFIG, date_range: str = "train"):
        """
        Args:
            config: AgentConfig with paths, split dates, and train_end (per-window scaler key).
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
        self.dates, self.tickers, self.mask, self.returns, self.features, derived = _load_slice(start, end, config.train_end)
        self._simple_rets = derived["simple_rets"]       # [D,N] next-day simple returns, 0 if inactive
        self._ew_log_returns = derived["ew_log_returns"] # [D] equal-weight log return per day
        self._mask_f32 = derived["mask_f32"]             # [D,N] mask as float32 for obs
        self._n_active = derived["n_active"]             # [D] active count per day
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

        # Next-day simple returns and equal-weight baseline are precomputed per
        # day in _load_slice (they depend only on t, not the action).
        simple_return = float(np.dot(weights, self._simple_rets[self._t + 1])) - transaction_cost
        # Clip at -99.99% to keep log finite even on catastrophic days
        portfolio_log_return = float(np.log1p(max(simple_return, -0.9999)))

        # Reward: excess log return (agent vs equal-weight) for credit assignment
        reward = portfolio_log_return - float(self._ew_log_returns[self._t + 1])

        # Detect reward anomalies early (should be roughly ±0.02 daily, anything >±0.5 is extreme)
        if np.isnan(reward) or np.isinf(reward):
            logger.error(
                "Invalid reward at t=%d (date=%s): reward=%.6f, portfolio_return=%.6f, ew_return=%.6f, "
                "portfolio_value=%.2f, weights_sum=%.4f",
                self._t, self.dates[self._t], reward, portfolio_log_return,
                float(self._ew_log_returns[self._t + 1]), self.portfolio_value, float(np.sum(weights))
            )
            raise ValueError(f"Invalid reward at timestep {self._t}: {reward}")
        if abs(reward) > 1.0:
            logger.warning("Extreme reward at t=%d: %.6f (normal range ~±0.02)", self._t, reward)

        self.portfolio_value *= 1.0 + max(simple_return, -0.9999)
        self._prev_weights = weights.copy()

        self._t += 1
        terminated = self._t >= self.n_steps
        info = {
            "date": self.dates[self._t],
            "portfolio_value": self.portfolio_value,
            "weights": weights,
            "log_return": portfolio_log_return,  # absolute return for backtest metrics
            "n_active": int(self._n_active[self._t]),
            "turnover": float(traded),
        }
        return self._obs(), reward, terminated, False, info

    # -------------------------------------------------------------- helpers

    def _obs(self) -> np.ndarray:
        t = min(self._t, self.n_steps)
        obs = np.concatenate(
            [self.features[t].ravel(), self._mask_f32[t], self._prev_weights]
        )
        # Detect NaNs early before they propagate into the network
        if np.isnan(obs).any():
            nan_count = np.isnan(obs).sum()
            features_t = self.features[t].ravel()
            logger.error(
                "NaN detected in observation at t=%d: %d NaN values total. "
                "features_nan=%d mask_nan=%d weights_nan=%d. "
                "features_range=[%.2e, %.2e], max_abs=%.2e. "
                "prev_weights_sum=%.4f",
                t, nan_count,
                np.isnan(features_t).sum(), np.isnan(self._mask_f32[t]).sum(), np.isnan(self._prev_weights).sum(),
                np.nanmin(features_t) if not np.all(np.isnan(features_t)) else np.nan,
                np.nanmax(features_t) if not np.all(np.isnan(features_t)) else np.nan,
                np.nanmax(np.abs(features_t)) if not np.all(np.isnan(features_t)) else np.nan,
                float(np.sum(self._prev_weights)),
            )
            raise ValueError(f"NaN in observation at timestep {t} (date={self.dates[t]})")

        # Clip extreme values to prevent numerical instability in the network
        # Features should be normalized around [-3, 3] from StandardScaler; clip at [-5, 5] to prevent gradient explosion
        obs = np.clip(obs, -5.0, 5.0, dtype=np.float32)
        return obs

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
