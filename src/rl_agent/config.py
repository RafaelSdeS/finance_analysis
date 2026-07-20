"""
config.py — reproducible, JSON-driven experiment configuration for the EIIE
agent (docs/eiie_agent/EIIE_AGENT_PLAN.md). Every knob a training/backtest run depends
on lives here; nothing downstream should hardcode a hyperparameter, a date,
or a cost rate.

Usage:
    cfg = ExperimentConfig.from_json("configs/eiie_baseline.json")
    cfg.save(out_dir / "config.json")   # copied into every experiment dir
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    window: int = 50  # trading days of lookback per price tensor (paper's n=50)
    features: tuple = ("close", "high", "low")  # per-slot channels; extend here for future iterations
    window_start: str = "2011-01-31"  # pre-built top-50 universe span (fundamentals-complete)
    window_end: str = "2026-07-14"
    cash_mode: str = "cdi"  # "cdi" (approved deviation) or "zero" (paper-faithful)


@dataclass(frozen=True)
class CostConfig:
    c_sell: float = 0.0003  # B3, 0.03%
    c_buy: float = 0.0003
    train_mu_iters: int = 1  # fixed-k differentiable mu during training (paper leaves k unspecified)
    backtest_mu_tol: float = 1e-10  # converged mu during backtest/eval


@dataclass(frozen=True)
class ModelConfig:
    encoder: str = "eiie_cnn"
    conv1_out_channels: int = 2
    conv2_out_channels: int = 20
    n_assets: int = 50  # fixed network slot width; universe rotates identities, not count


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 3e-5
    l2: float = 1e-8
    batch_size: int = 50  # n_b, paper Table B.1
    pretrain_steps: int = 100_000  # retuned on val; paper's 2e6 assumed ~30k periods, ours has ~2,700
    beta: float = 5e-4  # geometric sample-bias; retuned for daily (paper: 5e-5 over ~30k half-hour periods)
    rolling_steps: int = 30  # OSBL online updates per period during backtest (paper Table B.1)
    grad_clip_norm: float = 5.0
    # entropy bonus on the policy output. NOT in the paper: their cash asset returns 0%, so
    # cash can never dominate and the softmax never collapses onto it. Ours accrues CDI
    # (~8.65%/yr in log-space vs equal-weight's 8.22%), so the gradient pushes every asset
    # score down with no restoring force -- measured: scores ran to -20/-32 vs cash_bias
    # +0.06, softmax saturated to ~1e-9, gradient vanished, agent frozen all-cash and
    # unrecoverable. This term is that restoring force. Set both ends to 0.0 to reproduce
    # that failure.
    #
    # PREVENTIVE, NOT CURATIVE: measured at an already-collapsed checkpoint, this only lifts
    # the gradient norm 4.6e-11 -> 3.9e-9 (still vanishing). It has to be on from step 0; it
    # cannot rescue a saturated net. Old checkpoints are dead, not fixable.
    #
    # SCALE: the reward term is ~5e-4/day (mean log return) and entropy is ~ln(51)=3.9, so
    # the bonus is beta*3.9. At 1e-3 that is 8x the reward and the optimizer just maximizes
    # entropy -> uniform portfolio -> that IS UCRP (8.22%/yr), worse than cash. 1e-5 keeps it
    # at ~8% of the reward: enough to hold the softmax in its responsive range, not enough to
    # dictate the allocation.
    #
    # ANNEALED (not flat): a fixed value never reliably escapes the cash attractor -- an
    # entropy sweep at 1e-6/1e-5/1e-4 showed the same seed (3) escaping at every value while
    # two other seeds stayed 86-100% cash at every value. Seed dominated over beta, meaning a
    # fixed beta only helps when init got lucky. entropy_beta_start pushes exploration harder
    # for the first entropy_anneal_frac of pretrain (forcing the escape instead of hoping for
    # it), then decays linearly to entropy_beta_end -- the settled, scale-matched value above
    # -- for the remainder of pretrain AND the whole online/live phase (no annealing once
    # real trading starts). Set start == end to reproduce the old flat-beta behavior.
    entropy_beta_start: float = 1e-4
    entropy_beta_end: float = 1e-5
    entropy_anneal_frac: float = 0.1
    # Held-out checkpoint selection during pretrain: a fixed pretrain_steps budget isn't
    # reliably right -- the same seed that found a real edge at 100k steps overfit it away by
    # 2M (measured: seed 3 went from +47% to -71% return, same config, only budget changed).
    # Every checkpoint_eval_every steps, score the current policy on a frozen-weights
    # backtest over the last checkpoint_holdout_days of the TRAIN split (never val/test) and
    # keep the best-scoring one instead of trusting whatever step the loop happens to end on.
    checkpoint_holdout_days: int = 250  # ~1 trading year carved out of train's tail
    checkpoint_eval_every: int = 5000
    seed: int = 42
    device: str = "cuda"  # GPU enabled; falls back to CPU if unavailable
    compile: bool = False  # S3 (TRAINING_SPEEDUP_PLAN.md): torch.compile the training
    # forward/backward (mode="reduce-overhead", CUDA graphs). Wall-clock only in intent,
    # but compiled kernels may not be bit-identical to eager -- keep off for runs that
    # must reproduce an eager run exactly.


@dataclass(frozen=True)
class EvalConfig:
    baselines: tuple = (
        "ubah", "ucrp", "best_stock", "random_portfolio",
        "random_rebalancing", "constant_cash", "bova11",
    )
    var_level: float = 0.95
    bootstrap_n: int = 1000
    bootstrap_block: int = 20  # block-bootstrap block length (days)


@dataclass(frozen=True)
class ExperimentMeta:
    name: str = "eiie_baseline"
    out_dir: str = "experiments"


@dataclass(frozen=True)
class RiskConfig:
    """Risk/diversification mandate (RISK_MANDATE_PLAN.md Option A) -- no
    mu estimate anywhere, purely structural allocation from covariance."""
    policies: tuple = ("min_variance", "risk_parity",
                        "min_variance_voltarget", "risk_parity_voltarget")
    lookback: int = 126             # trading days; R2 grid {63, 126, 252}
    min_history_frac: float = 0.8   # eligibility: real (non-backfilled) coverage within lookback
    rebalance_every: int = 21       # trading days; R2 grid {1, 5, 21}
    cov_estimator: str = "ledoit_wolf"  # or "ewma"
    ewma_halflife: int = 63         # trading days
    vol_target_ann: float = 0.12    # ex-ante annualized sigma for *_voltarget policies
    max_weight: float = 1.0         # per-name cap; 1.0 = off
    solver_tol: float = 1e-9
    warm_start: bool = True         # reuse previous rebalance's solution as QP x0


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    experiment: ExperimentMeta = field(default_factory=ExperimentMeta)
    risk: RiskConfig = field(default_factory=RiskConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        data = dict(d.get("data", {}))
        if "features" in data:
            data["features"] = tuple(data["features"])
        ev = dict(d.get("eval", {}))
        if "baselines" in ev:
            ev["baselines"] = tuple(ev["baselines"])
        rk = dict(d.get("risk", {}))
        if "policies" in rk:
            rk["policies"] = tuple(rk["policies"])
        return cls(
            data=DataConfig(**data),
            costs=CostConfig(**d.get("costs", {})),
            model=ModelConfig(**d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
            eval=EvalConfig(**ev),
            experiment=ExperimentMeta(**d.get("experiment", {})),
            risk=RiskConfig(**rk),
        )

    @classmethod
    def from_json(cls, path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
