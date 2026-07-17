"""
config.py — reproducible, JSON-driven experiment configuration for the EIIE
agent (docs/EIIE_AGENT_PLAN.md). Every knob a training/backtest run depends
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
    seed: int = 42
    device: str = "cuda"  # GPU enabled; falls back to CPU if unavailable


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
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    experiment: ExperimentMeta = field(default_factory=ExperimentMeta)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        data = dict(d.get("data", {}))
        if "features" in data:
            data["features"] = tuple(data["features"])
        ev = dict(d.get("eval", {}))
        if "baselines" in ev:
            ev["baselines"] = tuple(ev["baselines"])
        return cls(
            data=DataConfig(**data),
            costs=CostConfig(**d.get("costs", {})),
            model=ModelConfig(**d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
            eval=EvalConfig(**ev),
            experiment=ExperimentMeta(**d.get("experiment", {})),
        )

    @classmethod
    def from_json(cls, path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
