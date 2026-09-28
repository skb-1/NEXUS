"""Strict, shared configuration; paths are relative to the YAML file."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import math
from typing import Any, TypeVar

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 259
    hidden_dim: int = 32
    depth: int = 2
    num_experts: int = 4
    active_experts: int = 2
    rank: int = 8
    context: int = 64
    local_window: int = 8
    precision: str = "fp32"


@dataclass
class TurboConfig:
    enabled: bool = True
    threads: int | str = "auto"
    deterministic: bool = True
    measure_effect: bool = True


@dataclass
class DatasetConfig:
    path: str = ""
    weight: float = 1.0
    modality: str = "text"
    format: str = "auto"
    limit: int | None = None


@dataclass
class TrainingConfig:
    datasets: list[DatasetConfig] = field(default_factory=list)
    validation: list[DatasetConfig] = field(default_factory=list)
    batch_size: int = 4
    grad_accum: int = 1
    lr: float = 0.003
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    clip_norm: float = 1.0
    balance_loss: float = 0.01
    epochs: int = 2
    steps_per_epoch: int = 30
    shuffle_buffer: int = 16
    checkpoint_every: int = 30
    log_every: int = 10
    eval_batches: int = 4
    schedule: str = "cosine"
    optimizer: str = "adamw"


@dataclass
class FinetuneConfig:
    base_model: str = "model.nexus"
    mode: str = "full"
    freeze: list[str] = field(default_factory=list)
    datasets: list[DatasetConfig] = field(default_factory=list)


@dataclass
class RuntimeConfig:
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.1


T = TypeVar("T")


def strict(cls: type[T], values: dict[str, Any]) -> T:
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


@dataclass
class Config:
    mode: str = "create"
    seed: int = 42
    output: str = "model.nexus"
    log_dir: str = "runs"
    model: ModelConfig = field(default_factory=ModelConfig)
    turbo: TurboConfig = field(default_factory=TurboConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        raw = dict(raw)
        for name, typ in (("model", ModelConfig), ("turbo", TurboConfig),
                          ("training", TrainingConfig), ("finetune", FinetuneConfig),
                          ("runtime", RuntimeConfig)):
            values = dict(raw.get(name, {}))
            for key in ("datasets", "validation"):
                if key in values:
                    values[key] = [strict(DatasetConfig, d) for d in values[key]]
            raw[name] = strict(typ, values)
        config = strict(cls, raw)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be a YAML mapping")
        config = cls.from_dict(raw)
        for ds in config.training.datasets + config.training.validation + config.finetune.datasets:
            ds.path = str((path.parent / ds.path).resolve())
        for obj, attr in ((config, "output"), (config, "log_dir"), (config.finetune, "base_model")):
            setattr(obj, attr, str((path.parent / getattr(obj, attr)).resolve()))
        return config

    def validate(self) -> None:
        m, t, r = self.model, self.training, self.runtime
        positive = [m.hidden_dim, m.depth, m.num_experts, m.active_experts, m.rank,
                    m.context, m.local_window, t.batch_size, t.grad_accum, t.epochs,
                    t.steps_per_epoch, t.shuffle_buffer, t.checkpoint_every,
                    t.log_every, t.eval_batches, r.max_new_tokens]
        if any(type(v) is not int or v < 1 for v in positive):
            raise ValueError("Dimensions, counts and intervals must be positive integers")
        if m.vocab_size != 259 or m.precision != "fp32":
            raise ValueError("v0.1 requires byte vocab_size=259 and precision=fp32")
        if m.active_experts > m.num_experts:
            raise ValueError("active_experts exceeds num_experts")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.mode not in {"create", "finetune", "use", "bench", "search", "status"}:
            raise ValueError("Invalid mode")
        if t.optimizer != "adamw" or t.schedule != "cosine":
            raise ValueError("v0.1 implements adamw and cosine")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
               for v in (t.lr, t.clip_norm, r.repetition_penalty)):
            raise ValueError("lr, clip_norm, repetition_penalty must be finite and positive")
        if not 0 <= t.min_lr_ratio <= 1 or not 0 <= r.top_p <= 1 or r.top_p == 0:
            raise ValueError("Invalid min_lr_ratio or top_p")
        if any(not math.isfinite(v) or v < 0 for v in (t.balance_loss, t.weight_decay, r.temperature)):
            raise ValueError("Invalid loss, decay or temperature")
        if type(r.top_k) is not int or r.top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if self.turbo.threads != "auto" and (type(self.turbo.threads) is not int or self.turbo.threads < 1):
            raise ValueError("turbo.threads must be auto or positive integer")
        for flag in (self.turbo.enabled, self.turbo.deterministic, self.turbo.measure_effect):
            if type(flag) is not bool:
                raise ValueError("Turbo flags must be booleans")
        if self.finetune.mode not in {"full", "instruction", "expert", "router"}:
            raise ValueError("Unsupported fine-tuning mode")
        if not isinstance(self.finetune.freeze, list) or set(self.finetune.freeze) - {"encoder", "position", "core", "router", "experts", "decoder"}:
            raise ValueError("Unknown freeze component")
        for d in t.datasets + t.validation + self.finetune.datasets:
            if not d.path or not math.isfinite(d.weight) or d.weight <= 0:
                raise ValueError("Dataset path and finite positive weight required")
            if d.modality not in {"text", "code"} or d.format not in {"auto", "txt", "jsonl"}:
                raise ValueError("v0.1 supports text/code TXT/JSONL only")
            if d.limit is not None and (type(d.limit) is not int or d.limit < 1):
                raise ValueError("Dataset limit must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
