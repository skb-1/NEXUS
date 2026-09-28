from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from nexus.config import ModelConfig
from nexus.context import RuntimeContext
from nexus.tensor import Tensor, rms_norm
from nexus.encoder import encode
from nexus.attention import summaries
from nexus.router import route
from nexus.experts import dispatch
from nexus.tokenizer import ByteTokenizer


@dataclass
class Forward:
    logits: Tensor
    balance: Tensor
    expert_counts: list[int]
    active_per_token: int
    active_forward: int


class Model:
    def __init__(self, context: RuntimeContext):
        self.context = context
        self.config: ModelConfig = context.config.model
        self.parameters: dict[str, Tensor] = {}
        self.tokenizer = ByteTokenizer()
        c = self.config
        d, r = c.hidden_dim, c.rank
        self._init("encoder.embedding", (c.vocab_size, d), 0.1)
        self._init("position.projection", (6, d), 0.02)
        for layer in range(c.depth):
            self._init(f"core.{layer}.up", (d, r), d ** -0.5)
            self._init(f"core.{layer}.down", (r, d), 0.1 * r ** -0.5)
            self._init(f"core.{layer}.local", (d,), 0, fill=0.3)
            self._init(f"core.{layer}.summary", (d,), 0, fill=0.1)
        self._init("router.weight", (d, c.num_experts), 0.05)
        for e in range(c.num_experts):
            self._init(f"experts.{e}.up", (d, r), d ** -0.5)
            self._init(f"experts.{e}.down", (r, d), 0.1 * r ** -0.5)
        self._init("decoder.weight", (d, c.vocab_size), d ** -0.5)
        self._init("decoder.bias", (c.vocab_size,), 0)

    def _init(self, name: str, shape: tuple[int, ...], scale: float, fill: float = 0) -> None:
        data = (self.context.rng.normal(size=shape) * scale + fill).astype(np.float32)
        self.parameters[name] = Tensor(data, self.context.turbo, True)

    def freeze(self, groups: list[str], only: str | None = None) -> None:
        allowed = {n.split(".")[0] for n in self.parameters}
        if set(groups) - allowed or (only is not None and only not in allowed):
            raise ValueError("Unknown parameter group")
        for name, p in self.parameters.items():
            group = name.split(".")[0]
            p.requires_grad = group not in groups and (only is None or only == group)
            p.grad = None
        if not any(p.requires_grad for p in self.parameters.values()):
            raise ValueError("All parameters are frozen")

    @property
    def total_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters.values())

    def active_parameters(self, experts: int) -> int:
        c = self.config
        per_expert = 2 * c.hidden_dim * c.rank
        return self.total_parameters - (c.num_experts - experts) * per_expert

    def forward(self, ids: np.ndarray, training: bool = False) -> Forward:
        c = self.config
        if ids.ndim != 2 or not 0 < ids.shape[1] <= c.context or ids.shape[0] < 1:
            raise ValueError("Expected nonempty [batch, time] within context")
        if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids >= c.vocab_size):
            raise ValueError("Invalid token ids")
        def p(name: str) -> Tensor:
            value = self.parameters[name]
            return value if training else Tensor(value.data, self.context.turbo)
        h = encode(ids, p("encoder.embedding"), p("position.projection"), c.context)
        batch, length = ids.shape
        for i in range(c.depth):
            z = rms_norm(h)
            local, summary = summaries(z, c.local_window)
            mixed = z + p(f"core.{i}.local") * local + p(f"core.{i}.summary") * summary
            delta = ((mixed.reshape(-1, c.hidden_dim) @ p(f"core.{i}.up")).tanh()
                     @ p(f"core.{i}.down")).reshape(batch, length, c.hidden_dim)
            h = h + delta * c.depth ** -0.5
        flat = h.reshape(-1, c.hidden_dim)
        normalized = rms_norm(flat)
        routing = route(normalized, p("router.weight"), c.active_experts,
                        (ids != ByteTokenizer.PAD).reshape(-1))
        residual, used = dispatch(normalized, routing, p, c.num_experts)
        logits = rms_norm(flat + residual) @ p("decoder.weight") + p("decoder.bias")
        return Forward(logits.reshape(batch, length, c.vocab_size), routing.balance,
                       routing.counts.tolist(), self.active_parameters(c.active_experts),
                       self.active_parameters(len(used)))
