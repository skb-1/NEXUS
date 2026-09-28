"""AdamW with global clipping and serializable state, no autograd framework."""
import numpy as np
from nexus.tensor import Tensor


class AdamW:
    def __init__(self, parameters: dict[str, Tensor], weight_decay: float, clip_norm: float):
        self.parameters = parameters
        self.weight_decay, self.clip_norm = weight_decay, clip_norm
        self.step_number = 0
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

    def zero_grad(self) -> None:
        for p in self.parameters.values():
            p.grad = None

    def step(self, lr: float) -> float:
        selected = {n: p for n, p in self.parameters.items() if p.requires_grad and p.grad is not None}
        norm = float(np.sqrt(sum(np.sum(p.grad.astype(np.float64) ** 2) for p in selected.values())))
        if not np.isfinite(norm):
            raise FloatingPointError("Non-finite gradient; checkpoint not updated")
        scale = min(1.0, self.clip_norm / (norm + 1e-12))
        self.step_number += 1
        for name, p in selected.items():
            g = p.grad * scale
            m = self.m.setdefault(name, np.zeros_like(p.data))
            v = self.v.setdefault(name, np.zeros_like(p.data))
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            corrected_m = m / (1 - 0.9 ** self.step_number)
            corrected_v = v / (1 - 0.999 ** self.step_number)
            p.data *= 1 - lr * self.weight_decay
            p.data -= lr * corrected_m / (np.sqrt(corrected_v) + 1e-8)
        return norm

    def tensors(self) -> dict[str, np.ndarray]:
        return {**{f"m/{n}": v for n, v in self.m.items()}, **{f"v/{n}": v for n, v in self.v.items()}}

    def restore(self, values: dict[str, np.ndarray], step: int) -> None:
        self.step_number = step
        for key, array in values.items():
            kind, name = key.split("/", 1)
            if kind not in {"m", "v"} or name not in self.parameters or array.shape != self.parameters[name].data.shape:
                raise ValueError("Invalid optimizer tensor")
            (self.m if kind == "m" else self.v)[name] = array.copy()
        if self.m.keys() != self.v.keys():
            raise ValueError("Incomplete optimizer state")
