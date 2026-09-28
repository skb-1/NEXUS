from dataclasses import dataclass
import numpy as np
from nexus.tensor import Tensor


@dataclass
class Routing:
    indices: np.ndarray
    gates: Tensor
    balance: Tensor
    counts: np.ndarray


def route(x: Tensor, weight: Tensor, k: int, valid: np.ndarray) -> Routing:
    scores = x @ weight
    # Stable ties are reproducible; selection itself is deliberately nondifferentiable.
    selected = np.argsort(-scores.data, axis=-1, kind="stable")[:, :k]
    probabilities = scores.softmax()
    selected_probs = probabilities[np.arange(len(selected))[:, None], selected]
    gates = selected_probs / selected_probs.sum(axis=-1, keepdims=True)
    counts = np.bincount(selected[valid].ravel(), minlength=weight.data.shape[1])
    fractions = counts / max(1, counts.sum())
    mean_prob = (probabilities * valid[:, None]).sum(axis=0) / max(1, valid.sum())
    balance = (mean_prob * fractions).sum() * weight.data.shape[1]
    return Routing(selected, gates, balance, counts)
