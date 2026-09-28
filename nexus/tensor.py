"""Small fp32 reverse-mode engine. Graphs are local, no global grad/backend state."""
from __future__ import annotations
from collections.abc import Callable
import numpy as np

from nexus.turbo import Turbo


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    def __init__(self, data, backend: Turbo, requires_grad: bool = False,
                 parents: tuple[Tensor, ...] = (), backward: Callable | None = None):
        self.data = np.asarray(data, dtype=np.float32)
        self.backend = backend
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self.parents = parents if requires_grad else ()
        self._backward = backward if requires_grad else None

    def add_grad(self, grad: np.ndarray) -> None:
        if self.requires_grad:
            grad = np.asarray(grad, dtype=np.float32)
            self.grad = grad.copy() if self.grad is None else self.grad + grad

    def _wrap(self, value) -> Tensor:
        return value if isinstance(value, Tensor) else Tensor(value, self.backend)

    def __add__(self, other) -> Tensor:
        other = self._wrap(other)
        def backward(g):
            self.add_grad(unbroadcast(g, self.data.shape))
            other.add_grad(unbroadcast(g, other.data.shape))
        return Tensor(self.data + other.data, self.backend,
                      self.requires_grad or other.requires_grad, (self, other), backward)

    __radd__ = __add__

    def __neg__(self) -> Tensor:
        return self * -1

    def __sub__(self, other) -> Tensor:
        return self + -self._wrap(other)

    def __mul__(self, other) -> Tensor:
        other = self._wrap(other)
        def backward(g):
            self.add_grad(unbroadcast(g * other.data, self.data.shape))
            other.add_grad(unbroadcast(g * self.data, other.data.shape))
        return Tensor(self.data * other.data, self.backend,
                      self.requires_grad or other.requires_grad, (self, other), backward)

    __rmul__ = __mul__

    def __pow__(self, exponent: float) -> Tensor:
        return Tensor(self.data ** exponent, self.backend, self.requires_grad, (self,),
                      lambda g: self.add_grad(g * exponent * self.data ** (exponent - 1)))

    def __truediv__(self, other) -> Tensor:
        return self * self._wrap(other) ** -1

    def __matmul__(self, other: Tensor) -> Tensor:
        def backward(g):
            self.add_grad(self.backend.matmul(g, other.data.T))
            other.add_grad(self.backend.matmul(self.data.T, g))
        return Tensor(self.backend.matmul(self.data, other.data), self.backend,
                      self.requires_grad or other.requires_grad, (self, other), backward)

    def reshape(self, *shape: int) -> Tensor:
        return Tensor(self.data.reshape(shape), self.backend, self.requires_grad, (self,),
                      lambda g: self.add_grad(g.reshape(self.data.shape)))

    def __getitem__(self, index) -> Tensor:
        def backward(g):
            full = np.zeros_like(self.data)
            np.add.at(full, index, g)
            self.add_grad(full)
        return Tensor(self.data[index], self.backend, self.requires_grad, (self,), backward)

    def sum(self, axis=None, keepdims: bool = False) -> Tensor:
        def backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.add_grad(np.broadcast_to(g, self.data.shape))
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.backend,
                      self.requires_grad, (self,), backward)

    def mean(self, axis=None, keepdims: bool = False) -> Tensor:
        count = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis, keepdims) / count

    def tanh(self) -> Tensor:
        out = np.tanh(self.data)
        return Tensor(out, self.backend, self.requires_grad, (self,),
                      lambda g: self.add_grad(g * (1 - out * out)))

    def softmax(self) -> Tensor:
        x = self.data - self.data.max(axis=-1, keepdims=True)
        exp = np.exp(x)
        out = exp / exp.sum(axis=-1, keepdims=True)
        return Tensor(out, self.backend, self.requires_grad, (self,),
                      lambda g: self.add_grad(out * (g - (g * out).sum(axis=-1, keepdims=True))))

    def cumsum(self, axis: int) -> Tensor:
        return Tensor(self.data.cumsum(axis=axis), self.backend, self.requires_grad, (self,),
                      lambda g: self.add_grad(np.flip(np.flip(g, axis).cumsum(axis), axis)))

    def scatter_rows(self, indices: np.ndarray, count: int) -> Tensor:
        out = np.zeros((count, self.data.shape[1]), np.float32)
        np.add.at(out, indices, self.data)
        return Tensor(out, self.backend, self.requires_grad, (self,),
                      lambda g: self.add_grad(g[indices]))

    def backward(self) -> None:
        if self.data.size != 1 or not self.requires_grad:
            raise ValueError("backward requires a differentiable scalar")
        # Iterative postorder avoids recursion limits on deep graphs.
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif id(node) not in seen:
                seen.add(id(node))
                stack.append((node, True))
                stack.extend((p, False) for p in node.parents)
        # Leaves accumulate across microbatches, intermediates do not.
        for node in order:
            if node.parents:
                node.grad = None
        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)


def rms_norm(x: Tensor) -> Tensor:
    return x * ((x * x).mean(axis=-1, keepdims=True) + 1e-5) ** -0.5


def cross_entropy(logits: Tensor, targets: np.ndarray, mask: np.ndarray) -> Tensor:
    x = logits.data.reshape(-1, logits.data.shape[-1])
    targets, mask = targets.reshape(-1), mask.reshape(-1).astype(np.float32)
    denominator = float(mask.sum())
    if denominator <= 0:
        raise ValueError("No supervised tokens")
    shifted = x - x.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    normalizer = exps.sum(axis=1, keepdims=True)
    losses = np.log(normalizer[:, 0]) - shifted[np.arange(len(x)), targets]
    value = (losses * mask).sum() / denominator
    def backward(g):
        delta = exps / normalizer
        delta[np.arange(len(x)), targets] -= 1
        logits.add_grad((delta * (mask / denominator)[:, None] * g).reshape(logits.data.shape))
    return Tensor(value, logits.backend, logits.requires_grad, (logits,), backward)
