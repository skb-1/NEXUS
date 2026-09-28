import numpy as np
import pytest
from nexus.tensor import Tensor, cross_entropy, rms_norm
from nexus.attention import summaries


def numerical_gradient(data, function, epsilon=0.001):
    gradient = np.zeros_like(data)
    for index in np.ndindex(data.shape):
        previous = data[index].copy()
        data[index] = previous + epsilon
        plus = float(function())
        data[index] = previous - epsilon
        minus = float(function())
        data[index] = previous
        gradient[index] = (plus - minus) / (2 * epsilon)
    return gradient


@pytest.mark.parametrize("operation", [
    lambda x: ((x * x + x + 2) ** -0.5).sum(),
    lambda x: x.tanh().mean(),
    lambda x: (x.softmax() * np.arange(3)).sum(),
    lambda x: rms_norm(x).sum(),
    lambda x: x.cumsum(1).sum(),
    lambda x: x[[0, 0, 1]].sum(),
    lambda x: x.reshape(3, 2).sum(axis=0).mean(),
    lambda x: x.scatter_rows(np.array([1, 1]), 3).sum(),
])
def test_gradients(context, operation):
    x = Tensor(np.array([[0.2, -0.3, 0.7], [1.1, 0.2, -0.5]]), context.turbo, True)
    operation(x).backward()
    expected = numerical_gradient(x.data, lambda: operation(x).data)
    np.testing.assert_allclose(x.grad, expected, rtol=0.01, atol=8e-4)


def test_matmul_broadcast_and_accumulation(context):
    x = Tensor([[0.1, 0.2], [0.3, -0.1]], context.turbo, True)
    w = Tensor([[0.2, 0.1, 0.3], [0.4, 0.5, 0.6]], context.turbo, True)
    bias = Tensor([0.1, 0.2, -0.3], context.turbo, True)
    def expression():
        y = (x @ w + bias).tanh()
        return (y * y + y).sum()
    loss = expression()
    loss.backward()
    for p in (x, w, bias):
        expected = numerical_gradient(p.data, lambda: expression().data)
        np.testing.assert_allclose(p.grad, expected, atol=5e-4, rtol=0.005)
    grad = x.grad.copy()
    expression().backward()
    np.testing.assert_allclose(x.grad, 2 * grad, atol=1e-6)


def test_cross_entropy_mask_and_gradient(context):
    x = Tensor([[1, 2, 3], [2, 0, -1]], context.turbo, True)
    targets = np.array([2, 0])
    mask = np.array([1, 0])
    cross_entropy(x, targets, mask).backward()
    expected = numerical_gradient(x.data, lambda: cross_entropy(x, targets, mask).data)
    np.testing.assert_allclose(x.grad, expected, atol=2e-4)
    np.testing.assert_array_equal(x.grad[1], 0)
    with pytest.raises(ValueError):
        cross_entropy(x, targets, np.zeros(2))


def test_summary_values_and_gradient(context):
    x = Tensor(np.arange(1, 13).reshape(1, 4, 3) / 10, context.turbo, True)
    local, global_ = summaries(x, 2)
    for t in range(4):
        np.testing.assert_allclose(local.data[:, t], x.data[:, max(0, t - 1):t + 1].mean(1), atol=1e-6)
        np.testing.assert_allclose(global_.data[:, t], x.data[:, :t + 1].mean(1), atol=1e-6)
    (local.sum() + global_.sum()).backward()
    def value():
        a, b = summaries(x, 2)
        return (a.sum() + b.sum()).data
    expected = numerical_gradient(x.data, value)
    np.testing.assert_allclose(x.grad, expected, atol=0.002)
