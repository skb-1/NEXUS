import numpy as np
from nexus.tensor import cross_entropy
from nexus.training.optimizer import AdamW
from nexus.decoder import sample
from nexus.tokenizer import ByteTokenizer


def test_causal_and_prefix_invariant(context):
    ids = np.array([[257, 5, 8, 10, 20, 30]])
    before = context.model.forward(ids).logits.data
    ids[0, 4:] = [42, 70]
    after = context.model.forward(ids).logits.data
    np.testing.assert_allclose(before[:, :4], after[:, :4], atol=1e-6)
    prefix = context.model.forward(ids[:, :4]).logits.data
    np.testing.assert_allclose(prefix, after[:, :4], atol=2e-6)


def test_model_gradient_and_router_learning(context):
    model = context.model
    ids = np.array([[257, 4, 5]])
    labels = np.array([[4, 5, 6]])
    def loss():
        result = model.forward(ids, training=True)
        return cross_entropy(result.logits, labels, np.ones_like(labels)) + result.balance * 0.01
    loss().backward()
    assert np.linalg.norm(model.parameters["router.weight"].grad) > 0
    for name, index in [("encoder.embedding", (257, 2)), ("core.0.up", (1, 1)),
                        ("core.0.summary", (2,)), ("router.weight", (0, 0)),
                        ("experts.0.down", (0, 2)), ("decoder.weight", (0, 6))]:
        p = model.parameters[name]
        analytic = 0 if p.grad is None else p.grad[index]
        saved = p.data[index].copy()
        epsilon = 0.001
        p.data[index] = saved + epsilon
        plus = float(loss().data)
        p.data[index] = saved - epsilon
        minus = float(loss().data)
        p.data[index] = saved
        np.testing.assert_allclose(analytic, (plus - minus) / (2 * epsilon), atol=0.002, rtol=0.1, err_msg=name)


def test_sparse_experts_really_not_called(context):
    model = context.model
    model.parameters["router.weight"].data[:] = 0  # stable ties select experts 0, 1
    result = model.forward(np.array([[257, 10, 11]]), training=True)
    result.logits.sum().backward()
    assert result.active_forward == model.active_parameters(2)
    assert result.expert_counts == [3, 3, 0, 0]
    for e in (0, 1):
        assert model.parameters[f"experts.{e}.up"].grad is not None
    for e in (2, 3):
        assert model.parameters[f"experts.{e}.up"].grad is None
    assert result.active_per_token < model.total_parameters


def test_training_reduces_fixed_batch_loss(context):
    model = context.model
    optimizer = AdamW(model.parameters, 0, 1)
    ids = np.array([[257, 97, 98, 97]])
    targets = np.array([[97, 98, 97, 258]])
    losses = []
    with context.turbo.scope():
        for _ in range(40):
            optimizer.zero_grad()
            result = model.forward(ids, training=True)
            loss = cross_entropy(result.logits, targets, np.ones_like(targets))
            losses.append(float(loss.data))
            loss.backward()
            optimizer.step(0.02)
    assert losses[-1] < losses[0] * 0.4


def test_tokenizer_utf8_and_sampling():
    tokenizer = ByteTokenizer()
    text = "Привет, NEXUS! 🧪"
    assert tokenizer.decode(tokenizer.encode(text, eos=True)) == text
    logits = np.zeros(259)
    logits[100] = 10
    logits[256] = 100
    assert sample(logits, [], np.random.default_rng(0), 0, 1, 0.9, 1) == 100
    assert sample(logits, [], np.random.default_rng(0), 1, 1, 0.9, 1) == 100


def test_top_one_gate_has_no_task_gradient(context):
    from nexus.router import route
    from nexus.tensor import Tensor
    x = Tensor([[0.2, 0.4, -0.1]], context.turbo, True)
    w = Tensor([[0.1, 0.2], [0.3, -0.1], [0.2, 0.5]], context.turbo, True)
    routing = route(x, w, 1, np.array([True]))
    routing.gates.sum().backward()
    np.testing.assert_allclose(w.grad, 0, atol=1e-6)
    w.grad = None
    route(x, w, 1, np.array([True])).balance.backward()
    assert np.linalg.norm(w.grad) > 1e-4


def test_inference_does_not_retain_graphs_or_mutate_weights(context):
    import gc
    import psutil
    ids = np.array([[257, 10, 20]])
    model = context.model
    before = {n: p.data.copy() for n, p in model.parameters.items()}
    for _ in range(10):
        model.forward(ids)
    gc.collect()
    rss = psutil.Process().memory_info().rss
    for _ in range(200):
        model.forward(ids)
    gc.collect()
    assert psutil.Process().memory_info().rss - rss < 16 * 2**20
    for name, p in model.parameters.items():
        assert p.grad is None and p.parents == ()
        np.testing.assert_array_equal(p.data, before[name])
