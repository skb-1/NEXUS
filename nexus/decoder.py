from dataclasses import asdict
import numpy as np
from nexus.context import RuntimeContext
from nexus.turbo.profile import Profile


def sample(logits: np.ndarray, history: list[int], rng: np.random.Generator,
           temperature: float, top_k: int, top_p: float, repetition_penalty: float) -> int:
    scores = logits.astype(np.float64).copy()
    scores[[256, 257]] = -np.inf
    for token in set(history):
        if token < 256:
            scores[token] = scores[token] / repetition_penalty if scores[token] > 0 else scores[token] * repetition_penalty
    if temperature == 0:
        return int(np.argmax(scores))
    scores /= temperature
    order = np.argsort(-scores, kind="stable")
    if top_k > 0:
        order = order[:min(top_k, len(order))]
    probabilities = np.exp(scores[order] - scores[order].max())
    probabilities /= probabilities.sum()
    # Include the token that crosses the nucleus threshold.
    keep = np.cumsum(probabilities) - probabilities < top_p
    probabilities = probabilities[keep]
    probabilities /= probabilities.sum()
    return int(rng.choice(order[keep], p=probabilities))


def generate(context: RuntimeContext, prompt: str, *, record_metrics: bool = True, **overrides) -> dict:
    model = context.model
    if model is None:
        raise ValueError("Load or create a model first")
    settings = asdict(context.config.runtime)
    if set(overrides) - set(settings):
        raise ValueError("Unknown generation options")
    settings.update(overrides)
    # Use the same validation rules as config, without mutating the context.
    from nexus.config import Config
    raw = context.config.to_dict()
    raw["runtime"] = settings
    Config.from_dict(raw)
    ids = model.tokenizer.encode(prompt)
    generated = []
    profiler = Profile()
    forward_active = 0
    with context.turbo.scope():
        for _ in range(settings.pop("max_new_tokens")):
            current = np.array([ids[-model.config.context:]], dtype=np.int64)
            result = model.forward(current)
            forward_active = max(forward_active, result.active_forward)
            token = sample(result.logits.data[0, -1], ids[-model.config.context:], context.rng, **settings)
            generated.append(token)
            ids.append(token)
            if token == model.tokenizer.EOS:
                break
    metrics = profiler.finish(len(generated))
    metrics.update({"generated_tokens": len(generated), "total_parameters": model.total_parameters,
                    "active_per_token": model.active_parameters(model.config.active_experts),
                    "active_forward_max": forward_active,
                    "weight_bytes": sum(p.data.nbytes for p in model.parameters.values()),
                    "turbo_enabled": context.turbo.enabled})
    if record_metrics:
        context.record("PERFORMANCE", **metrics)
    return {"text": model.tokenizer.decode(generated), "token_ids": generated, "metrics": metrics}
