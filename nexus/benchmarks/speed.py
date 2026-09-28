"""Paired full-forward benchmark; reports regressions instead of hiding them."""
import statistics
import time
import numpy as np
from nexus.context import RuntimeContext


def compare_turbo(context: RuntimeContext, repeats: int = 7) -> dict:
    if repeats < 2:
        raise ValueError("At least two repetitions required")
    model = context.model
    if model is None:
        raise ValueError("Model required")
    ids = np.random.default_rng(100).integers(0, 256, (1, model.config.context))
    was_enabled = context.turbo.enabled
    timings = {True: [], False: []}
    cpu = {True: [], False: []}
    outputs = {}
    try:
        with context.turbo.scope():
            for enabled in (False, True):
                context.turbo.enabled = enabled
                for _ in range(2):
                    outputs[enabled] = model.forward(ids).logits.data.copy()
            for i in range(repeats):
                # Alternate order to reduce thermal/order bias.
                for enabled in ((False, True) if i % 2 == 0 else (True, False)):
                    context.turbo.enabled = enabled
                    cpu_start, start = time.process_time(), time.perf_counter()
                    model.forward(ids)
                    timings[enabled].append(time.perf_counter() - start)
                    cpu[enabled].append(time.process_time() - cpu_start)
    finally:
        context.turbo.enabled = was_enabled
    ref, opt = statistics.median(timings[False]), statistics.median(timings[True])
    equivalent = bool(np.allclose(outputs[False], outputs[True], rtol=2e-4, atol=2e-5))
    if not equivalent:
        raise ArithmeticError("Turbo/reference logits failed numerical equivalence")
    return {"reference_forward_ms": 1000 * ref, "turbo_forward_ms": 1000 * opt,
            "turbo_speedup": ref / opt, "turbo_wall_reduction_percent": 100 * (1 - opt / ref),
            "reference_cpu_seconds_median": statistics.median(cpu[False]),
            "turbo_cpu_seconds_median": statistics.median(cpu[True]),
            "max_abs_logit_error": float(np.max(np.abs(outputs[False] - outputs[True]))),
            "numerically_equivalent": equivalent, "repeats": repeats,
            "batch": 1, "sequence_length": model.config.context,
            "control": "einsum optimize=False vs native BLAS, identical full forward",
            "reference_ms_samples": [v * 1000 for v in timings[False]],
            "turbo_ms_samples": [v * 1000 for v in timings[True]]}
