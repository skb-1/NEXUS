from dataclasses import asdict
import json
import platform
from pathlib import Path
import numpy as np

from nexus.context import RuntimeContext
from nexus.benchmarks.speed import compare_turbo
from nexus.benchmarks.baseline import run_baseline
from nexus.decoder import generate
from nexus.training.engine import evaluate


def benchmark(context: RuntimeContext, model_path: str, output: str, baseline: bool = False) -> dict:
    model = context.model
    if model is None:
        raise ValueError("Model required")
    if baseline and not context.config.training.datasets:
        raise ValueError("Configure training data for the baseline")
    original_rng = context.rng
    try:
        context.rng = np.random.default_rng(context.config.seed + 200)
        for _ in range(2):
            generate(context, "NEXUS", max_new_tokens=32, temperature=0, record_metrics=False)
        generation_samples = [generate(context, "NEXUS", max_new_tokens=32, temperature=0,
                                       record_metrics=False)["metrics"] for _ in range(7)]
    finally:
        context.rng = original_rng
    # Report the actual median-wall run (not ratios of unrelated medians).
    generation_metrics = sorted(generation_samples, key=lambda m: m["wall_seconds"])[3].copy()
    generation_metrics.update({"warmup_runs": 2, "measured_runs": 7,
                               "wall_seconds_samples": [m["wall_seconds"] for m in generation_samples]})
    context.record("PERFORMANCE", **generation_metrics)
    report = {"stage": 1, "model": asdict(model.config),
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "platform": platform.platform(), "machine": platform.machine()},
              "turbo_backend": context.turbo.status(), "total_parameters": model.total_parameters,
              "active_per_token_upper_bound": model.active_parameters(model.config.active_experts),
              "weight_bytes": sum(p.data.nbytes for p in model.parameters.values()),
              "checkpoint_directory_bytes": sum(p.stat().st_size for p in Path(model_path).rglob("*") if p.is_file()),
              "generation": generation_metrics,
              "turbo_comparison": compare_turbo(context) if context.config.turbo.measure_effect else {"measured": False},
              "quality": evaluate(context),
              "baseline": run_baseline(context) if baseline else {"measured": False, "reason": "use --baseline with optional CPU torch"},
              "limitations": ["byte tokens, not word/subword tokens", "no KV/recurrent inference cache: full rolling window recomputed",
                              "RSS snapshot, not peak RAM", "all weights resident", "no energy/IPC/cache-miss measurements",
                              "toy data is not a language/reasoning benchmark", "active parameters are not FLOPs",
                              "one host and seed; no general speed/quality superiority claim"]}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    context.record("BENCHMARK", report=str(path), **report["turbo_comparison"])
    return report
