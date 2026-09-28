"""Shared application services. CLI and future GUI call these exact methods."""
from pathlib import Path
import psutil

from nexus.config import Config
from nexus.context import RuntimeContext
from nexus.core import Model
from nexus.decoder import generate
from nexus.training.engine import train
from nexus.training.checkpoint import load_model
from nexus.benchmarks.report import benchmark


class Orchestrator:
    def __init__(self, config: Config):
        self.context = RuntimeContext(config)

    def create(self, output: str | None = None, resume: str | None = None,
               max_steps: int | None = None) -> dict:
        ctx = self.context
        ctx.record("MODE", mode="create")
        restored = load_model(ctx, resume) if resume else None
        if resume and ctx.state.get("mode") != "create":
            raise ValueError("create --resume requires a create checkpoint")
        if restored is None:
            ctx.model = Model(ctx)
            ctx.state = {"mode": "create", "step": 0}
        ctx.record("ARCHITECTURE", **ctx.config.to_dict()["model"])
        ctx.record("TURBO", **ctx.turbo.status())
        return train(ctx, ctx.config.training.datasets, output or ctx.config.output,
                     restored, max_steps=max_steps)

    def finetune(self, source: str | None = None, output: str | None = None,
                 resume: bool = False, max_steps: int | None = None) -> dict:
        ctx = self.context
        source = source or ctx.config.finetune.base_model
        target = output or ctx.config.output
        if Path(source).resolve() == Path(target).resolve():
            raise ValueError("Fine-tuning requires a new output; base model is immutable")
        restored = load_model(ctx, source)
        f = ctx.config.finetune
        if resume:
            if ctx.state.get("mode") != "finetune" or restored[0]["config"]["finetune"] != ctx.config.to_dict()["finetune"]:
                raise ValueError("Fine-tuning resume mode/freeze/data configuration changed")
        else:
            ctx.model.freeze(f.freeze, only={"expert": "experts", "router": "router"}.get(f.mode))
            ctx.state = {"mode": "finetune", "step": 0, "base_model": str(source)}
        ctx.record("MODE", mode="finetune", strategy=f.mode, freeze=f.freeze)
        specs = f.datasets or ctx.config.training.datasets
        return train(ctx, specs, target, restored if resume else None,
                     instruction=f.mode == "instruction", max_steps=max_steps)

    def load(self, source: str) -> None:
        load_model(self.context, source)

    def use(self, prompt: str, **options) -> dict:
        if self.context.config.turbo.measure_effect and not self.context.state.get("session_turbo_measured"):
            from nexus.benchmarks.speed import compare_turbo
            self.context.record("TURBO", **compare_turbo(self.context))
            self.context.state["session_turbo_measured"] = True
        result = generate(self.context, prompt, **options)
        if self.context.config.turbo.measure_effect:
            result["metrics"]["turbo_forward_speedup_measured"] = self.context.metrics.get("turbo_speedup")
        return result

    def bench(self, source: str, output: str, baseline: bool = False) -> dict:
        metadata, _ = load_model(self.context, source)
        if baseline:
            current = self.context.config.to_dict()
            if current["training"] != metadata["config"]["training"] or current["seed"] != metadata["config"]["seed"]:
                raise ValueError("Baseline needs the checkpoint's original training config and seed")
        return benchmark(self.context, source, output, baseline)

    def status(self) -> dict:
        ctx = self.context
        model = ctx.model
        process = psutil.Process()
        return {"state": ctx.state, "model_loaded": model is not None,
                "parameters": model.total_parameters if model else None,
                "rss_mb": process.memory_info().rss / 2**20,
                "logical_cpus": psutil.cpu_count(), "process_threads": process.num_threads(),
                "turbo": ctx.turbo.status(), "last_metrics": ctx.metrics}
