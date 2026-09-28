from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np

from nexus.context import RuntimeContext
from nexus.tensor import cross_entropy
from nexus.training.datasets import DatasetEngine
from nexus.training.optimizer import AdamW
from nexus.training.schedule import cosine_lr
from nexus.training.checkpoint import save_model
from nexus.turbo.profile import Profile


def evaluate(context: RuntimeContext, specs=None) -> dict:
    t = context.config.training
    specs = t.validation if specs is None else specs
    if not specs:
        return {"validation": "not measured: no held-out datasets configured"}
    data = DatasetEngine(specs, context.config.model.context, t.shuffle_buffer,
                         context.config.seed + 101, context.config.finetune.mode == "instruction")
    total, count = 0.0, 0
    with context.turbo.scope():
        for _ in range(t.eval_batches):
            x, y, mask = data.batch(t.batch_size)
            result = context.model.forward(x)
            loss = float(cross_entropy(result.logits, y, mask).data)
            total += loss * float(mask.sum())
            count += int(mask.sum())
    ce = total / count
    return {"validation_ce_nats": ce, "validation_bits_per_byte_token": ce / np.log(2),
            "validation_perplexity": float(np.exp(min(ce, 80))), "validation_tokens": count,
            "quality_note": "sampled held-out next-token loss; includes EOS, not reasoning quality"}


def train(context: RuntimeContext, specs, output: str, resume: tuple[dict, dict] | None = None,
          instruction: bool = False, max_steps: int | None = None) -> dict:
    if context.model is None:
        raise ValueError("Model required")
    if Path(output).exists():
        raise FileExistsError(f"Output already exists: {output}")
    t = context.config.training
    optimizer = AdamW(context.model.parameters, t.weight_decay, t.clip_norm)
    start = 0
    sampler_state = None
    if resume:
        metadata, tensors = resume
        saved = metadata.get("training")
        if saved is None:
            raise ValueError("Checkpoint has no training state")
        # Exact resume requires the same optimizer schedule and sampling contract.
        if saved["training_config"] != asdict(t) or saved["instruction"] != instruction:
            raise ValueError("Resume training configuration changed; use finetune for a new schedule")
        start = saved["step"]
        sampler_state = saved["sampler"]
        optimizer.restore(tensors, start)
    data = DatasetEngine(specs, context.config.model.context, t.shuffle_buffer,
                         context.config.seed + 1, instruction, sampler_state)
    total_steps = t.epochs * t.steps_per_epoch
    end = min(total_steps, start + max_steps) if max_steps is not None else total_steps
    if end <= start:
        raise ValueError("No training steps remain")
    profile, tokens = Profile(), 0
    context.record("TRAINING", datasets=[{"path": s.path, "weight": s.weight} for s in specs], start=start, target=end)
    checkpoint_parent = Path(output).parent / f"{Path(output).stem}_checkpoints"
    last_loss = 0.0
    with context.turbo.scope():
        for step in range(start, end):
            optimizer.zero_grad()
            last_loss, balance = 0.0, 0.0
            for _ in range(t.grad_accum):
                x, y, mask = data.batch(t.batch_size)
                result = context.model.forward(x, training=True)
                ce = cross_entropy(result.logits, y, mask)
                loss = (ce + t.balance_loss * result.balance) / t.grad_accum
                if not np.isfinite(loss.data):
                    raise FloatingPointError("Non-finite loss")
                loss.backward()
                last_loss += float(ce.data) / t.grad_accum
                balance += float(result.balance.data) / t.grad_accum
                tokens += int(mask.sum())
            lr = cosine_lr(step, total_steps, t.lr, t.min_lr_ratio)
            norm = optimizer.step(lr)
            context.state.update({"step": step + 1, "epoch": step // t.steps_per_epoch + 1})
            if step == start or (step + 1) % t.log_every == 0 or step + 1 == end:
                context.record("TRAINING", step=step + 1, ce_nats=last_loss, balance=balance,
                               lr=lr, grad_norm=norm, expert_counts=result.expert_counts,
                               **profile.finish(tokens))
            if (step + 1) % t.checkpoint_every == 0 and step + 1 < end:
                training_state = {"step": step + 1, "training_config": asdict(t),
                                  "sampler": data.state_dict(), "instruction": instruction}
                save_model(context, checkpoint_parent / f"step-{step + 1:08d}.nexus",
                           training_state, optimizer.tensors())
    quality = evaluate(context)
    context.record("QUALITY", **quality)
    if context.config.turbo.measure_effect:
        from nexus.benchmarks.speed import compare_turbo
        context.record("TURBO", **compare_turbo(context))
    metrics = {"final_train_ce_nats": last_loss, "total_parameters": context.model.total_parameters,
               "active_per_token": context.model.active_parameters(context.config.model.active_experts),
               "sampler_draws": data.draws, **profile.finish(tokens), **quality}
    context.record("TRAINING", **metrics)
    state = {"step": end, "training_config": asdict(t), "sampler": data.state_dict(), "instruction": instruction}
    save_model(context, output, state, optimizer.tensors())
    context.record("MODE", saved=str(output))
    return metrics
