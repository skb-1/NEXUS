"""Optional from-scratch torch Transformer control, never the NEXUS model's brain."""
import time
import math
import numpy as np

from nexus.context import RuntimeContext
from nexus.training.datasets import DatasetEngine
from nexus.training.schedule import cosine_lr


def run_baseline(context: RuntimeContext) -> dict:
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as error:
        raise RuntimeError("Transformer comparison requires optional CPU torch; see README") from error
    c, t = context.config.model, context.config.training
    if not t.datasets or not t.validation:
        raise ValueError("Baseline requires training and held-out validation datasets")
    if context.config.finetune.mode != "full" or context.state.get("mode") != "create":
        raise ValueError("Matched baseline is supported only for a create checkpoint (not fine-tuned)")
    if context.state.get("step") != t.epochs * t.steps_per_epoch:
        raise ValueError("Baseline needs a completed create training schedule")
    torch.manual_seed(context.config.seed)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(context.turbo.threads)

    class Baseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(c.vocab_size, c.hidden_dim)
            self.position = nn.Parameter(torch.randn(c.context, c.hidden_dim) * 0.02)
            heads = next(h for h in (4, 2, 1) if c.hidden_dim % h == 0)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(c.hidden_dim, heads, c.hidden_dim * 2,
                                           dropout=0, batch_first=True, norm_first=True)
                for _ in range(c.depth)])
            self.output = nn.Linear(c.hidden_dim, c.vocab_size)

        def forward(self, x):
            h = self.embedding(x) + self.position[:x.shape[1]]
            causal = torch.ones((x.shape[1], x.shape[1]), dtype=torch.bool).triu(1)
            for layer in self.layers:
                h = layer(h, src_mask=causal)
            return self.output(h)

    try:
        model = Baseline()
        optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay,
                                     betas=(0.9, 0.999), eps=1e-8)
        dataset = DatasetEngine(t.datasets, c.context, t.shuffle_buffer, context.config.seed + 1)
        started = time.perf_counter()
        steps = t.epochs * t.steps_per_epoch
        for step in range(steps):
            optimizer.zero_grad()
            for _ in range(t.grad_accum):
                x, y, mask = dataset.batch(t.batch_size)
                logits = model(torch.from_numpy(x.copy()))
                loss = F.cross_entropy(logits.reshape(-1, c.vocab_size), torch.from_numpy(y.copy()).reshape(-1), reduction="none")
                mask_tensor = torch.from_numpy(mask.copy()).reshape(-1)
                ((loss * mask_tensor).sum() / mask_tensor.sum() / t.grad_accum).backward()
            nn.utils.clip_grad_norm_(model.parameters(), t.clip_norm)
            optimizer.param_groups[0]["lr"] = cosine_lr(step, steps, t.lr, t.min_lr_ratio)
            optimizer.step()
        train_seconds = time.perf_counter() - started
        model.eval()
        dataset = DatasetEngine(t.validation, c.context, t.shuffle_buffer, context.config.seed + 101)
        summed, count = 0., 0
        ids = torch.from_numpy(np.random.default_rng(100).integers(0, 256, (1, c.context)))
        with torch.no_grad():
            for _ in range(t.eval_batches):
                x, y, mask = dataset.batch(t.batch_size)
                logits = model(torch.from_numpy(x.copy()))
                loss = F.cross_entropy(logits.reshape(-1, c.vocab_size), torch.from_numpy(y.copy()).reshape(-1), reduction="none").numpy()
                summed += float((loss * mask.reshape(-1)).sum())
                count += int(mask.sum())
            for _ in range(2):
                model(ids)
            samples = []
            for _ in range(7):
                start = time.perf_counter()
                model(ids)
                samples.append(time.perf_counter() - start)
        ce = summed / count
        return {"architecture": "torch causal Transformer, trained from scratch",
                "parameters": sum(p.numel() for p in model.parameters()),
                "train_seconds": train_seconds, "steps": steps, "forward_ms": float(np.median(samples) * 1000),
                "validation_ce_nats": ce, "validation_bits_per_byte_token": ce / math.log(2),
                "comparison": "same data draws, steps, width, depth, context, CPU threads, optimizer schedule; NOT parameter/FLOP-matched",
                "caveat": "different initialization and kernels; one seed, toy corpus, not evidence of architecture superiority"}
    finally:
        torch.set_num_threads(old_threads)
