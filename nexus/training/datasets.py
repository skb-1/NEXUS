"""Streaming documents, bounded shuffle buffers, weighted reproducible sampling."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from collections.abc import Iterator

import numpy as np

from nexus.config import DatasetConfig
from nexus.tokenizer import ByteTokenizer

Sample = tuple[list[int], list[float]]


def file_digest(path: str | Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def documents(spec: DatasetConfig, instruction: bool) -> Iterator[tuple[str, str | None]]:
    path = Path(spec.path)
    fmt = spec.format if spec.format != "auto" else path.suffix.lower().lstrip(".")
    if fmt not in {"txt", "jsonl"}:
        raise ValueError(f"Cannot infer supported format for {path}; set format: txt/jsonl")
    count = 0
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            if fmt == "txt":
                if instruction:
                    raise ValueError("Instruction mode requires JSONL prompt/response records")
                yield line.rstrip("\r\n"), None
            else:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{line_number}: JSON object required")
                if "prompt" in item and "response" in item:
                    if not isinstance(item["prompt"], str) or not isinstance(item["response"], str):
                        raise ValueError("prompt/response must be strings")
                    yield item["prompt"], item["response"]
                elif not instruction and isinstance(item.get("text"), str):
                    yield item["text"], None
                else:
                    raise ValueError(f"{path}:{line_number}: expected text or prompt/response")
            count += 1
            if spec.limit is not None and count >= spec.limit:
                return


def chunks(spec: DatasetConfig, context: int, instruction: bool) -> Iterator[Sample]:
    tokenizer = ByteTokenizer()
    for text, response in documents(spec, instruction):
        if response is None:
            ids = tokenizer.encode(text, eos=True)
            mask = [1.0] * len(ids)
        else:
            prompt = tokenizer.encode(text + "\n")
            answer = tokenizer.encode(response, bos=False, eos=True)
            ids = prompt + answer
            mask = ([0.0] * len(prompt) if instruction else [1.0] * len(prompt)) + [1.0] * len(answer)
        for start in range(0, len(ids) - 1, context):
            seq, valid = ids[start:start + context + 1], mask[start:start + context + 1]
            if sum(valid[1:]) == 0:
                continue
            pad = context + 1 - len(seq)
            yield seq + [tokenizer.PAD] * pad, valid + [0.0] * pad


def repeating(spec: DatasetConfig, context: int, instruction: bool) -> Iterator[Sample]:
    while True:
        found = False
        for sample in chunks(spec, context, instruction):
            found = True
            yield sample
        if not found:
            raise ValueError(f"Dataset contains no supervised samples: {spec.path}")


class DatasetEngine:
    def __init__(self, specs: list[DatasetConfig], context: int, buffer_size: int,
                 seed: int, instruction: bool = False, state: dict | None = None):
        if not specs or buffer_size < 1:
            raise ValueError("At least one dataset and a positive shuffle buffer are required")
        self.specs, self.context, self.buffer_size = specs, context, buffer_size
        self.instruction = instruction
        self.rng = np.random.default_rng(seed)
        self.identity = {"datasets": [{**asdict(s), "sha256": file_digest(s.path)} for s in specs],
                         "context": context, "buffer_size": buffer_size, "instruction": instruction}
        self.iterators = [repeating(s, context, instruction) for s in specs]
        self.consumed = [0] * len(specs)
        self.buffers: list[list[Sample]] = [[] for _ in specs]
        self.draws = [0] * len(specs)
        weights = np.array([s.weight for s in specs], dtype=np.float64)
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Dataset weights must be finite and positive")
        self.weights = weights / weights.sum()
        if state is None:
            for i in range(len(specs)):
                self.buffers[i] = [self._next(i) for _ in range(buffer_size)]
        else:
            if state["identity"] != self.identity:
                raise ValueError("Sampler config or dataset content changed since checkpoint")
            for i, count in enumerate(state["consumed"]):
                for _ in range(count):
                    self._next(i)
            self.buffers = state["buffers"]
            self.draws = state["draws"].copy()
            self.rng.bit_generator.state = state["rng"]

    def _next(self, index: int) -> Sample:
        value = next(self.iterators[index])
        self.consumed[index] += 1
        return value

    def batch(self, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        samples = []
        for _ in range(size):
            index = int(self.rng.choice(len(self.specs), p=self.weights))
            slot = int(self.rng.integers(self.buffer_size))
            samples.append(self.buffers[index][slot])
            self.buffers[index][slot] = self._next(index)
            self.draws[index] += 1
        ids = np.asarray([s[0] for s in samples], np.int64)
        mask = np.asarray([s[1] for s in samples], np.float32)
        return ids[:, :-1], ids[:, 1:], mask[:, 1:]

    def state_dict(self) -> dict:
        # JSON roundtrip snapshots mutable buffers and RNG, not live references.
        return json.loads(json.dumps({"identity": self.identity, "consumed": self.consumed,
                                      "buffers": self.buffers, "draws": self.draws,
                                      "rng": self.rng.bit_generator.state}))
