"""Versioned checksummed safetensors directories. No pickle or executable config."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from safetensors.numpy import load_file, save_file

from nexus.config import Config, ModelConfig
from nexus.context import RuntimeContext
from nexus.core import Model
from nexus.training.datasets import file_digest

FORMAT_VERSION = 1


def _json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_model(context: RuntimeContext, destination: str | Path, training_state: dict | None = None,
               optimizer_tensors: dict[str, np.ndarray] | None = None) -> Path:
    if context.model is None:
        raise ValueError("No model to save")
    destination = Path(destination).resolve()
    if destination.suffix != ".nexus":
        raise ValueError("Model directory must end in .nexus")
    # Immutable checkpoint destinations avoid partial overwrite and accidental loss.
    if destination.exists():
        raise FileExistsError(f"Destination exists: {destination}; choose a new output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".nexus-save-", dir=destination.parent))
    try:
        groups: dict[str, dict[str, np.ndarray]] = {}
        for name, p in context.model.parameters.items():
            if not np.isfinite(p.data).all():
                raise ValueError(f"Non-finite parameter: {name}")
            group = (f"experts/{name.split('.')[1]}.safetensors" if name.startswith("experts.") else
                     "router.safetensors" if name.startswith("router.") else "core.safetensors")
            groups.setdefault(group, {})[name] = np.ascontiguousarray(p.data)
        groups["tokenizer.safetensors"] = context.model.tokenizer.state()
        if optimizer_tensors:
            groups["optimizer.safetensors"] = optimizer_tensors
        for filename, values in groups.items():
            path = temp / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file(values, str(path))
        metadata = {"config": context.config.to_dict(), "state": context.state,
                    "metrics": context.metrics, "rng": context.rng.bit_generator.state,
                    "trainable": [n for n, p in context.model.parameters.items() if p.requires_grad],
                    "training": training_state}
        _json(temp / "metadata.json", metadata)
        files = {p.relative_to(temp).as_posix(): file_digest(p) for p in temp.rglob("*") if p.is_file()}
        _json(temp / "manifest.json", {"format": "nexus-x", "version": FORMAT_VERSION,
                                      "architecture": "causal-sparse-summary-v1",
                                      "model": context.config.to_dict()["model"], "files": files})
        temp.rename(destination)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return destination


def load_model(context: RuntimeContext, source: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    source = Path(source).resolve()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "nexus-x" or manifest.get("version") != FORMAT_VERSION or manifest.get("architecture") != "causal-sparse-summary-v1":
        raise ValueError("Unsupported model format/architecture")
    files = manifest["files"]
    required = {"metadata.json", "tokenizer.safetensors", "core.safetensors", "router.safetensors"}
    if not required <= files.keys():
        raise ValueError("Incomplete manifest")
    for relative, expected in files.items():
        path = source / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.resolve().is_relative_to(source):
            raise ValueError("Unsafe manifest path")
        if file_digest(path) != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    stored_config = Config.from_dict(metadata["config"])
    if manifest["model"] != stored_config.to_dict()["model"]:
        raise ValueError("Manifest/config architecture mismatch")
    old_model_config = context.config.model
    old_rng = context.rng.bit_generator.state
    try:
        context.config.model = ModelConfig(**manifest["model"])
        model = Model(context)
        tensors = {}
        for relative in files:
            if relative.endswith(".safetensors") and relative not in {"tokenizer.safetensors", "optimizer.safetensors"}:
                loaded = load_file(str(source / relative))
                if tensors.keys() & loaded.keys():
                    raise ValueError("Duplicate parameter names")
                tensors.update(loaded)
        if tensors.keys() != model.parameters.keys():
            raise ValueError("Parameter keys differ from architecture")
        for name, p in model.parameters.items():
            array = tensors[name]
            if array.dtype != np.float32 or array.shape != p.data.shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid parameter: {name}")
            p.data = array.copy()
            p.requires_grad = name in metadata["trainable"]
        tokenizer = load_file(str(source / "tokenizer.safetensors"))
        expected_tokenizer = model.tokenizer.state()
        if tokenizer.keys() != expected_tokenizer.keys() or any(not np.array_equal(tokenizer[k], v) for k, v in expected_tokenizer.items()):
            raise ValueError("Unsupported tokenizer")
        optimizer = load_file(str(source / "optimizer.safetensors")) if "optimizer.safetensors" in files else {}
    except BaseException:
        context.config.model = old_model_config
        context.rng.bit_generator.state = old_rng
        raise
    context.model = model
    context.state = metadata["state"]
    context.metrics = metadata["metrics"]
    context.rng.bit_generator.state = metadata["rng"]
    return metadata, optimizer
