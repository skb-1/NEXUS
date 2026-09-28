import copy
import json
from pathlib import Path
import numpy as np
import pytest

from nexus.context import RuntimeContext
from nexus.format.nexus_format import save_model, load_model
from nexus.orchestrator import Orchestrator


def test_roundtrip_and_tampering(context, tmp_path):
    ids = np.array([[257, 97, 98]])
    before = context.model.forward(ids).logits.data
    destination = tmp_path / "model.nexus"
    save_model(context, destination)
    other = RuntimeContext(copy.deepcopy(context.config))
    load_model(other, destination)
    np.testing.assert_array_equal(before, other.model.forward(ids).logits.data)
    with pytest.raises(FileExistsError):
        save_model(context, destination)
    with (destination / "core.safetensors").open("ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="Checksum"):
        load_model(other, destination)


def test_manifest_traversal_rejected(context, tmp_path):
    destination = tmp_path / "model.nexus"
    save_model(context, destination)
    path = destination / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../outside"] = "x"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe"):
        load_model(context, destination)


def test_exact_training_resume(config, tmp_path):
    config.training.grad_accum = 2
    whole = Orchestrator(copy.deepcopy(config))
    whole.create(str(tmp_path / "whole.nexus"))
    split = Orchestrator(copy.deepcopy(config))
    split.create(str(tmp_path / "part.nexus"), max_steps=3)
    resumed = Orchestrator(copy.deepcopy(config))
    resumed.create(str(tmp_path / "resumed.nexus"), resume=str(tmp_path / "part.nexus"))
    for name, p in whole.context.model.parameters.items():
        np.testing.assert_array_equal(p.data, resumed.context.model.parameters[name].data, err_msg=name)
    a = json.loads((tmp_path / "whole.nexus/metadata.json").read_text())
    b = json.loads((tmp_path / "resumed.nexus/metadata.json").read_text())
    assert a["training"] == b["training"]
    assert a["rng"] == b["rng"]


@pytest.mark.parametrize("mode,freeze,groups", [
    ("full", ["encoder", "position", "core"], {"router", "experts", "decoder"}),
    ("expert", [], {"experts"}), ("router", [], {"router"})])
def test_finetune_freezes(config, tmp_path, mode, freeze, groups):
    base = Orchestrator(copy.deepcopy(config))
    source = tmp_path / "base.nexus"
    base.create(str(source), max_steps=2)
    before = {n: p.data.copy() for n, p in base.context.model.parameters.items()}
    config.finetune.mode = mode
    config.finetune.freeze = freeze
    ft = Orchestrator(config)
    ft.finetune(str(source), str(tmp_path / "ft.nexus"), max_steps=2)
    changed = set()
    for name, p in ft.context.model.parameters.items():
        if not np.array_equal(p.data, before[name]):
            changed.add(name.split(".")[0])
    assert changed == groups
    reload_base = RuntimeContext(copy.deepcopy(config))
    load_model(reload_base, source)
    for name, p in reload_base.model.parameters.items():
        np.testing.assert_array_equal(p.data, before[name])


def test_adamw_first_step_and_clipping(context):
    from nexus.training.optimizer import AdamW
    from nexus.tensor import Tensor
    p = Tensor([1.0, -2.0], context.turbo, True)
    p.grad = np.array([3.0, 4.0], np.float32)
    opt = AdamW({"p": p}, weight_decay=0.1, clip_norm=1.0)
    assert opt.step(0.01) == 5.0
    expected = np.array([1.0, -2.0]) * 0.999 - 0.01
    np.testing.assert_allclose(p.data, expected, atol=1e-6)
    np.testing.assert_allclose(opt.m["p"], [0.06, 0.08], atol=1e-7)


def test_changed_resume_config_rejected(config, tmp_path):
    app = Orchestrator(copy.deepcopy(config))
    source = tmp_path / "base.nexus"
    app.create(str(source), max_steps=2)
    config.training.lr *= 2
    with pytest.raises(ValueError, match="configuration changed"):
        Orchestrator(config).create(str(tmp_path / "bad.nexus"), str(source))
