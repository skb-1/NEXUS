import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from nexus.config import Config
from nexus.context import RuntimeContext
from nexus.core import Model
from nexus.turbo.precision import quantize_int8, dequantize_int8
from nexus.benchmarks.speed import compare_turbo


@pytest.mark.parametrize("change", [
    {"mystery": 1}, {"model": {"hidden_dim": 0}}, {"model": {"active_experts": 9}},
    {"turbo": {"threads": 0}}, {"training": {"lr": -1}},
    {"model": {"precision": "int4"}}, {"finetune": {"freeze": ["unknown"]}},
    {"training": {"batch_size": True}}, {"runtime": {"top_p": 0}},
    {"turbo": {"enabled": "true"}}, {"training": {"lr": float("nan")}},
])
def test_invalid_config(change):
    with pytest.raises((ValueError, TypeError)):
        Config.from_dict(change)


def test_config_roundtrip_and_relative_paths(config, tmp_path):
    path = tmp_path / "config.yaml"
    raw = config.to_dict()
    raw["training"]["datasets"][0]["path"] = "train.txt"
    path.write_text(yaml.safe_dump(raw))
    loaded = Config.load(path)
    assert loaded.training.datasets[0].path == str(tmp_path / "train.txt")
    assert Config.from_dict(loaded.to_dict()) == loaded


def test_int8_error_bound():
    matrix = np.random.default_rng(2).normal(size=(10, 32)).astype(np.float32)
    matrix[0] = 0
    q, scale = quantize_int8(matrix)
    reconstructed = dequantize_int8(q, scale)
    assert q.dtype == np.int8
    assert np.all(np.abs(matrix - reconstructed) <= scale / 2 + 1e-6)


def test_turbo_equivalence_benchmark(context):
    context.turbo.enabled = False
    result = compare_turbo(context, repeats=3)
    assert result["numerically_equivalent"]
    assert result["turbo_speedup"] > 0
    assert context.turbo.enabled is False


def test_multithreading_determinism(config):
    def worker(threads):
        local = copy.deepcopy(config)
        local.turbo.deterministic = False
        local.turbo.threads = threads
        context = RuntimeContext(local)
        context.model = Model(context)
        with context.turbo.scope():
            return context.model.forward(np.array([[257, 1, 2]])).logits.data
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, [1, 2]))
    np.testing.assert_allclose(results[0], results[1], atol=1e-6)


def test_cli_orchestration(config, tmp_path):
    root = Path(__file__).resolve().parents[2]
    path = tmp_path / "config.yaml"
    config.training.steps_per_epoch = 2
    path.write_text(yaml.safe_dump(config.to_dict()))
    def cli(*args, input=None):
        command = [sys.executable, str(root / "nexus.py"), *args, "--config", str(path)]
        result = subprocess.run(command, cwd=root, input=input, text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr
        return result
    cli("create")
    generated = cli("use", "--model", config.output, "--prompt", "test", "--max-new-tokens", "3")
    assert json.loads(generated.stdout)["metrics"]["generated_tokens"] <= 3
    cli("finetune", "--model", config.output, "--output", str(tmp_path / "ft.nexus"))
    report = tmp_path / "bench.json"
    cli("bench", "--model", config.output, "--output", str(report))
    assert json.loads(report.read_text())["turbo_comparison"]["numerically_equivalent"]
    cli("status", "--model", config.output)
    menu = cli(input="6\n0\n")
    assert "CONTROL CENTER" in menu.stdout


def test_inference_no_graph_and_invalid_input(context):
    output = context.model.forward(np.array([[257, 97]]))
    assert output.logits.requires_grad is False
    assert output.logits.parents == ()
    with pytest.raises(ValueError):
        context.model.forward(np.array([[999]]))


def test_optional_transformer_baseline(config):
    pytest.importorskip("torch")
    from nexus.orchestrator import Orchestrator
    from nexus.benchmarks.baseline import run_baseline
    config.training.steps_per_epoch = 2
    app = Orchestrator(config)
    app.create()
    result = run_baseline(app.context)
    assert np.isfinite(result["validation_ce_nats"])
    assert result["parameters"] > 0


def test_menu_create_finetune_use(config, tmp_path):
    root = Path(__file__).resolve().parents[2]
    path = tmp_path / "menu.yaml"
    config.training.steps_per_epoch = 1
    config.runtime.max_new_tokens = 2
    path.write_text(yaml.safe_dump(config.to_dict()))
    ft = tmp_path / "ft.nexus"
    inputs = f"1\n2\n{config.output}\n{ft}\n3\n{ft}\nhi\n/exit\n0\n"
    result = subprocess.run([sys.executable, str(root / "nexus.py"), "--config", str(path)],
                            cwd=root, input=inputs, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "[ERROR]" not in result.stderr
    assert "NEXUS>" in result.stdout
    assert ft.exists()
