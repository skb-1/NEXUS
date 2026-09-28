import json
import numpy as np
import pytest
from nexus.config import DatasetConfig
from nexus.training.datasets import DatasetEngine, chunks


def test_sampler_exact_restore_and_weights(config, tmp_path):
    second = tmp_path / "code.jsonl"
    second.write_text('{"text": "code code code"}\n')
    specs = config.training.datasets + [DatasetConfig(str(second), weight=3)]
    engine = DatasetEngine(specs, 8, 3, 42)
    engine.batch(10)
    state = engine.state_dict()
    expected = engine.batch(200)
    loaded = DatasetEngine(specs, 8, 3, 42, state=state)
    actual = loaded.batch(200)
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)
    assert 0.65 < loaded.draws[1] / sum(loaded.draws) < 0.85
    second.write_text('{"text": "changed"}\n')
    with pytest.raises(ValueError, match="changed"):
        DatasetEngine(specs, 8, 3, 42, state=state)


def test_instruction_mask(tmp_path):
    path = tmp_path / "instruction.jsonl"
    path.write_text(json.dumps({"prompt": "abc", "response": "de"}) + "\n")
    samples = list(chunks(DatasetConfig(str(path)), 12, True))
    ids, mask = samples[0]
    assert ids[:8] == [257, 97, 98, 99, 10, 100, 101, 258]
    assert mask[:8] == [0, 0, 0, 0, 0, 1, 1, 1]
    assert sum(mask) == 3


def test_empty_dataset_errors(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("\n")
    with pytest.raises(ValueError, match="no supervised"):
        DatasetEngine([DatasetConfig(str(path))], 8, 2, 0)


def test_chunk_targets_not_lost(tmp_path):
    path = tmp_path / "long.txt"
    text = "abcdefghijklmnop"
    path.write_text(text + "\n")
    targets = []
    for ids, mask in chunks(DatasetConfig(str(path)), 4, False):
        targets.extend(i for i, m in zip(ids[1:], mask[1:]) if m)
    assert targets == list(text.encode()) + [258]
