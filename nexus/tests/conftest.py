import pytest
from nexus.config import Config, DatasetConfig
from nexus.context import RuntimeContext
from nexus.core import Model


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "train.txt"
    path.write_text("alpha beta gamma\nbeta alpha beta\nsmall example\n", encoding="utf-8")
    validation = tmp_path / "validation.txt"
    validation.write_text("gamma beta\n", encoding="utf-8")
    c = Config()
    c.log_dir = str(tmp_path / "logs")
    c.output = str(tmp_path / "model.nexus")
    c.model.hidden_dim = 8
    c.model.rank = 3
    c.model.depth = 1
    c.model.context = 12
    c.training.datasets = [DatasetConfig(str(path))]
    c.training.validation = [DatasetConfig(str(validation))]
    c.training.batch_size = 2
    c.training.shuffle_buffer = 3
    c.training.epochs = 1
    c.training.steps_per_epoch = 6
    c.training.checkpoint_every = 100
    c.training.log_every = 100
    c.training.eval_batches = 1
    return c


@pytest.fixture
def context(config):
    ctx = RuntimeContext(config)
    ctx.model = Model(ctx)
    return ctx
