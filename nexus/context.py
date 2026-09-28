from __future__ import annotations
from dataclasses import dataclass, field
import logging
from typing import Any, TYPE_CHECKING

import numpy as np

from nexus.config import Config
from nexus.logging import get_logger, journal
from nexus.turbo import Turbo

if TYPE_CHECKING:
    from nexus.core import Model


@dataclass
class RuntimeContext:
    config: Config
    model: Model | None = None
    state: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    logger: logging.Logger = field(default_factory=get_logger)
    turbo: Turbo = field(init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.config.validate()
        self.turbo = Turbo(self.config.turbo)
        self.rng = np.random.default_rng(self.config.seed)

    def record(self, category: str, **values: Any) -> None:
        self.metrics.update(values)
        self.logger.info("[%s] %s", category, " ".join(f"{k}={v}" for k, v in values.items()))
        journal(self.config.log_dir, {"category": category, **values})
