"""One console logger and machine-readable metrics journal per context."""
import json
import logging
from pathlib import Path
import time
from typing import Any


def get_logger() -> logging.Logger:
    logger = logging.getLogger("nexus")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def journal(directory: str, event: dict[str, Any]) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.time(), **event}, allow_nan=False) + "\n")
