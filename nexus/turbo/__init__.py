"""Native NumPy kernels, bounded BLAS pools, and an explicit reference path."""
from contextlib import contextmanager
import os
import threading
from collections.abc import Iterator

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

from nexus.config import TurboConfig

# threadpoolctl changes process-global pools. Serialize scoped pool changes.
_POOL_LOCK = threading.RLock()


class Turbo:
    def __init__(self, config: TurboConfig):
        self.enabled = config.enabled
        self.threads = (1 if config.deterministic else
                        min(4, os.cpu_count() or 1) if config.threads == "auto" else config.threads)

    @contextmanager
    def scope(self) -> Iterator[None]:
        with _POOL_LOCK, threadpool_limits(limits=self.threads):
            yield

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Kernel requires matrices")
        return a @ b if self.enabled else np.einsum("ik,kj->ij", a, b, optimize=False)

    def status(self) -> dict:
        with self.scope():
            pools = threadpool_info()
        return {"enabled": self.enabled, "threads": self.threads,
                "backend": "NumPy BLAS" if self.enabled else "NumPy reference einsum",
                "native_pools_within_scope": pools,
                "custom_simd": False, "numa": False, "lazy_weights": False}
