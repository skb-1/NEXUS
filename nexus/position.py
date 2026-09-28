import numpy as np
from nexus.tensor import Tensor


def position_features(length: int, context: int, backend) -> Tensor:
    t = np.arange(length, dtype=np.float32)
    x = t / context
    return Tensor(np.stack([x, x * x, np.sin(t), np.cos(t),
                            np.sin(t / 16), np.cos(t / 16)], axis=-1), backend)
