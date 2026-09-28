"""Causal compressed-context mixer; intentionally no pairwise attention matrix."""
import numpy as np
from nexus.tensor import Tensor


def summaries(x: Tensor, window: int) -> tuple[Tensor, Tensor]:
    length = x.data.shape[1]
    prefix = x.cumsum(axis=1)
    t = np.arange(length)
    previous = prefix[:, np.maximum(t - window, 0), :] * (t >= window)[None, :, None]
    local = (prefix - previous) / np.minimum(t + 1, window)[None, :, None]
    global_summary = prefix / (t + 1)[None, :, None]
    return local, global_summary
