"""Experimental symmetric per-row int8 storage codec (not used for inference)."""
import numpy as np


def quantize_int8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite matrix")
    scale = np.maximum(np.max(np.abs(x), axis=1, keepdims=True) / 127, 1e-12).astype(np.float32)
    return np.clip(np.rint(x / scale), -127, 127).astype(np.int8), scale


def dequantize_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale
