import numpy as np
from nexus.position import position_features
from nexus.tensor import Tensor


def encode(ids: np.ndarray, embedding: Tensor, position: Tensor, context: int) -> Tensor:
    return embedding[ids] + position_features(ids.shape[1], context, embedding.backend) @ position
