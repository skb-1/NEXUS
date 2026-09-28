import math


def cosine_lr(step: int, total: int, initial: float, minimum_ratio: float) -> float:
    fraction = min(step / max(1, total - 1), 1.0)
    return initial * (minimum_ratio + (1 - minimum_ratio) * (1 + math.cos(math.pi * fraction)) / 2)
