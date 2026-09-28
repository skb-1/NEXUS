"""Lossless, training-free UTF-8 byte tokenizer."""
import numpy as np


class ByteTokenizer:
    PAD, BOS, EOS = 256, 257, 258
    vocab_size = 259

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> list[int]:
        return ([self.BOS] if bos else []) + list(text.encode("utf-8")) + ([self.EOS] if eos else [])

    def decode(self, tokens) -> str:
        return bytes(int(t) for t in tokens if 0 <= t < 256).decode("utf-8", errors="replace")

    def state(self) -> dict[str, np.ndarray]:
        return {"byte_ids": np.arange(256, dtype=np.int32),
                "special_ids": np.array([self.PAD, self.BOS, self.EOS], np.int32)}
