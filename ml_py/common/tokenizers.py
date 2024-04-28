from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class EncodeConfig:
    max_length: int | None = None
    pad: bool = False


class SimpleTokenizer:
    def pad_token(self) -> int:
        raise NotImplementedError()

    def eos_token(self) -> int:
        raise NotImplementedError()

    def vocab_size(self) -> int:
        raise NotImplementedError()

    def encode(self, text: str, cfg: EncodeConfig) -> np.ndarray:
        raise NotImplementedError()

    def decode(self, seq: np.ndarray) -> str:
        raise NotImplementedError()


@dataclass
class ByteTokenizerConfig:
    extra_ids: int = 0


class ByteTokenizer(SimpleTokenizer):
    cfg: ByteTokenizerConfig
    byte_offset: int

    def __init__(self, cfg: ByteTokenizerConfig):
        self.cfg = cfg
        self.byte_offset = cfg.extra_ids + 2

    def pad_token(self) -> int:
        return 0

    def eos_token(self) -> int:
        return 1

    def vocab_size(self) -> int:
        return self.byte_offset + 256

    def encode(self, text: str, cfg: EncodeConfig) -> np.ndarray:
        raw = text.encode("utf-8")
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) + self.byte_offset

        if cfg.max_length is not None:
            arr = arr[: min(arr.shape[0], cfg.max_length)]
            if cfg.pad:
                num_pad = cfg.max_length - arr.shape[0]
                arr = np.pad(
                    arr, (0, num_pad), "constant", constant_values=self.pad_token()
                )

        return arr

    def decode(self, seq: np.ndarray) -> str:
        seq = seq[seq >= self.byte_offset]
        seq = seq - self.byte_offset
        raw = seq.tobytes()
        return raw.decode("utf-8")
