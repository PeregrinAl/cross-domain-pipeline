from dataclasses import dataclass
import numpy as np


@dataclass
class BasePreprocessor:
    normalize: bool = True
    eps: float = 1e-8

    def __call__(self, x: np.ndarray, metadata: dict | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)

        if self.normalize:
            mean = float(np.mean(x))
            std = float(np.std(x))
            x = (x - mean) / (std + self.eps)

        return x.astype(np.float32)


@dataclass
class FilterPreprocessor:
    normalize: bool = True
    detrend: bool = True
    eps: float = 1e-8

    def __call__(self, x: np.ndarray, metadata: dict | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)

        if self.detrend:
            x = x - np.mean(x)

        if self.normalize:
            std = float(np.std(x))
            x = x / (std + self.eps)

        return x.astype(np.float32)


@dataclass
class DomainNormPreprocessor:
    mode: str = "per_record"
    normalize: bool = True
    eps: float = 1e-8

    def __call__(self, x: np.ndarray, metadata: dict | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)

        if not self.normalize:
            return x.astype(np.float32)

        if self.mode == "per_record":
            median = float(np.median(x))
            mad = float(np.median(np.abs(x - median)))
            x = (x - median) / (mad + self.eps)
            return x.astype(np.float32)

        if self.mode == "zscore":
            x = (x - np.mean(x)) / (np.std(x) + self.eps)
            return x.astype(np.float32)

        raise ValueError(f"Unknown domain normalization mode: {self.mode}")