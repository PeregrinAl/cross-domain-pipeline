from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardSignalPreprocessor:
    detrend: bool = False
    scaler: str = "zscore"
    clip_quantile: float | None = None
    eps: float = 1e-8

    def __post_init__(self) -> None:
        allowed_scalers = {"none", "zscore", "robust"}

        if self.scaler not in allowed_scalers:
            raise ValueError(
                f"Unsupported scaler: {self.scaler}. "
                f"Expected one of: {sorted(allowed_scalers)}"
            )

        if self.clip_quantile is not None:
            if not 0.0 < self.clip_quantile < 1.0:
                raise ValueError("clip_quantile must be between 0 and 1.")

    def __call__(self, x: np.ndarray, metadata: dict | None = None) -> np.ndarray:
        signal = np.asarray(x, dtype=np.float32)

        if self.detrend:
            signal = self._remove_linear_trend(signal)

        if self.clip_quantile is not None:
            signal = self._clip_by_quantile(signal, self.clip_quantile)

        if self.scaler == "none":
            return signal.astype(np.float32)

        if self.scaler == "zscore":
            return self._zscore(signal)

        if self.scaler == "robust":
            return self._robust_scale(signal)

        raise ValueError(f"Unsupported scaler: {self.scaler}")

    def _remove_linear_trend(self, x: np.ndarray) -> np.ndarray:
        if x.size < 2:
            return x

        t = np.linspace(-1.0, 1.0, x.size, dtype=np.float32)
        slope, intercept = np.polyfit(t, x, deg=1)
        trend = slope * t + intercept

        return (x - trend).astype(np.float32)

    def _clip_by_quantile(self, x: np.ndarray, q: float) -> np.ndarray:
        limit = float(np.quantile(np.abs(x), q))

        if limit <= self.eps:
            return x

        return np.clip(x, -limit, limit).astype(np.float32)

    def _zscore(self, x: np.ndarray) -> np.ndarray:
        mean = float(np.mean(x))
        std = float(np.std(x))

        return ((x - mean) / (std + self.eps)).astype(np.float32)

    def _robust_scale(self, x: np.ndarray) -> np.ndarray:
        median = float(np.median(x))
        mad = float(np.median(np.abs(x - median)))

        return ((x - median) / (mad + self.eps)).astype(np.float32)