from __future__ import annotations

from dataclasses import dataclass

from src.preprocessing.standard import StandardSignalPreprocessor


@dataclass
class BasePreprocessor(StandardSignalPreprocessor):
    normalize: bool = True

    def __post_init__(self) -> None:
        self.detrend = False
        self.scaler = "zscore" if self.normalize else "none"
        super().__post_init__()


@dataclass
class FilterPreprocessor(StandardSignalPreprocessor):
    normalize: bool = True

    def __post_init__(self) -> None:
        self.scaler = "zscore" if self.normalize else "none"
        super().__post_init__()


@dataclass
class DomainNormPreprocessor(StandardSignalPreprocessor):
    mode: str = "per_record"
    normalize: bool = True

    def __post_init__(self) -> None:
        if not self.normalize:
            self.scaler = "none"
        elif self.mode == "per_record":
            self.scaler = "robust"
        elif self.mode == "zscore":
            self.scaler = "zscore"
        else:
            raise ValueError(f"Unknown domain normalization mode: {self.mode}")

        super().__post_init__()