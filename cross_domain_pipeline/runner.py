from __future__ import annotations

from pathlib import Path

from cross_domain_pipeline.api import prepare_windows, train_source_only
from cross_domain_pipeline.config import PipelineConfig


class BenchmarkRunner:
    def __init__(self, config: str | Path | PipelineConfig):
        self.config = (
            config
            if isinstance(config, PipelineConfig)
            else PipelineConfig.from_yaml(config)
        )

    @classmethod
    def from_config(cls, path: str | Path) -> "BenchmarkRunner":
        return cls(PipelineConfig.from_yaml(path))

    def prepare_windows(self):
        return prepare_windows(self.config)

    def train_source_only(self, variant: str):
        return train_source_only(self.config, variant=variant)

    def train_source_only_baselines(self, variants: tuple[str, ...] = ("raw_only", "tfr_only", "fused")):
        results = {}

        for variant in variants:
            results[variant] = self.train_source_only(variant)

        return results