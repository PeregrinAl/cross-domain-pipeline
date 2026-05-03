from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    data: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]
    outputs: dict[str, Any]
    seed: int = 42
    path: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        config_path = Path(path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls(
            data=raw.get("data", {}),
            model=raw.get("model", {}),
            training=raw.get("training", {}),
            outputs=raw.get("outputs", {}),
            seed=int(raw.get("seed", 42)),
            path=config_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "data": self.data,
            "model": self.model,
            "training": self.training,
            "outputs": self.outputs,
        }

    def require_path(self) -> Path:
        if self.path is None:
            raise ValueError("Config was not loaded from a YAML file.")
        return self.path