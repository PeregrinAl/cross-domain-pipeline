from pathlib import Path
import yaml


class BenchmarkRunner:
    def __init__(self, config: dict):
        self.config = config

    @classmethod
    def from_config(cls, path: str | Path) -> "BenchmarkRunner":
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def run(self) -> None:
        raise NotImplementedError(
            "BenchmarkRunner.run() will be implemented after method registry is added."
        )