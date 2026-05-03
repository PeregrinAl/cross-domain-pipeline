from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from cross_domain_pipeline.config import PipelineConfig
from cross_domain_pipeline.model_selection.data_profile import DataProfile, profile_from_config


def _load_config(config: str | Path | PipelineConfig) -> PipelineConfig:
    if isinstance(config, PipelineConfig):
        return config

    return PipelineConfig.from_yaml(config)


def prepare_windows(config: str | Path | PipelineConfig):
    cfg = _load_config(config)

    from src.data.windowing import build_window_manifest

    data = cfg.data

    manifest = build_window_manifest(
        records_csv=data["raw_records_csv"],
        output_dir=data["processed_dir"],
        window_size=data["window_size"],
        stride=data["stride"],
        drop_last=data.get("drop_last", True),
        anomaly_intervals_column=data.get("anomaly_intervals_column", "anomaly_intervals"),
        window_label_mode=data.get("window_label_mode", "any_overlap"),
        min_anomaly_fraction=data.get("min_anomaly_fraction", 0.0),
    )

    return manifest


def profile_data(
    config: str | Path | PipelineConfig,
    source: str = "auto",
    max_files: int = 200,
) -> DataProfile:
    cfg = _load_config(config)

    return profile_from_config(
        config=cfg,
        source=source,
        max_files=max_files,
    )


def train_source_only(
    config: str | Path | PipelineConfig,
    variant: str,
) -> subprocess.CompletedProcess:
    cfg = _load_config(config)
    config_path = cfg.require_path()

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train_source_only",
            "--config",
            str(config_path),
            "--variant",
            variant,
        ],
        check=True,
    )


def load_config_dict(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)