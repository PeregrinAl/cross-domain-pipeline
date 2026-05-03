from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from cross_domain_pipeline.config import PipelineConfig
from cross_domain_pipeline.model_selection.benchmark_plan import build_benchmark_plan
from cross_domain_pipeline.model_selection.data_profile import profile_from_config


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
):
    cfg = _load_config(config)

    return profile_from_config(
        config=cfg,
        source=source,
        max_files=max_files,
    )


def plan_benchmark(
    config: str | Path | PipelineConfig,
    source: str = "auto",
    max_files: int = 200,
    mode: str = "compact",
    max_candidates: int = 12,
    include_research_only: bool = False,
):
    profile = profile_data(
        config=config,
        source=source,
        max_files=max_files,
    )

    return build_benchmark_plan(
        profile=profile,
        mode=mode,
        max_candidates=max_candidates,
        include_research_only=include_research_only,
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

def train_classical(
    config: str | Path | PipelineConfig,
    method: str,
    source: str = "auto",
    output_dir: str | Path | None = None,
) -> subprocess.CompletedProcess:
    cfg = _load_config(config)
    config_path = cfg.require_path()

    command = [
        sys.executable,
        "-m",
        "cross_domain_pipeline.classical.train_classical",
        "--config",
        str(config_path),
        "--method",
        method,
        "--source",
        source,
    ]

    if output_dir is not None:
        command.extend(["--output-dir", str(output_dir)])

    return subprocess.run(command, check=True)

def execute_plan(
    config: str | Path | PipelineConfig,
    plan_path: str | Path | None = None,
    output_dir: str | Path = "experiments/benchmark_execution",
    source: str = "auto",
    mode: str = "compact",
    max_files: int = 200,
    max_candidates: int = 12,
    dry_run: bool = False,
) -> subprocess.CompletedProcess:
    cfg = _load_config(config)
    config_path = cfg.require_path()

    command = [
        sys.executable,
        "-m",
        "cross_domain_pipeline.benchmark.execute_plan",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--source",
        source,
        "--mode",
        mode,
        "--max-files",
        str(max_files),
        "--max-candidates",
        str(max_candidates),
    ]

    if plan_path is not None:
        command.extend(["--plan", str(plan_path)])

    if dry_run:
        command.append("--dry-run")

    return subprocess.run(command, check=True)


def load_config_dict(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
