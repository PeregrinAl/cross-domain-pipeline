from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cross_domain_pipeline.api import plan_benchmark, profile_data


def _write_config(tmp_path: Path, records_csv: Path) -> Path:
    config = {
        "seed": 42,
        "data": {
            "raw_records_csv": str(records_csv),
            "processed_dir": str(tmp_path / "processed"),
            "window_size": 1024,
            "stride": 512,
        },
        "model": {},
        "training": {},
        "outputs": {},
    }

    config_path = tmp_path / "config.yaml"

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    return config_path


def test_single_channel_profile_excludes_graph(tmp_path: Path):
    paths = []

    for i in range(4):
        signal = np.sin(np.linspace(0, 20, 2048)).astype(np.float32)
        path = tmp_path / f"signal_{i}.npy"
        np.save(path, signal)
        paths.append(path)

    records = pd.DataFrame(
        {
            "path": [str(p) for p in paths],
            "label": [0, 0, 1, 1],
            "domain": ["source", "source", "target", "target"],
            "record_id": ["s0", "s1", "t0", "t1"],
            "split": ["train", "train", "adapt", "test"],
        }
    )

    records_csv = tmp_path / "records.csv"
    records.to_csv(records_csv, index=False)

    config_path = _write_config(tmp_path, records_csv)

    profile = profile_data(config_path, source="raw", max_files=10)

    assert profile.n_rows == 4
    assert profile.n_sampled_arrays == 4
    assert profile.likely_multichannel is False
    assert "graph_hybrid" not in profile.recommended_model_families
    assert "graph_hybrid" in profile.excluded_model_families
    assert "sfda_prototype" in profile.recommended_adaptation_methods


def test_multichannel_profile_allows_graph_when_channels_are_correlated(tmp_path: Path):
    paths = []

    base = np.sin(np.linspace(0, 40, 4096)).astype(np.float32)

    for i in range(6):
        signal = np.stack(
            [
                base,
                base * 0.9 + 0.01,
                base * 1.1 - 0.01,
            ],
            axis=0,
        ).astype(np.float32)

        path = tmp_path / f"multi_{i}.npy"
        np.save(path, signal)
        paths.append(path)

    records = pd.DataFrame(
        {
            "path": [str(p) for p in paths],
            "label": [0, 0, 0, 1, 1, 1],
            "domain": ["source", "source", "source", "target", "target", "target"],
            "record_id": [f"r{i}" for i in range(6)],
            "split": ["train", "train", "train", "adapt", "adapt", "test"],
        }
    )

    records_csv = tmp_path / "records.csv"
    records.to_csv(records_csv, index=False)

    config_path = _write_config(tmp_path, records_csv)

    profile = profile_data(config_path, source="raw", max_files=10)

    assert profile.likely_multichannel is True
    assert profile.n_channels_max == 3
    assert profile.channel_correlation_score is not None
    assert profile.channel_correlation_score > 0.1
    assert "graph_hybrid" in profile.recommended_model_families


def test_benchmark_plan_contains_core_candidates(tmp_path: Path):
    paths = []

    for i in range(8):
        signal = np.random.default_rng(i).normal(size=2048).astype(np.float32)
        path = tmp_path / f"signal_{i}.npy"
        np.save(path, signal)
        paths.append(path)

    records = pd.DataFrame(
        {
            "path": [str(p) for p in paths],
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
            "domain": ["source", "source", "source", "source", "target", "target", "target", "target"],
            "record_id": [f"r{i}" for i in range(8)],
            "split": ["train", "train", "val", "val", "adapt", "adapt", "test", "test"],
        }
    )

    records_csv = tmp_path / "records.csv"
    records.to_csv(records_csv, index=False)

    config_path = _write_config(tmp_path, records_csv)

    plan = plan_benchmark(
        config=config_path,
        source="raw",
        mode="compact",
        max_files=20,
        max_candidates=12,
    )

    candidate_names = {candidate.model for candidate in plan.candidates}

    assert "statistical_threshold" in candidate_names
    assert "pca_reconstruction" in candidate_names
    assert "isolation_forest" in candidate_names
    assert "conv1d_autoencoder" in candidate_names
    assert "fusion_autoencoder" in candidate_names
    assert "graph_hybrid" not in candidate_names