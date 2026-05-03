from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cross_domain_pipeline.benchmark.execute_plan import execute_benchmark_plan


def _write_dataset(tmp_path: Path) -> Path:
    paths = []
    labels = []
    domains = []
    splits = []
    record_ids = []

    rng = np.random.default_rng(123)

    for i in range(12):
        if i < 8:
            signal = rng.normal(0.0, 1.0, size=1024).astype(np.float32)
            label = 0
        else:
            signal = rng.normal(0.0, 1.0, size=1024).astype(np.float32)
            signal[300:360] += 5.0
            label = 1

        path = tmp_path / f"signal_{i}.npy"
        np.save(path, signal)

        paths.append(str(path))
        labels.append(label)
        domains.append("source" if i < 6 else "target")

        if i < 6:
            splits.append("train")
        elif i < 8:
            splits.append("val")
        else:
            splits.append("test")

        record_ids.append(f"r{i}")

    records = pd.DataFrame(
        {
            "path": paths,
            "label": labels,
            "domain": domains,
            "record_id": record_ids,
            "split": splits,
        }
    )

    records_csv = tmp_path / "records.csv"
    records.to_csv(records_csv, index=False)

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


def test_execute_plan_dry_run(tmp_path: Path):
    config_path = _write_dataset(tmp_path)

    result = execute_benchmark_plan(
        config_path=config_path,
        output_dir=tmp_path / "execution",
        source="raw",
        mode="compact",
        max_candidates=5,
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["n_candidates"] > 0
    assert all(entry["status"] == "planned" for entry in result["entries"])
    assert (tmp_path / "execution" / "execution_summary.json").exists()
    assert (tmp_path / "execution" / "execution_summary.csv").exists()


def test_execute_plan_runs_classical_and_marks_neural_not_implemented(tmp_path: Path):
    config_path = _write_dataset(tmp_path)

    result = execute_benchmark_plan(
        config_path=config_path,
        output_dir=tmp_path / "execution",
        source="raw",
        mode="compact",
        max_candidates=6,
        dry_run=False,
    )

    statuses = {entry["status"] for entry in result["entries"]}

    assert "completed" in statuses
    assert "not_implemented" in statuses
    assert result["n_completed"] >= 1
    assert (tmp_path / "execution" / "execution_summary.json").exists()
    assert (tmp_path / "execution" / "execution_summary.csv").exists()