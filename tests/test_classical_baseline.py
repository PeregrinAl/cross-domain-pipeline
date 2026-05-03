from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cross_domain_pipeline.classical.features import build_feature_table
from cross_domain_pipeline.classical.train_classical import run_classical_baseline


def _write_dataset(tmp_path: Path) -> Path:
    paths = []
    labels = []
    domains = []
    splits = []
    record_ids = []

    rng = np.random.default_rng(42)

    for i in range(12):
        if i < 8:
            signal = rng.normal(0.0, 1.0, size=1024).astype(np.float32)
            label = 0
        else:
            signal = rng.normal(0.0, 1.0, size=1024).astype(np.float32)
            signal[300:360] += 6.0
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


def test_feature_table_is_created(tmp_path: Path):
    config_path = _write_dataset(tmp_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    features = build_feature_table(cfg["data"]["raw_records_csv"])

    assert not features.empty
    assert "rms" in features.columns
    assert "spectral_entropy" in features.columns


def test_classical_baseline_runs(tmp_path: Path):
    config_path = _write_dataset(tmp_path)

    summary = run_classical_baseline(
        config_path=config_path,
        method="isolation_forest",
        source="raw",
        output_dir=tmp_path / "run",
    )

    assert summary["method"] == "isolation_forest"
    assert summary["n_train_normal"] > 0
    assert summary["n_eval"] > 0
    assert (tmp_path / "run" / "features.csv").exists()
    assert (tmp_path / "run" / "scores.csv").exists()
    assert (tmp_path / "run" / "summary.json").exists()