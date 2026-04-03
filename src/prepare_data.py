from pathlib import Path

import yaml

from src.data.windowing import build_window_manifest


def main():
    config_path = Path("configs/base.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    records_csv = config["data"]["raw_records_csv"]
    output_dir = config["data"]["processed_dir"]
    window_size = config["data"]["window_size"]
    stride = config["data"]["stride"]
    drop_last = config["data"].get("drop_last", True)

    intervals_column = config["data"].get("anomaly_intervals_column", "anomaly_intervals")
    window_label_mode = config["data"].get("window_label_mode", "any_overlap")
    min_anomaly_fraction = config["data"].get("min_anomaly_fraction", 0.0)

    manifest = build_window_manifest(
        records_csv=records_csv,
        output_dir=output_dir,
        window_size=window_size,
        stride=stride,
        drop_last=drop_last,
        anomaly_intervals_column=intervals_column,
        window_label_mode=window_label_mode,
        min_anomaly_fraction=min_anomaly_fraction,
    )

    print("Processed manifest created")
    print(f"Total windows: {len(manifest)}")
    print(manifest.head())


if __name__ == "__main__":
    main()