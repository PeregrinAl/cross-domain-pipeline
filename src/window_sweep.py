# src/window_sweep.py

import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


CONFIG_PATH = Path("configs/base.yaml")
OUT_DIR = Path("experiments/window_sweep")
VARIANTS = ["raw_only", "tfr_only", "fused"]

WINDOW_GRID = [
    {"window_size": 1024, "stride": 256},
    {"window_size": 2048, "stride": 512},
    {"window_size": 2048, "stride": 256},
    {"window_size": 4096, "stride": 1024},
]


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def run_cmd(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def make_run_name(window_size: int, stride: int) -> str:
    return f"ws_{window_size}_st_{stride}"


def collect_window_balance(manifest_path: Path):
    df = pd.read_csv(manifest_path)

    rows = []
    for split in ["train", "val", "test", "adapt"]:
        for domain in ["source", "target"]:
            subset = df[(df["split"] == split) & (df["domain"] == domain)]
            if subset.empty:
                continue

            normal_count = int((subset["label"] == 0).sum())
            anomaly_count = int((subset["label"] == 1).sum())

            rows.append(
                {
                    "split": split,
                    "domain": domain,
                    "num_windows": int(len(subset)),
                    "num_normal_windows": normal_count,
                    "num_anomaly_windows": anomaly_count,
                }
            )

    return pd.DataFrame(rows)


def flatten_balance(balance_df: pd.DataFrame):
    flat = {}
    for _, row in balance_df.iterrows():
        prefix = f'{row["split"]}_{row["domain"]}'
        flat[f"{prefix}_num_windows"] = row["num_windows"]
        flat[f"{prefix}_num_normal_windows"] = row["num_normal_windows"]
        flat[f"{prefix}_num_anomaly_windows"] = row["num_anomaly_windows"]
    return flat


def read_variant_summary(source_only_dir: Path, variant: str):
    summary_path = source_only_dir / variant / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    original_config = load_yaml(CONFIG_PATH)
    all_rows = []

    try:
        for combo in WINDOW_GRID:
            window_size = combo["window_size"]
            stride = combo["stride"]
            run_name = make_run_name(window_size, stride)

            run_dir = OUT_DIR / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 80)
            print(f"RUN: {run_name}")
            print("=" * 80)

            run_config = copy.deepcopy(original_config)

            run_config["data"]["window_size"] = window_size
            run_config["data"]["stride"] = stride

            run_source_only_dir = OUT_DIR / run_name / "source_only_supervised"
            run_config["outputs"]["source_only_dir"] = str(run_source_only_dir).replace("\\", "/")

            save_yaml(CONFIG_PATH, run_config)

            run_cmd([sys.executable, "-m", "src.prepare_data"])

            manifest_path = Path(run_config["data"]["manifest_path"])
            balance_df = collect_window_balance(manifest_path)
            balance_df.to_csv(OUT_DIR / run_name / "window_balance.csv", index=False)

            balance_flat = flatten_balance(balance_df)

            for variant in VARIANTS:
                run_cmd(
                    [
                        sys.executable,
                        "-m",
                        "src.train_source_only",
                        "--config",
                        str(CONFIG_PATH),
                        "--variant",
                        variant,
                    ]
                )

                summary = read_variant_summary(run_source_only_dir, variant)

                row = {
                    "run_name": run_name,
                    "window_size": window_size,
                    "stride": stride,
                    "variant": variant,
                    "source_val_roc_auc": summary["source_val"]["roc_auc"],
                    "target_test_roc_auc": summary["target_test"]["roc_auc"],
                    "source_val_pr_auc": summary["source_val"]["pr_auc"],
                    "target_test_pr_auc": summary["target_test"]["pr_auc"],
                    "source_val_f1": summary["source_val"]["f1"],
                    "target_test_f1": summary["target_test"]["f1"],
                    "delta_roc_auc": summary["source_val"]["roc_auc"] - summary["target_test"]["roc_auc"],
                    "delta_pr_auc": summary["source_val"]["pr_auc"] - summary["target_test"]["pr_auc"],
                }

                row.update(balance_flat)
                all_rows.append(row)

            pd.DataFrame(all_rows).to_csv(OUT_DIR / "window_sweep_summary.csv", index=False)

    finally:
        save_yaml(CONFIG_PATH, original_config)

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(OUT_DIR / "window_sweep_summary.csv", index=False)

    print("\nSaved:", OUT_DIR / "window_sweep_summary.csv")
    print(summary_df)


if __name__ == "__main__":
    main()