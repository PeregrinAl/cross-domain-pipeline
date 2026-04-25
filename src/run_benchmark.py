import argparse
import subprocess
import sys

import pandas as pd
import yaml

from src.benchmark.grid import build_stage_1_grid


REPRESENTATION_TO_VARIANT = {
    "raw_time": "raw_only",
    "stft_spectrogram": "tfr_only",
    "fused_multiview": "fused",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/benchmark_grid.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument("--train-config", type=str, default=None)
    return parser.parse_args()


def build_train_command(row: dict, benchmark_config: dict, train_config_override: str | None):
    if row["adaptation"] != "source_only":
        raise NotImplementedError(
            f"Only source_only stage-1 execution is supported now, got: {row['adaptation']}"
        )

    representation = row["representation"]

    if representation not in REPRESENTATION_TO_VARIANT:
        return None

    variant = REPRESENTATION_TO_VARIANT[representation]

    train_config = (
        train_config_override
        or row.get("config_path")
        or benchmark_config.get("default_train_config")
        or "configs/base.yaml"
    )

    cmd = [
        sys.executable,
        "-m",
        "src.train_source_only",
        "--config",
        train_config,
        "--variant",
        variant,
    ]

    preprocessing = row["preprocessing"]
    if preprocessing != "none":
        cmd.extend(["--preprocessing", preprocessing])

    return cmd


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        benchmark_config = yaml.safe_load(f)

    grid = build_stage_1_grid(benchmark_config)
    df = pd.DataFrame(grid)

    if args.dry_run or args.row is None:
        print(df)
        print(f"\nTotal stage-1 runs: {len(df)}")
        return

    if args.row < 0 or args.row >= len(grid):
        raise IndexError(f"Row index out of range: {args.row}. Grid size: {len(grid)}")

    row = grid[args.row]

    print("Selected benchmark row:")
    print(pd.DataFrame([row]))

    cmd = build_train_command(
        row=row,
        benchmark_config=benchmark_config,
        train_config_override=args.train_config,
    )

    if cmd is None:
        print(
            "\nSKIPPED: representation is not implemented in train_source_only yet: "
            f"{row['representation']}"
        )
        print("Implemented now: raw_time, stft_spectrogram, fused_multiview")
        return

    print("\nExecuting:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()