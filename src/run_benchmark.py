import argparse
import subprocess
import sys

import pandas as pd
import yaml

from src.benchmark.grid import build_stage_1_grid


REPRESENTATION_TO_TRAIN_ARGS = {
    "raw_time": {
        "variant": "raw_only",
        "tfr_type": None,
    },
    "stft_spectrogram": {
        "variant": "tfr_only",
        "tfr_type": "stft",
    },
    "cwt_scalogram": {
        "variant": "tfr_only",
        "tfr_type": "cwt",
    },
    "fused_multiview": {
        "variant": "fused",
        "tfr_type": "stft",
    },
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

    if representation not in REPRESENTATION_TO_TRAIN_ARGS:
        return None

    train_args = REPRESENTATION_TO_TRAIN_ARGS[representation]
    variant = train_args["variant"]
    tfr_type = train_args["tfr_type"]

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

    if tfr_type is not None:
        cmd.extend(["--tfr-type", tfr_type])

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
        print("Implemented now: raw_time, stft_spectrogram, cwt_scalogram, fused_multiview")        
        return

    print("\nExecuting:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()