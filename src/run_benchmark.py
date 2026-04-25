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
    "fused_cwt": {
        "variant": "fused",
        "tfr_type": "cwt",
    },
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/benchmark_grid.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--stage-1-all", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--train-config", type=str, default=None)
    return parser.parse_args()

def parse_rows(rows_arg: str) -> list[int]:
    rows = []

    for part in rows_arg.split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            start, end = part.split("-", 1)
            rows.extend(range(int(start), int(end) + 1))
        else:
            rows.append(int(part))

    return rows


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
        cmd += ["--tfr-type", tfr_type]

    preprocessing = row["preprocessing"]
    if preprocessing != "none":
        cmd += ["--preprocessing", preprocessing]

    cmd += [
        "--benchmark-dataset-id",
        row["dataset_id"],
        "--benchmark-representation",
        row["representation"],
        "--benchmark-adaptation",
        row["adaptation"],
    ]

    return cmd


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        benchmark_config = yaml.safe_load(f)

    grid = build_stage_1_grid(benchmark_config)
    df = pd.DataFrame(grid)

    if args.dry_run:
        print(df)
        print(f"\nTotal stage-1 runs: {len(df)}")
        return

    if args.stage_1_all:
        selected_row_ids = list(range(len(grid)))
    elif args.rows is not None:
        selected_row_ids = parse_rows(args.rows)
    elif args.row is not None:
        selected_row_ids = [args.row]
    else:
        print(df)
        print(f"\nTotal stage-1 runs: {len(df)}")
        return

    for row_id in selected_row_ids:
        if row_id < 0 or row_id >= len(grid):
            raise IndexError(f"Row index out of range: {row_id}. Grid size: {len(grid)}")

        row = grid[row_id]

        print("\n" + "=" * 80)
        print(f"Running benchmark row: {row_id}")
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
            continue

        print("\nExecuting:")
        print(" ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            if args.continue_on_error:
                print(f"FAILED row {row_id}: {exc}")
                continue
            raise

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