import argparse
import yaml
import pandas as pd

from src.benchmark.grid import build_stage_1_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/benchmark_grid.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    grid = build_stage_1_grid(config)
    df = pd.DataFrame(grid)

    if args.dry_run:
        print(df)
        print(f"\nTotal stage-1 runs: {len(df)}")
        return

    raise NotImplementedError("Actual benchmark execution will be added next.")


if __name__ == "__main__":
    main()