from __future__ import annotations

import argparse

from cross_domain_pipeline.api import prepare_windows, train_source_only
from cross_domain_pipeline.runner import BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cdp",
        description="Cross-domain anomaly detection pipeline",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-windows",
        help="Build window-level manifest from record-level CSV.",
    )
    prepare_parser.add_argument("--config", required=True)

    train_parser = subparsers.add_parser(
        "train-source-only",
        help="Train current source-only baseline.",
    )
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument(
        "--variant",
        required=True,
        choices=["raw_only", "tfr_only", "fused"],
    )

    baseline_parser = subparsers.add_parser(
        "train-source-only-baselines",
        help="Train raw_only, tfr_only, and fused source-only baselines.",
    )
    baseline_parser.add_argument("--config", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-windows":
        manifest = prepare_windows(args.config)
        print("Processed manifest created")
        print(f"Total windows: {len(manifest)}")
        print(manifest.head())
        return

    if args.command == "train-source-only":
        train_source_only(args.config, variant=args.variant)
        return

    if args.command == "train-source-only-baselines":
        runner = BenchmarkRunner.from_config(args.config)
        runner.train_source_only_baselines()
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()