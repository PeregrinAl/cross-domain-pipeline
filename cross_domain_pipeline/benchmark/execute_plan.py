from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from cross_domain_pipeline.classical.train_classical import run_classical_baseline
from cross_domain_pipeline.config import PipelineConfig
from cross_domain_pipeline.model_selection.benchmark_plan import build_benchmark_plan
from cross_domain_pipeline.model_selection.data_profile import profile_from_config


CLASSICAL_METHODS = {
    "statistical_threshold",
    "pca_reconstruction",
    "isolation_forest",
}


def execute_benchmark_plan(
    config_path: str | Path,
    plan_path: str | Path | None = None,
    output_dir: str | Path = "experiments/benchmark_execution",
    source: str = "auto",
    mode: str = "compact",
    max_files: int = 200,
    max_candidates: int = 12,
    dry_run: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if plan_path is None:
        cfg = PipelineConfig.from_yaml(config_path)
        profile = profile_from_config(
            config=cfg,
            source=source,
            max_files=max_files,
        )
        plan = build_benchmark_plan(
            profile=profile,
            mode=mode,
            max_candidates=max_candidates,
        )
        plan_dict = plan.to_dict()
        (output_root / "benchmark_plan.json").write_text(
            plan.to_json(),
            encoding="utf-8",
        )
    else:
        plan_path = Path(plan_path)
        plan_dict = json.loads(plan_path.read_text(encoding="utf-8"))

    entries = []

    for idx, candidate in enumerate(plan_dict.get("candidates", []), start=1):
        model = candidate["model"]
        representation = candidate["representation"]
        adaptation = candidate["adaptation"]

        run_name = f"{idx:03d}_{model}_{representation}_{adaptation}"
        run_dir = output_root / run_name

        entry = {
            "idx": idx,
            "model": model,
            "family": candidate.get("family", ""),
            "tier": candidate.get("tier", ""),
            "representation": representation,
            "adaptation": adaptation,
            "run_dir": str(run_dir),
            "status": "pending",
            "roc_auc": None,
            "pr_auc": None,
            "best_f1": None,
            "threshold": None,
            "error": None,
        }

        if dry_run:
            entry["status"] = "planned"
            entries.append(entry)
            continue

        if model not in CLASSICAL_METHODS:
            entry["status"] = "not_implemented"
            entry["error"] = "Only classical baselines are executable at this step."
            entries.append(entry)
            continue

        try:
            summary = run_classical_baseline(
                config_path=config_path,
                method=model,
                source=source,
                output_dir=run_dir,
            )

            entry["status"] = "completed"
            entry["roc_auc"] = summary.get("roc_auc")
            entry["pr_auc"] = summary.get("pr_auc")
            entry["best_f1"] = summary.get("best_f1")
            entry["threshold"] = summary.get("threshold")

        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)

        entries.append(entry)

    result = {
        "config_path": str(config_path),
        "output_dir": str(output_root),
        "dry_run": dry_run,
        "n_candidates": len(entries),
        "n_completed": sum(1 for x in entries if x["status"] == "completed"),
        "n_not_implemented": sum(1 for x in entries if x["status"] == "not_implemented"),
        "n_failed": sum(1 for x in entries if x["status"] == "failed"),
        "entries": entries,
    }

    (output_root / "execution_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pd.DataFrame(entries).to_csv(
        output_root / "execution_summary.csv",
        index=False,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute an evidence-constrained benchmark plan.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", default=None)
    parser.add_argument("--output-dir", default="experiments/benchmark_execution")
    parser.add_argument(
        "--source",
        default="auto",
        choices=["auto", "raw", "manifest"],
    )
    parser.add_argument(
        "--mode",
        default="compact",
        choices=["compact", "extended"],
    )
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    execute_benchmark_plan(
        config_path=args.config,
        plan_path=args.plan,
        output_dir=args.output_dir,
        source=args.source,
        mode=args.mode,
        max_files=args.max_files,
        max_candidates=args.max_candidates,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()