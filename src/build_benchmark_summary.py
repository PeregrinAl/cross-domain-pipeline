import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=str, default="experiments")
    parser.add_argument("--out", type=str, default="reports/benchmark/stage1_summary.csv")
    parser.add_argument("--top-k", type=int, default=2)
    return parser.parse_args()


def metric(summary: dict, split: str, name: str):
    return summary.get(split, {}).get(name)


def summary_to_row(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    return {
        "dataset_id": summary.get("dataset_id", "unknown"),
        "preprocessing": summary.get("preprocessing", "none"),
        "representation": summary.get("representation", summary.get("variant")),
        "adaptation": summary.get("adaptation", "source_only"),
        "variant": summary.get("variant"),
        "variant_run_name": summary.get("variant_run_name", summary.get("variant")),
        "tfr_type": summary.get("tfr_type", "stft"),
        "source_val_roc_auc": metric(summary, "source_val", "roc_auc"),
        "source_val_pr_auc": metric(summary, "source_val", "pr_auc"),
        "source_val_f1": metric(summary, "source_val", "f1"),
        "target_test_roc_auc": metric(summary, "target_test", "roc_auc"),
        "target_test_pr_auc": metric(summary, "target_test", "pr_auc"),
        "target_test_f1": metric(summary, "target_test", "f1"),
        "threshold_used": summary.get("threshold_used"),
        "summary_path": str(path),
    }


def main():
    args = parse_args()

    root = Path(args.experiments_root)
    summaries = sorted(root.rglob("source_only_training/**/summary.json"))

    rows = [summary_to_row(path) for path in summaries]
    df = pd.DataFrame(rows)
    df = df[df["dataset_id"] != "unknown"].copy()
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)
    print(f"Saved summary: {out_path}")
    print(f"Rows: {len(df)}")

    if df.empty:
        return

    sort_cols = ["target_test_pr_auc", "target_test_f1"]
    top_rows = []

    for dataset_id, group in df.groupby("dataset_id", dropna=False):
        group_sorted = group.sort_values(
            by=sort_cols,
            ascending=False,
            na_position="last",
        )
        top_rows.append(group_sorted.head(args.top_k))

    top_df = pd.concat(top_rows, ignore_index=True)
    top_path = out_path.with_name("stage1_top2.csv")
    top_df.to_csv(top_path, index=False)

    print(f"Saved top-{args.top_k}: {top_path}")


if __name__ == "__main__":
    main()