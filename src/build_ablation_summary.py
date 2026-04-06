from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


DAY5_CSV = Path("experiments/day5_evaluation/day5_summary.csv")
DAY6_MINIMAL_JSON = Path("experiments/ablation_inputs/day6_minimal_sfda_summary.json")
DAY7_CONSISTENCY_JSON = Path("experiments/ablation_inputs/day7_consistency_summary.json")

OUT_DIR = Path("experiments/day8_ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_day5_rows():
    if not DAY5_CSV.exists():
        raise FileNotFoundError(f"Missing file: {DAY5_CSV}")

    df = pd.read_csv(DAY5_CSV)

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "experiment": row["variant"],
                "stage": "source_only",
                "source_val_roc_auc": row["source_val_roc_auc"],
                "target_test_roc_auc": row["target_test_roc_auc"],
                "source_val_pr_auc": row["source_val_pr_auc"],
                "target_test_pr_auc": row["target_test_pr_auc"],
                "source_val_f1": row["source_val_f1"],
                "target_test_f1": row["target_test_f1"],
                "delta_roc_auc": row["delta_roc_auc"],
                "delta_pr_auc": row["delta_pr_auc"],
            }
        )
    return rows


def load_sfda_json(path: Path, experiment_name: str, stage_name: str):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    row = {
        "experiment": experiment_name,
        "stage": stage_name,
        "source_val_roc_auc": data["after"]["source_val"]["roc_auc"],
        "target_test_roc_auc": data["after"]["target_test"]["roc_auc"],
        "source_val_pr_auc": data["after"]["source_val"]["pr_auc"],
        "target_test_pr_auc": data["after"]["target_test"]["pr_auc"],
        "source_val_f1": data["after"]["source_val"]["f1"],
        "target_test_f1": data["after"]["target_test"]["f1"],
        "delta_roc_auc": data["after"]["source_val"]["roc_auc"] - data["after"]["target_test"]["roc_auc"],
        "delta_pr_auc": data["after"]["source_val"]["pr_auc"] - data["after"]["target_test"]["pr_auc"],
    }
    return row


def plot_target_metrics(df: pd.DataFrame):
    plt.figure(figsize=(9, 5))
    plt.bar(df["experiment"] + " | " + df["stage"], df["target_test_roc_auc"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Target ROC-AUC")
    plt.title("Ablation summary: target ROC-AUC")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "target_roc_auc_bar.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(df["experiment"] + " | " + df["stage"], df["target_test_pr_auc"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Target PR-AUC")
    plt.title("Ablation summary: target PR-AUC")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "target_pr_auc_bar.png", dpi=150)
    plt.close()


def plot_source_vs_target(df: pd.DataFrame):
    plt.figure(figsize=(7, 6))
    for _, row in df.iterrows():
        label = f'{row["experiment"]} | {row["stage"]}'
        plt.scatter(row["source_val_roc_auc"], row["target_test_roc_auc"])
        plt.text(row["source_val_roc_auc"], row["target_test_roc_auc"], label, fontsize=8)

    plt.xlabel("Source ROC-AUC")
    plt.ylabel("Target ROC-AUC")
    plt.title("Source vs Target ROC-AUC")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "source_vs_target_roc_auc.png", dpi=150)
    plt.close()


def plot_fused_before_after():
    # prefers Day7 consistency json if present
    if DAY7_CONSISTENCY_JSON.exists():
        with open(DAY7_CONSISTENCY_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        labels = ["before", "after"]
        target_roc = [
            data["before"]["target_test"]["roc_auc"],
            data["after"]["target_test"]["roc_auc"],
        ]
        target_pr = [
            data["before"]["target_test"]["pr_auc"],
            data["after"]["target_test"]["pr_auc"],
        ]

        plt.figure(figsize=(6, 4))
        plt.plot(labels, target_roc, marker="o")
        plt.ylabel("Target ROC-AUC")
        plt.title("Fused: before vs after adaptation")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "fused_before_after_target_roc.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(labels, target_pr, marker="o")
        plt.ylabel("Target PR-AUC")
        plt.title("Fused: before vs after adaptation")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "fused_before_after_target_pr.png", dpi=150)
        plt.close()


def main():
    rows = load_day5_rows()

    day6_row = load_sfda_json(
        DAY6_MINIMAL_JSON,
        experiment_name="fused",
        stage_name="sfda_minimal",
    )
    if day6_row is not None:
        rows.append(day6_row)

    day7_row = load_sfda_json(
        DAY7_CONSISTENCY_JSON,
        experiment_name="fused",
        stage_name="sfda_consistency",
    )
    if day7_row is not None:
        rows.append(day7_row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_DIR / "ablation_summary.csv", index=False)

    plot_target_metrics(summary_df)
    plot_source_vs_target(summary_df)
    plot_fused_before_after()

    print("Saved:", OUT_DIR / "ablation_summary.csv")
    print(summary_df)


if __name__ == "__main__":
    main()