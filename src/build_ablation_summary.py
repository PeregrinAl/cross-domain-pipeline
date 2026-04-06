from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


OUT_DIR = Path("experiments/ablation_summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_ONLY_DIR_CANDIDATES = [
    Path("experiments/source_only_training"),
    Path("experiments/source_only_supervised"),
]

SFDA_DIR_CANDIDATES = [
    Path("experiments/source_free_adaptation"),
    Path("experiments/day6_sfda"),
]


def resolve_existing_dir(candidates):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of the candidate directories exist:\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_source_only_row(source_only_dir: Path, variant: str):
    summary_path = source_only_dir / variant / "summary.json"
    data = load_json(summary_path)

    return {
        "experiment": variant,
        "stage": "source_only",
        "source_val_roc_auc": data["source_val"]["roc_auc"],
        "target_test_roc_auc": data["target_test"]["roc_auc"],
        "source_val_pr_auc": data["source_val"]["pr_auc"],
        "target_test_pr_auc": data["target_test"]["pr_auc"],
        "source_val_f1": data["source_val"]["f1"],
        "target_test_f1": data["target_test"]["f1"],
        "delta_roc_auc": data["source_val"]["roc_auc"] - data["target_test"]["roc_auc"],
        "delta_pr_auc": data["source_val"]["pr_auc"] - data["target_test"]["pr_auc"],
    }


def build_sfda_row(sfda_dir: Path, variant: str = "fused", stage_name: str = "sfda_minimal"):
    summary_path = sfda_dir / variant / "summary.json"
    data = load_json(summary_path)

    after = data["after"]
    target = after["target_test_target_calibrated"]
    source = after["source_val"]

    return {
        "experiment": variant,
        "stage": stage_name,
        "source_val_roc_auc": source["roc_auc"],
        "target_test_roc_auc": target["roc_auc"],
        "source_val_pr_auc": source["pr_auc"],
        "target_test_pr_auc": target["pr_auc"],
        "source_val_f1": source["f1"],
        "target_test_f1": target["f1"],
        "delta_roc_auc": source["roc_auc"] - target["roc_auc"],
        "delta_pr_auc": source["pr_auc"] - target["pr_auc"],
    }


def plot_target_metrics(df: pd.DataFrame):
    labels = df["experiment"] + " | " + df["stage"]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, df["target_test_roc_auc"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Target ROC-AUC")
    plt.title("Ablation summary: target ROC-AUC")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "target_roc_auc_bar.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(labels, df["target_test_pr_auc"])
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
        plt.text(
            row["source_val_roc_auc"],
            row["target_test_roc_auc"],
            label,
            fontsize=8,
        )

    plt.xlabel("Source ROC-AUC")
    plt.ylabel("Target ROC-AUC")
    plt.title("Source vs Target ROC-AUC")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "source_vs_target_roc_auc.png", dpi=150)
    plt.close()


def plot_fused_before_after(sfda_dir: Path):
    summary_path = sfda_dir / "fused" / "summary.json"
    if not summary_path.exists():
        return

    data = load_json(summary_path)

    labels = ["before", "after"]
    target_roc = [
        data["before"]["target_test_target_calibrated"]["roc_auc"],
        data["after"]["target_test_target_calibrated"]["roc_auc"],
    ]
    target_pr = [
        data["before"]["target_test_target_calibrated"]["pr_auc"],
        data["after"]["target_test_target_calibrated"]["pr_auc"],
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
    source_only_dir = resolve_existing_dir(SOURCE_ONLY_DIR_CANDIDATES)
    sfda_dir = resolve_existing_dir(SFDA_DIR_CANDIDATES)

    rows = [
        build_source_only_row(source_only_dir, "raw_only"),
        build_source_only_row(source_only_dir, "tfr_only"),
        build_source_only_row(source_only_dir, "fused"),
    ]

    try:
        rows.append(build_sfda_row(sfda_dir, variant="fused", stage_name="sfda_minimal"))
    except FileNotFoundError:
        pass

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_DIR / "ablation_summary.csv", index=False)

    plot_target_metrics(summary_df)
    plot_source_vs_target(summary_df)
    plot_fused_before_after(sfda_dir)

    print("Saved:", OUT_DIR / "ablation_summary.csv")
    print(summary_df)


if __name__ == "__main__":
    main()