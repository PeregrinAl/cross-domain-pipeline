from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    f1_score,
)


BASE_DIR = Path("experiments/source_only_supervised")
OUT_DIR = Path("experiments/day5_evaluation")
VARIANTS = ["raw_only", "tfr_only", "fused"]


def safe_roc_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def compute_metrics(y_true, y_score, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    roc_auc = safe_roc_auc(y_true, y_score)
    pr_auc = safe_pr_auc(y_true, y_score)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_f1_threshold(y_true, y_score, grid_size=1001):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    low = float(np.min(y_score))
    high = float(np.max(y_score))

    if np.isclose(low, high):
        threshold = low
        metrics = compute_metrics(y_true, y_score, threshold)
        return threshold, metrics["f1"]

    thresholds = np.linspace(low, high, grid_size)

    best_threshold = thresholds[0]
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, float(best_f1)


def load_scores(variant):
    source_path = BASE_DIR / variant / "source_val_scores.csv"
    target_path = BASE_DIR / variant / "target_test_scores.csv"

    if not source_path.exists():
        raise FileNotFoundError(f"Missing file: {source_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Missing file: {target_path}")

    source_df = pd.read_csv(source_path)
    target_df = pd.read_csv(target_path)

    return source_df, target_df


def save_score_distribution_plot(df, variant, split_name, out_path):
    normal_scores = df.loc[df["label"] == 0, "score"].values
    anomaly_scores = df.loc[df["label"] == 1, "score"].values

    plt.figure(figsize=(8, 5))
    plt.hist(normal_scores, bins=20, alpha=0.6, label="normal", density=True)
    plt.hist(anomaly_scores, bins=20, alpha=0.6, label="anomaly", density=True)
    plt.xlabel("Anomaly score")
    plt.ylabel("Density")
    plt.title(f"{variant} - {split_name} score distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_roc_curve_plot(source_df, target_df, variant, out_path):
    plt.figure(figsize=(6, 6))

    for df, name in [(source_df, "source_val"), (target_df, "target_test")]:
        y_true = df["label"].values
        y_score = df["score"].values

        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{variant} - ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_pr_curve_plot(source_df, target_df, variant, out_path):
    plt.figure(figsize=(6, 6))

    for df, name in [(source_df, "source_val"), (target_df, "target_test")]:
        y_true = df["label"].values
        y_score = df["score"].values

        if len(np.unique(y_true)) < 2:
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        plt.plot(recall, precision, label=f"{name} AP={ap:.3f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{variant} - Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def analyze_variant(variant):
    source_df, target_df = load_scores(variant)

    source_y_true = source_df["label"].values
    source_y_score = source_df["score"].values
    target_y_true = target_df["label"].values
    target_y_score = target_df["score"].values

    best_threshold, best_source_f1 = find_best_f1_threshold(source_y_true, source_y_score)

    source_metrics = compute_metrics(source_y_true, source_y_score, best_threshold)
    target_metrics = compute_metrics(target_y_true, target_y_score, best_threshold)

    variant_out_dir = OUT_DIR / variant
    variant_out_dir.mkdir(parents=True, exist_ok=True)

    save_score_distribution_plot(
        df=source_df,
        variant=variant,
        split_name="source_val",
        out_path=variant_out_dir / "source_val_score_distribution.png",
    )
    save_score_distribution_plot(
        df=target_df,
        variant=variant,
        split_name="target_test",
        out_path=variant_out_dir / "target_test_score_distribution.png",
    )
    save_roc_curve_plot(
        source_df=source_df,
        target_df=target_df,
        variant=variant,
        out_path=variant_out_dir / "roc_curve.png",
    )
    save_pr_curve_plot(
        source_df=source_df,
        target_df=target_df,
        variant=variant,
        out_path=variant_out_dir / "pr_curve.png",
    )

    summary = {
        "variant": variant,
        "best_threshold_source_val_f1": best_threshold,
        "best_source_val_f1": best_source_f1,
        "source_val": source_metrics,
        "target_test": target_metrics,
        "delta_roc_auc": (
            source_metrics["roc_auc"] - target_metrics["roc_auc"]
            if not (np.isnan(source_metrics["roc_auc"]) or np.isnan(target_metrics["roc_auc"]))
            else np.nan
        ),
        "delta_pr_auc": (
            source_metrics["pr_auc"] - target_metrics["pr_auc"]
            if not (np.isnan(source_metrics["pr_auc"]) or np.isnan(target_metrics["pr_auc"]))
            else np.nan
        ),
    }

    with open(variant_out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for variant in VARIANTS:
        summary = analyze_variant(variant)

        row = {
            "variant": variant,
            "threshold": summary["best_threshold_source_val_f1"],
            "source_val_roc_auc": summary["source_val"]["roc_auc"],
            "source_val_pr_auc": summary["source_val"]["pr_auc"],
            "source_val_f1": summary["source_val"]["f1"],
            "source_val_tn": summary["source_val"]["tn"],
            "source_val_fp": summary["source_val"]["fp"],
            "source_val_fn": summary["source_val"]["fn"],
            "source_val_tp": summary["source_val"]["tp"],
            "target_test_roc_auc": summary["target_test"]["roc_auc"],
            "target_test_pr_auc": summary["target_test"]["pr_auc"],
            "target_test_f1": summary["target_test"]["f1"],
            "target_test_tn": summary["target_test"]["tn"],
            "target_test_fp": summary["target_test"]["fp"],
            "target_test_fn": summary["target_test"]["fn"],
            "target_test_tp": summary["target_test"]["tp"],
            "delta_roc_auc": summary["delta_roc_auc"],
            "delta_pr_auc": summary["delta_pr_auc"],
        }
        all_rows.append(row)

        print(f"\nVariant: {variant}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(OUT_DIR / "day5_summary.csv", index=False)

    print("\nSaved day5 summary to:", OUT_DIR / "day5_summary.csv")
    print(summary_df)


if __name__ == "__main__":
    main()