#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# utils
# ----------------------------

def ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return path


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(ensure_exists(path))


def has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def col_lower_contains(df: pd.DataFrame, col: str, needle: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([True] * len(df), index=df.index)
    return df[col].astype(str).str.lower().str.contains(needle.lower(), na=False)


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fmt(x):
    if pd.isna(x):
        return ""
    if isinstance(x, (int, float, np.floating)):
        return f"{float(x):.4f}"
    return str(x)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_placeholder_png(out_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.35, message, ha="center", va="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# extraction
# ----------------------------

def extract_synthetic(repo_root: Path) -> dict[str, pd.DataFrame]:
    out = {}

    ablation_path = repo_root / "experiments" / "ablation_summary" / "ablation_summary.csv"
    event_path = repo_root / "experiments" / "ablation_summary" / "event_level_summary_compact.csv"

    ablation = load_csv(ablation_path)
    event = load_csv(event_path)

    exp_col = pick_col(ablation, ["experiment", "variant"])
    stage_col = pick_col(ablation, ["stage"])
    roc_col = pick_col(ablation, ["target_test_roc_auc", "target_roc_auc"])
    pr_col = pick_col(ablation, ["target_test_pr_auc", "target_pr_auc"])
    f1_col = pick_col(ablation, ["target_test_f1", "target_f1"])

    # source-only comparison
    syn_src = ablation.copy()
    if stage_col:
        syn_src = syn_src[col_lower_contains(syn_src, stage_col, "source_only")]
    if exp_col:
        syn_src = syn_src[syn_src[exp_col].astype(str).isin(["raw_only", "tfr_only", "fused"])]

    syn_src_table = pd.DataFrame()
    if len(syn_src) and exp_col and roc_col and pr_col and f1_col:
        syn_src_table = syn_src[[exp_col, roc_col, pr_col, f1_col]].copy()
        syn_src_table.columns = ["variant", "target_roc_auc", "target_pr_auc", "target_f1"]

    out["synthetic_source_only"] = syn_src_table

    # fused sfda
    syn_sfda = ablation.copy()
    if exp_col:
        syn_sfda = syn_sfda[col_lower_contains(syn_sfda, exp_col, "fused")]
    if stage_col:
        syn_sfda = syn_sfda[col_lower_contains(syn_sfda, stage_col, "sfda")]

    syn_sfda_table = pd.DataFrame()
    if len(syn_sfda) and roc_col and pr_col and f1_col:
        # обычно здесь одна строка
        syn_sfda_table = syn_sfda[[roc_col, pr_col, f1_col]].copy()
        syn_sfda_table.insert(0, "variant", "fused_after_sfda")
        syn_sfda_table.columns = ["variant", "target_roc_auc", "target_pr_auc", "target_f1"]

    out["synthetic_fused_sfda"] = syn_sfda_table

    # event level
    event_variant_col = pick_col(event, ["stage", "variant"])
    event_postproc_col = pick_col(event, ["postproc_tag"])
    event_exp_col = pick_col(event, ["experiment"])
    event_f1_col = pick_col(event, ["iou_0p050__event_f1", "event_f1_iou_0p05"])
    event_fa_col = pick_col(
        event,
        ["iou_0p050__false_alarms_per_record", "false_alarms_per_record_iou_0p05"],
    )

    syn_event = event.copy()
    if event_postproc_col:
        syn_event = syn_event[col_lower_contains(syn_event, event_postproc_col, "default")]
    if event_exp_col:
        syn_event = syn_event[col_lower_contains(syn_event, event_exp_col, "fused")]

    syn_event_table = pd.DataFrame()
    if len(syn_event) and event_variant_col and event_f1_col and event_fa_col:
        syn_event_table = syn_event[[event_variant_col, event_f1_col, event_fa_col]].copy()
        syn_event_table.columns = ["variant", "event_f1_iou_0p05", "false_alarms_per_record"]

        stage_map = {
            "source_only": "source_only_fused",
            "sfda_before": "sfda_before_fused",
            "sfda_after": "sfda_after_fused",
        }
        syn_event_table["variant"] = syn_event_table["variant"].astype(str).map(lambda x: stage_map.get(x, x))

    out["synthetic_event_level"] = syn_event_table

    return out


def extract_source_only_block(df: pd.DataFrame, dataset_key: str, run_hint: str | None = None) -> pd.DataFrame:
    dataset_col = pick_col(df, ["dataset"])
    run_col = pick_col(df, ["run_name"])
    stage_col = pick_col(df, ["stage"])
    variant_col = pick_col(df, ["variant"])
    roc_col = pick_col(df, ["target_test_roc_auc", "target_roc_auc"])
    pr_col = pick_col(df, ["target_test_pr_auc", "target_pr_auc"])
    f1_col = pick_col(df, ["target_test_f1", "target_f1"])

    sub = df.copy()

    if dataset_col:
        sub = sub[col_lower_contains(sub, dataset_col, dataset_key)]

    # сначала пробуем run_hint, если есть
    if run_hint and run_col:
        sub_run = sub[col_lower_contains(sub, run_col, run_hint)]
        if len(sub_run):
            sub = sub_run

    if stage_col:
        sub = sub[col_lower_contains(sub, stage_col, "source_only")]

    if variant_col:
        sub = sub[sub[variant_col].astype(str).isin(["raw_only", "tfr_only", "fused"])]

    if not len(sub) or not all([variant_col, roc_col, pr_col, f1_col]):
        return pd.DataFrame()

    out = sub[[variant_col, roc_col, pr_col, f1_col]].copy()
    out.columns = ["variant", "target_roc_auc", "target_pr_auc", "target_f1"]
    return out


def extract_sfda_block(df: pd.DataFrame, dataset_key: str, run_hint: str | None = None) -> pd.DataFrame:
    dataset_col = pick_col(df, ["dataset"])
    run_col = pick_col(df, ["run_name"])
    variant_col = pick_col(df, ["variant"])

    sub = df.copy()

    if dataset_col:
        sub = sub[col_lower_contains(sub, dataset_col, dataset_key)]

    if run_hint and run_col:
        sub_run = sub[col_lower_contains(sub, run_col, run_hint)]
        if len(sub_run):
            sub = sub_run

    if variant_col:
        sub = sub[col_lower_contains(sub, variant_col, "fused")]

    before_roc = pick_col(df, ["before_target_test_source_threshold_roc_auc"])
    before_pr = pick_col(df, ["before_target_test_source_threshold_pr_auc"])
    before_f1 = pick_col(df, ["before_target_test_source_threshold_f1"])
    after_roc = pick_col(df, ["after_target_test_source_threshold_roc_auc"])
    after_pr = pick_col(df, ["after_target_test_source_threshold_pr_auc"])
    after_f1 = pick_col(df, ["after_target_test_source_threshold_f1"])
    before_tcal_f1 = pick_col(df, ["before_target_test_target_calibrated_f1"])
    after_tcal_f1 = pick_col(df, ["after_target_test_target_calibrated_f1"])

    needed = [before_roc, before_pr, before_f1, after_roc, after_pr, after_f1]
    if not len(sub) or any(c is None for c in needed):
        return pd.DataFrame()

    out_cols = [c for c in [
        variant_col,
        before_roc, before_pr, before_f1,
        after_roc, after_pr, after_f1,
        before_tcal_f1, after_tcal_f1,
    ] if c is not None]

    out = sub[out_cols].copy()

    rename_map = {}
    if variant_col: rename_map[variant_col] = "variant"
    if before_roc: rename_map[before_roc] = "before_roc_auc"
    if before_pr: rename_map[before_pr] = "before_pr_auc"
    if before_f1: rename_map[before_f1] = "before_f1"
    if after_roc: rename_map[after_roc] = "after_roc_auc"
    if after_pr: rename_map[after_pr] = "after_pr_auc"
    if after_f1: rename_map[after_f1] = "after_f1"
    if before_tcal_f1: rename_map[before_tcal_f1] = "before_target_cal_f1"
    if after_tcal_f1: rename_map[after_tcal_f1] = "after_target_cal_f1"

    out = out.rename(columns=rename_map)
    return out


def extract_real(repo_root: Path) -> dict[str, pd.DataFrame]:
    out = {}

    src_path = repo_root / "experiments" / "_reports" / "source_only_comparison.csv"
    sfda_path = repo_root / "experiments" / "_reports" / "sfda_comparison.csv"

    src = load_csv(src_path)
    sfda = load_csv(sfda_path)

    out["paderborn_source_only"] = extract_source_only_block(src, "pader", "repr_sweep")
    out["paderborn_fused_sfda"] = extract_sfda_block(sfda, "pader", "repr_sweep")

    out["mimii_source_only"] = extract_source_only_block(src, "mimii", "supervised")
    out["mimii_fused_sfda"] = extract_sfda_block(sfda, "mimii", "supervised")

    return out


# ----------------------------
# png tables
# ----------------------------

def save_table_png(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df is None or df.empty:
        save_placeholder_png(out_path, title, "No matching rows were found for this block.")
        return

    df_plot = df.copy()
    for c in df_plot.columns:
        df_plot[c] = df_plot[c].map(fmt)

    nrows, ncols = df_plot.shape
    fig_width = max(8, min(18, 1.8 * ncols + 1))
    fig_height = max(2.2, 0.58 * (nrows + 2))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df_plot.values,
        colLabels=df_plot.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
        cell.set_linewidth(0.8)

    plt.title(title, fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# charts
# ----------------------------

def save_source_only_chart(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df is None or df.empty:
        save_placeholder_png(out_path, title, "No matching rows were found for this chart.")
        return

    metrics = ["target_roc_auc", "target_pr_auc", "target_f1"]
    available = [m for m in metrics if m in df.columns]
    if not available or "variant" not in df.columns:
        save_placeholder_png(out_path, title, "Expected columns are missing.")
        return

    variants = df["variant"].astype(str).tolist()
    x = np.arange(len(variants))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, metric in enumerate(available):
        vals = df[metric].astype(float).to_numpy()
        ax.bar(x + (i - (len(available)-1)/2) * width, vals, width=width, label=metric.replace("target_", ""))

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_before_after_chart(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df is None or df.empty:
        save_placeholder_png(out_path, title, "No matching rows were found for this chart.")
        return

    row = df.iloc[0]

    pairs = [
        ("ROC AUC", "before_roc_auc", "after_roc_auc"),
        ("PR AUC", "before_pr_auc", "after_pr_auc"),
        ("F1", "before_f1", "after_f1"),
    ]

    # если есть target-calibrated, тоже покажем
    if "before_target_cal_f1" in df.columns and "after_target_cal_f1" in df.columns:
        pairs.append(("Target-cal F1", "before_target_cal_f1", "after_target_cal_f1"))

    labels = []
    before_vals = []
    after_vals = []

    for label, b, a in pairs:
        if b in df.columns and a in df.columns:
            labels.append(label)
            before_vals.append(float(row[b]))
            after_vals.append(float(row[a]))

    if not labels:
        save_placeholder_png(out_path, title, "Expected before/after columns are missing.")
        return

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, before_vals, width=width, label="before")
    ax.bar(x + width / 2, after_vals, width=width, label="after")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_event_chart(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df is None or df.empty:
        save_placeholder_png(out_path, title, "No matching rows were found for this chart.")
        return

    if "variant" not in df.columns or "event_f1_iou_0p05" not in df.columns:
        save_placeholder_png(out_path, title, "Expected event-level columns are missing.")
        return

    variants = df["variant"].astype(str).tolist()
    vals = df["event_f1_iou_0p05"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(np.arange(len(variants)), vals)
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels(variants, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("event F1")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_false_alarm_chart(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df is None or df.empty:
        save_placeholder_png(out_path, title, "No matching rows were found for this chart.")
        return

    if "variant" not in df.columns or "false_alarms_per_record" not in df.columns:
        save_placeholder_png(out_path, title, "Expected false-alarm columns are missing.")
        return

    variants = df["variant"].astype(str).tolist()
    vals = df["false_alarms_per_record"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(np.arange(len(variants)), vals)
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels(variants, rotation=0)
    ax.set_ylabel("false alarms / record")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build presentation PNGs from experiment summaries.")
    parser.add_argument("repo_root", nargs="?", default=".", help="Path to repository root")
    parser.add_argument("--out-dir", default="presentation_figures", help="Output directory for PNGs")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    safe_mkdir(out_dir)

    all_tables = {}
    all_tables.update(extract_synthetic(repo_root))
    all_tables.update(extract_real(repo_root))

    titles = {
        "synthetic_source_only": "Synthetic: source-only comparison",
        "synthetic_fused_sfda": "Synthetic: fused after SFDA",
        "synthetic_event_level": "Synthetic: event-level comparison",
        "paderborn_source_only": "Paderborn: source-only comparison",
        "paderborn_fused_sfda": "Paderborn: fused before/after SFDA",
        "mimii_source_only": "MIMII DUE: source-only comparison",
        "mimii_fused_sfda": "MIMII DUE: fused before/after SFDA",
    }

    # png tables
    for name, df in all_tables.items():
        save_table_png(df, out_dir / f"{name}_table.png", titles.get(name, name))

    # charts
    save_source_only_chart(
        all_tables.get("synthetic_source_only", pd.DataFrame()),
        out_dir / "synthetic_source_only_chart.png",
        "Synthetic: source-only comparison"
    )
    save_before_after_chart(
        all_tables.get("synthetic_fused_sfda", pd.DataFrame()),
        out_dir / "synthetic_fused_sfda_chart.png",
        "Synthetic: fused before/after SFDA"
    )
    save_event_chart(
        all_tables.get("synthetic_event_level", pd.DataFrame()),
        out_dir / "synthetic_event_f1_chart.png",
        "Synthetic: event-level F1"
    )
    save_false_alarm_chart(
        all_tables.get("synthetic_event_level", pd.DataFrame()),
        out_dir / "synthetic_false_alarms_chart.png",
        "Synthetic: false alarms per record"
    )

    save_source_only_chart(
        all_tables.get("paderborn_source_only", pd.DataFrame()),
        out_dir / "paderborn_source_only_chart.png",
        "Paderborn: source-only comparison"
    )
    save_before_after_chart(
        all_tables.get("paderborn_fused_sfda", pd.DataFrame()),
        out_dir / "paderborn_fused_sfda_chart.png",
        "Paderborn: fused before/after SFDA"
    )

    save_source_only_chart(
        all_tables.get("mimii_source_only", pd.DataFrame()),
        out_dir / "mimii_source_only_chart.png",
        "MIMII DUE: source-only comparison"
    )
    save_before_after_chart(
        all_tables.get("mimii_fused_sfda", pd.DataFrame()),
        out_dir / "mimii_fused_sfda_chart.png",
        "MIMII DUE: fused before/after SFDA"
    )

    print(f"Saved PNGs to: {out_dir}")
    print("Generated files:")
    for p in sorted(out_dir.glob("*.png")):
        print(f" - {p.name}")


if __name__ == "__main__":
    main()