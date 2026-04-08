import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


EXPERIMENTS_ROOT = Path("experiments")
REPORTS_DIR = EXPERIMENTS_ROOT / "_reports"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def infer_dataset(summary_path: Path) -> str:
    parts = summary_path.parts
    try:
        idx = parts.index("experiments")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return "unknown"


def infer_stage(summary_path: Path) -> Optional[str]:
    parts = set(summary_path.parts)
    if "source_only_training" in parts:
        return "source_only"
    if "source_free_adaptation" in parts:
        return "sfda"
    return None


def parse_source_only_summary(summary_path: Path) -> Dict[str, Any]:
    data = read_json(summary_path)

    row = {
        "dataset": infer_dataset(summary_path),
        "stage": "source_only",
        "variant": data.get("variant"),
        "experiment_name": data.get("experiment_name"),
        "run_name": data.get("run_name"),
        "summary_path": str(summary_path).replace("\\", "/"),
        "threshold_config": data.get("threshold_config"),
        "threshold_used": data.get("threshold_used"),
        "best_threshold_source_val_f1": data.get("best_threshold_source_val_f1"),
        "source_val_roc_auc": safe_get(data, "source_val", "roc_auc"),
        "source_val_pr_auc": safe_get(data, "source_val", "pr_auc"),
        "source_val_f1": safe_get(data, "source_val", "f1"),
        "source_val_loss": safe_get(data, "source_val", "loss"),
        "source_val_tn": safe_get(data, "source_val", "tn"),
        "source_val_fp": safe_get(data, "source_val", "fp"),
        "source_val_fn": safe_get(data, "source_val", "fn"),
        "source_val_tp": safe_get(data, "source_val", "tp"),
        "target_test_roc_auc": safe_get(data, "target_test", "roc_auc"),
        "target_test_pr_auc": safe_get(data, "target_test", "pr_auc"),
        "target_test_f1": safe_get(data, "target_test", "f1"),
        "target_test_loss": safe_get(data, "target_test", "loss"),
        "target_test_tn": safe_get(data, "target_test", "tn"),
        "target_test_fp": safe_get(data, "target_test", "fp"),
        "target_test_fn": safe_get(data, "target_test", "fn"),
        "target_test_tp": safe_get(data, "target_test", "tp"),
    }
    return row


def parse_sfda_summary(summary_path: Path) -> Dict[str, Any]:
    data = read_json(summary_path)

    row = {
        "dataset": infer_dataset(summary_path),
        "stage": "sfda",
        "variant": data.get("variant"),
        "experiment_name": data.get("experiment_name"),
        "run_name": data.get("run_name"),
        "summary_path": str(summary_path).replace("\\", "/"),
        "threshold_used_source_val": data.get("threshold_used_source_val"),
        "source_anomaly_rate": data.get("source_anomaly_rate"),
        "target_threshold_before": data.get("target_threshold_before"),
        "target_threshold_after": data.get("target_threshold_after"),
        "target_threshold_before_mode": safe_get(data, "target_threshold_before_info", "mode"),
        "target_threshold_after_mode": safe_get(data, "target_threshold_after_info", "mode"),
        "before_source_val_roc_auc": safe_get(data, "before", "source_val", "roc_auc"),
        "before_source_val_pr_auc": safe_get(data, "before", "source_val", "pr_auc"),
        "before_source_val_f1": safe_get(data, "before", "source_val", "f1"),
        "after_source_val_roc_auc": safe_get(data, "after", "source_val", "roc_auc"),
        "after_source_val_pr_auc": safe_get(data, "after", "source_val", "pr_auc"),
        "after_source_val_f1": safe_get(data, "after", "source_val", "f1"),
        "before_target_test_source_threshold_roc_auc": safe_get(
            data, "before", "target_test_source_threshold", "roc_auc"
        ),
        "before_target_test_source_threshold_pr_auc": safe_get(
            data, "before", "target_test_source_threshold", "pr_auc"
        ),
        "before_target_test_source_threshold_f1": safe_get(
            data, "before", "target_test_source_threshold", "f1"
        ),
        "after_target_test_source_threshold_roc_auc": safe_get(
            data, "after", "target_test_source_threshold", "roc_auc"
        ),
        "after_target_test_source_threshold_pr_auc": safe_get(
            data, "after", "target_test_source_threshold", "pr_auc"
        ),
        "after_target_test_source_threshold_f1": safe_get(
            data, "after", "target_test_source_threshold", "f1"
        ),
        "before_target_test_target_calibrated_roc_auc": safe_get(
            data, "before", "target_test_target_calibrated", "roc_auc"
        ),
        "before_target_test_target_calibrated_pr_auc": safe_get(
            data, "before", "target_test_target_calibrated", "pr_auc"
        ),
        "before_target_test_target_calibrated_f1": safe_get(
            data, "before", "target_test_target_calibrated", "f1"
        ),
        "after_target_test_target_calibrated_roc_auc": safe_get(
            data, "after", "target_test_target_calibrated", "roc_auc"
        ),
        "after_target_test_target_calibrated_pr_auc": safe_get(
            data, "after", "target_test_target_calibrated", "pr_auc"
        ),
        "after_target_test_target_calibrated_f1": safe_get(
            data, "after", "target_test_target_calibrated", "f1"
        ),
        "delta_source_val_roc_auc": safe_get(data, "delta", "source_val_roc_auc"),
        "delta_source_val_pr_auc": safe_get(data, "delta", "source_val_pr_auc"),
        "delta_source_val_f1": safe_get(data, "delta", "source_val_f1"),
        "delta_target_test_source_threshold_roc_auc": safe_get(
            data, "delta", "target_test_source_threshold_roc_auc"
        ),
        "delta_target_test_source_threshold_pr_auc": safe_get(
            data, "delta", "target_test_source_threshold_pr_auc"
        ),
        "delta_target_test_source_threshold_f1": safe_get(
            data, "delta", "target_test_source_threshold_f1"
        ),
        "delta_target_test_target_calibrated_roc_auc": safe_get(
            data, "delta", "target_test_target_calibrated_roc_auc"
        ),
        "delta_target_test_target_calibrated_pr_auc": safe_get(
            data, "delta", "target_test_target_calibrated_pr_auc"
        ),
        "delta_target_test_target_calibrated_f1": safe_get(
            data, "delta", "target_test_target_calibrated_f1"
        ),
    }
    return row


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows found._\n"

    cols = list(df.columns)
    rows: List[List[str]] = [[format_value(v) for v in row] for row in df.to_numpy().tolist()]
    headers = [str(c) for c in cols]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: List[str]) -> str:
        cells = [cell.ljust(widths[i]) for i, cell in enumerate(row)]
        return "| " + " | ".join(cells) + " |"

    lines = [
        fmt_row(headers),
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines) + "\n"


def collect_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    source_only_rows: List[Dict[str, Any]] = []
    sfda_rows: List[Dict[str, Any]] = []

    for summary_path in sorted(EXPERIMENTS_ROOT.rglob("summary.json")):
        if REPORTS_DIR in summary_path.parents:
            continue

        stage = infer_stage(summary_path)
        if stage == "source_only":
            try:
                source_only_rows.append(parse_source_only_summary(summary_path))
            except Exception as exc:
                print(f"[WARN] Failed to parse source-only summary: {summary_path} :: {exc}")
        elif stage == "sfda":
            try:
                sfda_rows.append(parse_sfda_summary(summary_path))
            except Exception as exc:
                print(f"[WARN] Failed to parse SFDA summary: {summary_path} :: {exc}")

    source_only_df = pd.DataFrame(source_only_rows)
    sfda_df = pd.DataFrame(sfda_rows)

    if not source_only_df.empty:
        source_only_df = source_only_df.sort_values(
            ["dataset", "experiment_name", "run_name", "variant"]
        ).reset_index(drop=True)

    if not sfda_df.empty:
        sfda_df = sfda_df.sort_values(
            ["dataset", "experiment_name", "run_name", "variant"]
        ).reset_index(drop=True)

    return source_only_df, sfda_df


def write_outputs(source_only_df: pd.DataFrame, sfda_df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    source_only_csv = REPORTS_DIR / "source_only_comparison.csv"
    source_only_md = REPORTS_DIR / "source_only_comparison.md"
    sfda_csv = REPORTS_DIR / "sfda_comparison.csv"
    sfda_md = REPORTS_DIR / "sfda_comparison.md"

    source_only_df.to_csv(source_only_csv, index=False)
    sfda_df.to_csv(sfda_csv, index=False)

    source_only_md.write_text(dataframe_to_markdown(source_only_df), encoding="utf-8")
    sfda_md.write_text(dataframe_to_markdown(sfda_df), encoding="utf-8")

    print(f"Saved: {source_only_csv}")
    print(f"Saved: {source_only_md}")
    print(f"Saved: {sfda_csv}")
    print(f"Saved: {sfda_md}")


def main():
    source_only_df, sfda_df = collect_summaries()
    write_outputs(source_only_df, sfda_df)

    print()
    print(f"Source-only rows: {len(source_only_df)}")
    print(f"SFDA rows: {len(sfda_df)}")


if __name__ == "__main__":
    main()