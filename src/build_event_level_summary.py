from pathlib import Path
import json
import pandas as pd

OUT_DIR = Path("experiments/ablation_summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_ROOT = Path("experiments/event_level")


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_event_summary_path(summary_path: Path, event_root: Path):
    """
    Supports both layouts:

    Old:
      experiments/event_level/<match_tag>/<stage>/<variant>/summary.json

    New:
      experiments/event_level/<match_tag>/<postproc_tag>/<stage>/<variant>/summary.json
    """
    rel_parts = summary_path.relative_to(event_root).parts

    # Expected endings:
    # old -> [match_tag, stage, variant, "summary.json"]
    # new -> [match_tag, postproc_tag, stage, variant, "summary.json"]

    if len(rel_parts) == 4 and rel_parts[-1] == "summary.json":
        match_tag, stage, variant, _ = rel_parts
        postproc_tag = "default"
        return match_tag, postproc_tag, stage, variant

    if len(rel_parts) == 5 and rel_parts[-1] == "summary.json":
        match_tag, postproc_tag, stage, variant, _ = rel_parts
        return match_tag, postproc_tag, stage, variant

    return None

def collect_event_rows(event_root: Path):
    if not event_root.exists():
        raise FileNotFoundError(f"Missing directory: {event_root}")

    rows = []

    for summary_path in sorted(event_root.rglob("summary.json")):
        parsed = parse_event_summary_path(summary_path, event_root)
        if parsed is None:
            continue

        match_tag_from_path, postproc_tag_from_path, stage_from_path, variant_from_path = parsed

        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        match_tag = data.get("match_tag", match_tag_from_path)
        postproc_tag = data.get("postproc_tag", postproc_tag_from_path)
        stage = data.get("stage", stage_from_path)
        variant = data.get("variant", variant_from_path)

        min_iou = data.get("min_iou")
        if min_iou is None:
            try:
                min_iou = float(match_tag.replace("iou_", "").replace("p", "."))
            except ValueError:
                min_iou = None

        rows.append(
            {
                "match_tag": match_tag,
                "min_iou": min_iou,
                "postproc_tag": postproc_tag,
                "score_smoothing_windows": data.get("score_smoothing_windows", 1),
                "hysteresis_low_ratio": data.get("hysteresis_low_ratio", 1.0),
                "max_gap_samples": data.get("max_gap_samples", 0),
                "min_event_length_samples": data.get("min_event_length_samples", 1),
                "experiment": variant,
                "stage": stage,
                "n_records": data.get("n_records"),
                "n_gt_events": data.get("n_gt_events"),
                "n_pred_events": data.get("n_pred_events"),
                "n_matched_events": data.get(
                    "n_matched_events",
                    data.get("n_matched_gt_events"),
                ),
                "n_missed_events": data.get("n_missed_events"),
                "n_false_alarm_events": data.get("n_false_alarm_events"),
                "event_precision": data.get("event_precision"),
                "event_recall": data.get("event_recall"),
                "event_f1": data.get("event_f1"),
                "false_alarms_per_record": data.get("false_alarms_per_record"),
                "false_alarms_per_normal_record": data.get("false_alarms_per_normal_record"),
                "mean_detection_delay_samples": data.get("mean_detection_delay_samples"),
                "median_detection_delay_samples": data.get("median_detection_delay_samples"),
                "mean_matched_iou": data.get("mean_matched_iou"),
                "mean_gt_coverage": data.get("mean_gt_coverage"),
                "mean_pred_coverage": data.get("mean_pred_coverage"),
                "summary_path": str(summary_path),
            }
        )

    if not rows:
        raise RuntimeError(f"No event-level summaries found under: {event_root}")

    return pd.DataFrame(rows)


def build_compact_view(df: pd.DataFrame):
    metric_cols = [
        "event_precision",
        "event_recall",
        "event_f1",
        "false_alarms_per_record",
        "false_alarms_per_normal_record",
        "mean_detection_delay_samples",
        "mean_matched_iou",
    ]

    index_cols = ["experiment", "stage"]
    if "postproc_tag" in df.columns:
        index_cols.append("postproc_tag")

    compact = (
        df.set_index(index_cols + ["match_tag"])[metric_cols]
        .unstack("match_tag")
        .sort_index()
    )

    compact.columns = [
        f"{match_tag}__{metric}" for metric, match_tag in compact.columns
    ]
    compact = compact.reset_index()
    return compact


def main():
    df = collect_event_rows(EVENT_ROOT)
    sort_cols = ["min_iou", "experiment", "stage"]
    if "postproc_tag" in df.columns:
        sort_cols.append("postproc_tag")

    df = df.sort_values(sort_cols).reset_index(drop=True)

    full_csv = OUT_DIR / "event_level_summary.csv"
    compact_csv = OUT_DIR / "event_level_summary_compact.csv"

    df.to_csv(full_csv, index=False)

    compact_df = build_compact_view(df)
    compact_df.to_csv(compact_csv, index=False)

    print("Saved:", full_csv)
    print("Saved:", compact_csv)
    print(df)


if __name__ == "__main__":
    main()