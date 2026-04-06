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


def collect_event_rows(event_root: Path):
    if not event_root.exists():
        raise FileNotFoundError(f"Missing directory: {event_root}")

    rows = []
    for summary_path in sorted(event_root.glob("iou_*/*/*/summary.json")):
        data = load_json(summary_path)

        rows.append(
            {
                "match_tag": data["match_tag"],
                "min_iou": data["min_iou"],
                "experiment": data["variant"],
                "stage": data["stage"],
                "n_records": data["n_records"],
                "n_gt_events": data["n_gt_events"],
                "n_pred_events": data["n_pred_events"],
                "n_matched_events": data["n_matched_events"],
                "n_missed_events": data["n_missed_events"],
                "n_false_alarm_events": data["n_false_alarm_events"],
                "event_precision": data["event_precision"],
                "event_recall": data["event_recall"],
                "event_f1": data["event_f1"],
                "false_alarms_per_record": data["false_alarms_per_record"],
                "false_alarms_per_normal_record": data["false_alarms_per_normal_record"],
                "mean_detection_delay_samples": data["mean_detection_delay_samples"],
                "median_detection_delay_samples": data["median_detection_delay_samples"],
                "mean_matched_iou": data["mean_matched_iou"],
                "mean_gt_coverage": data["mean_gt_coverage"],
                "mean_pred_coverage": data["mean_pred_coverage"],
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

    compact = (
        df.set_index(["experiment", "stage", "match_tag"])[metric_cols]
        .unstack("match_tag")
        .sort_index()
    )

    compact.columns = [
        f"{match_tag}__{metric}"
        for metric, match_tag in compact.columns
    ]
    compact = compact.reset_index()

    return compact


def main():
    df = collect_event_rows(EVENT_ROOT)
    df = df.sort_values(["min_iou", "experiment", "stage"]).reset_index(drop=True)

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