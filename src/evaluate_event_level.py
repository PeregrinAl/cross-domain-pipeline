from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.windowing import parse_intervals


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["source_only", "sfda_before", "sfda_after"],
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["raw_only", "tfr_only", "fused"],
    )
    parser.add_argument("--scores-csv", type=str, default=None)
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="experiments/event_level")
    parser.add_argument("--min-iou", type=float, default=0.0)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_threshold(summary: dict, stage: str) -> float:
    if stage == "source_only":
        if "threshold_used" in summary:
            return float(summary["threshold_used"])
        if "best_threshold_source_val_f1" in summary:
            return float(summary["best_threshold_source_val_f1"])
        raise KeyError("Could not find source-only threshold in summary.json")

    if stage == "sfda_before":
        if "target_threshold_before" in summary:
            return float(summary["target_threshold_before"])
        raise KeyError("Could not find target_threshold_before in SFDA summary.json")

    if stage == "sfda_after":
        if "target_threshold_after" in summary:
            return float(summary["target_threshold_after"])
        raise KeyError("Could not find target_threshold_after in SFDA summary.json")

    raise ValueError(f"Unsupported stage: {stage}")


def resolve_artifacts(config: dict, stage: str, variant: str, scores_csv: str | None, summary_json: str | None):
    if stage != "source_only" and variant != "fused":
        raise ValueError("SFDA event evaluation is currently supported only for variant='fused'")

    if stage == "source_only":
        base_dir = Path(config["outputs"]["source_only_dir"]) / variant
        default_scores = base_dir / "target_test_scores.csv"
        default_summary = base_dir / "summary.json"
        run_name = f"{variant}__source_only"
    elif stage == "sfda_before":
        base_dir = Path(config["outputs"]["sfda_dir"]) / variant
        default_scores = base_dir / "target_test_scores_before.csv"
        default_summary = base_dir / "summary.json"
        run_name = f"{variant}__sfda_before"
    elif stage == "sfda_after":
        base_dir = Path(config["outputs"]["sfda_dir"]) / variant
        default_scores = base_dir / "target_test_scores_after.csv"
        default_summary = base_dir / "summary.json"
        run_name = f"{variant}__sfda_after"
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    resolved_scores = Path(scores_csv) if scores_csv is not None else default_scores
    resolved_summary = Path(summary_json) if summary_json is not None else default_summary
    return resolved_scores, resolved_summary, run_name


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []

    intervals = sorted((int(s), int(e)) for s, e in intervals if int(e) > int(s))
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def interval_overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    overlap = interval_overlap(a, b)
    if overlap <= 0:
        return 0.0
    len_a = a[1] - a[0]
    len_b = b[1] - b[0]
    union = len_a + len_b - overlap
    return float(overlap / union) if union > 0 else 0.0


def build_predicted_events(record_windows: pd.DataFrame, threshold: float) -> list[tuple[int, int]]:
    positive = record_windows.loc[record_windows["score"] >= threshold].copy()
    if positive.empty:
        return []

    positive = positive.sort_values(["start", "end"]).reset_index(drop=True)
    raw_intervals = [
        (int(row.start), int(row.end))
        for row in positive.itertuples(index=False)
    ]
    return merge_intervals(raw_intervals)


def match_events(
    gt_events: list[tuple[int, int]],
    pred_events: list[tuple[int, int]],
    min_iou: float = 0.0,
):
    candidates = []

    for gt_idx, gt_event in enumerate(gt_events):
        for pred_idx, pred_event in enumerate(pred_events):
            overlap = interval_overlap(gt_event, pred_event)
            if overlap <= 0:
                continue

            iou = interval_iou(gt_event, pred_event)
            if iou < min_iou:
                continue

            gt_len = gt_event[1] - gt_event[0]
            pred_len = pred_event[1] - pred_event[0]
            delay = max(0, pred_event[0] - gt_event[0])

            candidates.append(
                {
                    "gt_idx": gt_idx,
                    "pred_idx": pred_idx,
                    "gt_start": int(gt_event[0]),
                    "gt_end": int(gt_event[1]),
                    "pred_start": int(pred_event[0]),
                    "pred_end": int(pred_event[1]),
                    "overlap_samples": int(overlap),
                    "iou": float(iou),
                    "gt_coverage": float(overlap / gt_len) if gt_len > 0 else np.nan,
                    "pred_coverage": float(overlap / pred_len) if pred_len > 0 else np.nan,
                    "delay_samples": int(delay),
                }
            )

    candidates.sort(
        key=lambda x: (x["overlap_samples"], x["iou"], -x["delay_samples"]),
        reverse=True,
    )

    matched_gt = set()
    matched_pred = set()
    matches = []

    for item in candidates:
        if item["gt_idx"] in matched_gt or item["pred_idx"] in matched_pred:
            continue
        matched_gt.add(item["gt_idx"])
        matched_pred.add(item["pred_idx"])
        matches.append(item)

    missed_gt_indices = [i for i in range(len(gt_events)) if i not in matched_gt]
    false_pred_indices = [i for i in range(len(pred_events)) if i not in matched_pred]

    return matches, missed_gt_indices, false_pred_indices


def f1_from_precision_recall(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def load_scores_with_timeline(scores_path: Path, manifest_path: Path) -> pd.DataFrame:
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores CSV not found: {scores_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

    scores_df = pd.read_csv(scores_path)
    manifest_df = pd.read_csv(manifest_path)

    required_scores = {"path", "label", "score", "domain", "record_id", "split"}
    missing_scores = required_scores - set(scores_df.columns)
    if missing_scores:
        raise ValueError(f"Scores CSV is missing required columns: {missing_scores}")

    required_manifest = {"path", "start", "end", "window_idx"}
    missing_manifest = required_manifest - set(manifest_df.columns)
    if missing_manifest:
        raise ValueError(f"Manifest CSV is missing required columns: {missing_manifest}")

    manifest_cols = [
        col
        for col in [
            "path",
            "window_idx",
            "start",
            "end",
            "record_label",
            "overlap_samples",
            "overlap_fraction",
        ]
        if col in manifest_df.columns
    ]

    merged = scores_df.merge(
        manifest_df[manifest_cols],
        on="path",
        how="left",
        validate="one_to_one",
    )

    if merged["start"].isna().any() or merged["end"].isna().any():
        bad_paths = merged.loc[merged["start"].isna() | merged["end"].isna(), "path"].head(5).tolist()
        raise ValueError(
            "Could not join some score rows with manifest rows by path. "
            f"Example paths: {bad_paths}"
        )

    merged["start"] = merged["start"].astype(int)
    merged["end"] = merged["end"].astype(int)
    if "window_idx" in merged.columns:
        merged["window_idx"] = merged["window_idx"].astype(int)

    return merged.sort_values(["record_id", "start", "end"]).reset_index(drop=True)


def load_records_for_eval(records_csv: Path, anomaly_intervals_column: str, relevant_records: pd.DataFrame):
    if not records_csv.exists():
        raise FileNotFoundError(f"Records CSV not found: {records_csv}")

    records_df = pd.read_csv(records_csv)

    required_cols = {"record_id", "domain", "split"}
    missing = required_cols - set(records_df.columns)
    if missing:
        raise ValueError(f"Records CSV is missing required columns: {missing}")

    if anomaly_intervals_column not in records_df.columns:
        raise ValueError(
            f"Records CSV does not contain '{anomaly_intervals_column}' column"
        )

    subset = records_df.merge(
        relevant_records.drop_duplicates(),
        on=["record_id", "domain", "split"],
        how="inner",
        validate="one_to_one",
    ).copy()

    records_map = {}
    for _, row in subset.iterrows():
        key = (str(row["record_id"]), str(row["domain"]), str(row["split"]))
        intervals = parse_intervals(row[anomaly_intervals_column])
        records_map[key] = {
            "record_label": int(row["label"]) if "label" in row else None,
            "anomaly_intervals": merge_intervals(intervals),
        }

    return records_map


def summarize_match_metric(values: list[float]) -> float | None:
    if len(values) == 0:
        return None
    return float(np.mean(values))


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    manifest_path = Path(config["data"]["manifest_path"])
    records_csv = Path(config["data"]["raw_records_csv"])
    anomaly_intervals_column = config["data"].get("anomaly_intervals_column", "anomaly_intervals")

    scores_path, summary_path, run_name = resolve_artifacts(
        config=config,
        stage=args.stage,
        variant=args.variant,
        scores_csv=args.scores_csv,
        summary_json=args.summary_json,
    )

    summary_data = load_json(summary_path)
    threshold = float(args.threshold) if args.threshold is not None else resolve_threshold(summary_data, args.stage)

    window_df = load_scores_with_timeline(scores_path=scores_path, manifest_path=manifest_path)
    window_df["pred"] = (window_df["score"] >= threshold).astype(int)

    relevant_records = window_df[["record_id", "domain", "split"]].drop_duplicates().copy()
    records_map = load_records_for_eval(
        records_csv=records_csv,
        anomaly_intervals_column=anomaly_intervals_column,
        relevant_records=relevant_records,
    )

    match_tag = f"iou_{args.min_iou:.3f}".replace(".", "p")

    out_dir = Path(args.output_dir) / match_tag / args.stage / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    matched_rows = []
    missed_rows = []
    false_alarm_rows = []
    per_record_rows = []
    gt_rows = []
    pred_rows = []

    total_gt_events = 0
    total_pred_events = 0
    total_matched_events = 0
    total_false_alarm_events = 0
    total_missed_events = 0

    all_delays = []
    all_ious = []
    all_gt_coverages = []
    all_pred_coverages = []

    for (record_id, domain, split), record_windows in window_df.groupby(["record_id", "domain", "split"], sort=False):
        key = (str(record_id), str(domain), str(split))
        if key not in records_map:
            raise KeyError(f"Record not found in records.csv for key={key}")

        gt_events = records_map[key]["anomaly_intervals"]
        pred_events = build_predicted_events(record_windows, threshold=threshold)

        for gt_idx, (gt_start, gt_end) in enumerate(gt_events):
            gt_rows.append(
                {
                    "record_id": record_id,
                    "domain": domain,
                    "split": split,
                    "gt_idx": gt_idx,
                    "gt_start": gt_start,
                    "gt_end": gt_end,
                    "gt_len": gt_end - gt_start,
                }
            )

        for pred_idx, (pred_start, pred_end) in enumerate(pred_events):
            pred_rows.append(
                {
                    "record_id": record_id,
                    "domain": domain,
                    "split": split,
                    "pred_idx": pred_idx,
                    "pred_start": pred_start,
                    "pred_end": pred_end,
                    "pred_len": pred_end - pred_start,
                }
            )

        matches, missed_gt_indices, false_pred_indices = match_events(
            gt_events=gt_events,
            pred_events=pred_events,
            min_iou=args.min_iou,
        )

        for item in matches:
            row = {
                "record_id": record_id,
                "domain": domain,
                "split": split,
                **item,
            }
            matched_rows.append(row)
            all_delays.append(item["delay_samples"])
            all_ious.append(item["iou"])
            all_gt_coverages.append(item["gt_coverage"])
            all_pred_coverages.append(item["pred_coverage"])

        for gt_idx in missed_gt_indices:
            gt_start, gt_end = gt_events[gt_idx]
            missed_rows.append(
                {
                    "record_id": record_id,
                    "domain": domain,
                    "split": split,
                    "gt_idx": gt_idx,
                    "gt_start": gt_start,
                    "gt_end": gt_end,
                    "gt_len": gt_end - gt_start,
                }
            )

        for pred_idx in false_pred_indices:
            pred_start, pred_end = pred_events[pred_idx]
            false_alarm_rows.append(
                {
                    "record_id": record_id,
                    "domain": domain,
                    "split": split,
                    "pred_idx": pred_idx,
                    "pred_start": pred_start,
                    "pred_end": pred_end,
                    "pred_len": pred_end - pred_start,
                }
            )

        n_gt = len(gt_events)
        n_pred = len(pred_events)
        n_matched = len(matches)
        n_missed = len(missed_gt_indices)
        n_false = len(false_pred_indices)

        total_gt_events += n_gt
        total_pred_events += n_pred
        total_matched_events += n_matched
        total_false_alarm_events += n_false
        total_missed_events += n_missed

        record_delays = [item["delay_samples"] for item in matches]
        per_record_rows.append(
            {
                "record_id": record_id,
                "domain": domain,
                "split": split,
                "n_windows": int(len(record_windows)),
                "n_gt_events": int(n_gt),
                "n_pred_events": int(n_pred),
                "n_matched_events": int(n_matched),
                "n_missed_events": int(n_missed),
                "n_false_alarm_events": int(n_false),
                "record_event_recall": float(n_matched / n_gt) if n_gt > 0 else np.nan,
                "record_event_precision": float(n_matched / n_pred) if n_pred > 0 else np.nan,
                "record_mean_delay_samples": float(np.mean(record_delays)) if len(record_delays) > 0 else np.nan,
            }
        )

    n_records = int(len(per_record_rows))
    precision = float(total_matched_events / total_pred_events) if total_pred_events > 0 else 0.0
    recall = float(total_matched_events / total_gt_events) if total_gt_events > 0 else 0.0
    event_f1 = f1_from_precision_recall(precision, recall)

    n_normal_records = int(sum(1 for row in per_record_rows if row["n_gt_events"] == 0))
    false_alarm_events_on_normal_records = int(
        sum(row["n_false_alarm_events"] for row in per_record_rows if row["n_gt_events"] == 0)
    )

    summary = {
        "run_name": run_name,
        "stage": args.stage,
        "variant": args.variant,
        "scores_csv": str(scores_path),
        "summary_json": str(summary_path),
        "manifest_csv": str(manifest_path),
        "records_csv": str(records_csv),
        "threshold_used": float(threshold),
        "min_iou": float(args.min_iou),
        "match_tag": match_tag,
        "n_records": int(n_records),
        "n_gt_events": int(total_gt_events),
        "n_pred_events": int(total_pred_events),
        "n_matched_events": int(total_matched_events),
        "n_missed_events": int(total_missed_events),
        "n_false_alarm_events": int(total_false_alarm_events),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(event_f1),
        "false_alarms_per_record": float(total_false_alarm_events / n_records) if n_records > 0 else None,
        "false_alarms_per_normal_record": float(false_alarm_events_on_normal_records / n_normal_records)
        if n_normal_records > 0
        else None,
        "mean_detection_delay_samples": summarize_match_metric(all_delays),
        "median_detection_delay_samples": float(np.median(all_delays)) if len(all_delays) > 0 else None,
        "mean_matched_iou": summarize_match_metric(all_ious),
        "mean_gt_coverage": summarize_match_metric(all_gt_coverages),
        "mean_pred_coverage": summarize_match_metric(all_pred_coverages),
    }

    window_df.to_csv(out_dir / "window_scores_with_timeline.csv", index=False)
    pd.DataFrame(gt_rows).to_csv(out_dir / "ground_truth_events.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out_dir / "predicted_events.csv", index=False)
    pd.DataFrame(matched_rows).to_csv(out_dir / "matched_events.csv", index=False)
    pd.DataFrame(missed_rows).to_csv(out_dir / "missed_events.csv", index=False)
    pd.DataFrame(false_alarm_rows).to_csv(out_dir / "false_alarm_events.csv", index=False)
    pd.DataFrame(per_record_rows).to_csv(out_dir / "per_record_summary.csv", index=False)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved event-level outputs to: {out_dir}")


if __name__ == "__main__":
    main()