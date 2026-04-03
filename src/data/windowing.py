import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_intervals(value) -> List[Tuple[int, int]]:
    if pd.isna(value) or value in ("", "[]", None):
        return []

    raw = json.loads(value)
    intervals = []
    for item in raw:
        if len(item) != 2:
            raise ValueError(f"Invalid interval entry: {item}")
        start, end = int(item[0]), int(item[1])
        if end <= start:
            continue
        intervals.append((start, end))
    return intervals


def compute_overlap_samples(
    window_start: int,
    window_end: int,
    intervals: List[Tuple[int, int]],
) -> int:
    overlap = 0
    for start, end in intervals:
        overlap += max(0, min(window_end, end) - max(window_start, start))
    return int(overlap)


def compute_window_label(
    window_start: int,
    window_end: int,
    intervals: List[Tuple[int, int]],
    mode: str = "any_overlap",
    min_anomaly_fraction: float = 0.0,
):
    overlap_samples = compute_overlap_samples(window_start, window_end, intervals)
    window_len = max(window_end - window_start, 1)
    overlap_fraction = overlap_samples / window_len

    if mode == "any_overlap":
        label = int(overlap_samples > 0)
    elif mode == "min_fraction":
        label = int(overlap_fraction >= min_anomaly_fraction)
    else:
        raise ValueError(f"Unsupported window_label_mode: {mode}")

    return label, overlap_samples, overlap_fraction


def make_windows(
    signal: np.ndarray,
    window_size: int,
    stride: int,
    drop_last: bool = True,
) -> List[np.ndarray]:
    """
    Split a 1D signal into overlapping windows.

    Parameters
    ----------
    signal : np.ndarray
        Input 1D signal of shape [L].
    window_size : int
        Window length.
    stride : int
        Step between consecutive windows.
    drop_last : bool
        If True, discard the last incomplete window.

    Returns
    -------
    List[np.ndarray]
        List of windows, each of shape [window_size].
    """
    if signal.ndim != 1:
        raise ValueError(f"Signal must be 1D, got shape {signal.shape}")

    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    n = len(signal)
    windows = []

    if n < window_size:
        return windows

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        windows.append(signal[start:end].copy())

    if not drop_last:
        last_start = ((n - window_size) // stride + 1) * stride
        if last_start < n:
            tail = signal[last_start:]
            if len(tail) > 0 and len(tail) < window_size:
                padded = np.zeros(window_size, dtype=signal.dtype)
                padded[: len(tail)] = tail
                windows.append(padded)

    return windows


def build_window_manifest(
    records_csv: str,
    output_dir: str,
    window_size: int,
    stride: int,
    drop_last: bool = True,
    anomaly_intervals_column: str = "anomaly_intervals",
    window_label_mode: str = "any_overlap",
    min_anomaly_fraction: float = 0.0,
) -> pd.DataFrame:
    """
    Read long raw signals from records_csv, split them into windows,
    save each window as .npy, and build processed manifest.

    Expected input CSV columns:
    - path
    - label
    - domain
    - record_id
    - split

    Output manifest columns:
    - path
    - label
    - domain
    - record_id
    - split
    - window_idx
    - start
    - end
    """
    records_path = Path(records_csv)
    if not records_path.exists():
        raise FileNotFoundError(f"Records CSV not found: {records_path}")

    df = pd.read_csv(records_path)

    required_columns = {"path", "label", "domain", "record_id", "split"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"records.csv is missing required columns: {missing}")

    output_dir = Path(output_dir)
    windows_root = output_dir / "windows"
    windows_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    has_intervals = anomaly_intervals_column in df.columns  

    for _, row in df.iterrows():
        signal_path = Path(row["path"])
        if not signal_path.exists():
            raise FileNotFoundError(f"Signal file not found: {signal_path}")

        signal = np.load(signal_path).astype(np.float32)

        if signal.ndim != 1:
            raise ValueError(
                f"Expected raw signal shape [L], got {signal.shape} in {signal_path}"
            )

        domain = str(row["domain"])
        split = str(row["split"])
        record_id = str(row["record_id"])
        record_label = int(row["label"])

        if has_intervals:
            intervals = parse_intervals(row[anomaly_intervals_column])
        else:
            intervals = []

        record_windows = make_windows(
            signal=signal,
            window_size=window_size,
            stride=stride,
            drop_last=drop_last,
        )

        save_dir = windows_root / domain / split
        save_dir.mkdir(parents=True, exist_ok=True)
            
        for window_idx, window in enumerate(record_windows):
            start = window_idx * stride
            end = start + window_size

            file_name = f"{record_id}_win_{window_idx:05d}.npy"
            save_path = save_dir / file_name
            np.save(save_path, window)

            if has_intervals:
                label, overlap_samples, overlap_fraction = compute_window_label(
                    window_start=start,
                    window_end=end,
                    intervals=intervals,
                    mode=window_label_mode,
                    min_anomaly_fraction=min_anomaly_fraction,
                )
            else:
                # backward-compatible fallback
                label = record_label
                overlap_samples = None
                overlap_fraction = None

            rows.append(
                {
                    "path": str(save_path).replace("\\", "/"),
                    "label": label,                   # window-level label
                    "record_label": record_label,    # original record-level label
                    "domain": domain,
                    "record_id": record_id,
                    "split": split,
                    "window_idx": window_idx,
                    "start": start,
                    "end": end,
                    "overlap_samples": overlap_samples,
                    "overlap_fraction": overlap_fraction,
                }
            )

    manifest = pd.DataFrame(rows)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    return manifest