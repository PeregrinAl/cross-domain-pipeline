from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "mean",
    "std",
    "rms",
    "abs_mean",
    "ptp",
    "min",
    "max",
    "energy",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_spread",
    "spectral_entropy",
    "dominant_freq_ratio",
]


def build_feature_table(
    csv_path: str | Path,
    base_dir: str | Path | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    csv_file = Path(csv_path)
    base = Path(base_dir) if base_dir is not None else csv_file.parent

    df = pd.read_csv(csv_file)

    if "path" not in df.columns:
        raise ValueError("CSV must contain column: path")

    if max_rows is not None:
        df = df.head(max_rows)

    rows = []

    for _, row in df.iterrows():
        signal_path = _resolve_path(row["path"], base)

        if not signal_path.exists():
            continue

        try:
            signal = np.load(signal_path)
        except Exception:
            continue

        features = extract_features(signal)

        meta = {
            "path": str(signal_path),
            "label": int(row["label"]) if "label" in row and not pd.isna(row["label"]) else -1,
            "domain": str(row["domain"]) if "domain" in row else "",
            "split": str(row["split"]) if "split" in row else "",
            "record_id": str(row["record_id"]) if "record_id" in row else "",
        }

        if "window_idx" in row:
            meta["window_idx"] = int(row["window_idx"])

        if "start" in row:
            meta["start"] = int(row["start"])

        if "end" in row:
            meta["end"] = int(row["end"])

        rows.append({**meta, **features})

    return pd.DataFrame(rows)


def extract_features(x: np.ndarray) -> dict[str, float]:
    signal = _to_1d(x)

    if signal.size == 0:
        return {name: 0.0 for name in FEATURE_COLUMNS}

    signal = signal.astype(np.float32)
    centered = signal - float(np.mean(signal))

    spectrum = np.fft.rfft(centered)
    power = np.abs(spectrum) ** 2

    if power.size > 1:
        power_no_dc = power[1:]
    else:
        power_no_dc = power

    total_power = float(np.sum(power_no_dc))

    if total_power <= 1e-12:
        spectral_centroid = 0.0
        spectral_spread = 0.0
        spectral_entropy = 0.0
        dominant_freq_ratio = 0.0
    else:
        freqs = np.arange(power_no_dc.size, dtype=np.float32)
        probs = power_no_dc / total_power

        spectral_centroid = float(np.sum(freqs * probs))
        spectral_spread = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * probs)))
        spectral_entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs) + 1e-12))
        dominant_freq_ratio = float(np.max(power_no_dc) / total_power)

    zero_crossings = np.mean(np.diff(np.signbit(centered)).astype(np.float32))

    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "rms": float(np.sqrt(np.mean(signal**2))),
        "abs_mean": float(np.mean(np.abs(signal))),
        "ptp": float(np.ptp(signal)),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "energy": float(np.mean(signal**2)),
        "zero_crossing_rate": float(zero_crossings),
        "spectral_centroid": spectral_centroid,
        "spectral_spread": spectral_spread,
        "spectral_entropy": spectral_entropy,
        "dominant_freq_ratio": dominant_freq_ratio,
    }


def _to_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)

    if arr.ndim == 1:
        return arr

    if arr.ndim == 2:
        return arr.reshape(-1)

    return arr.reshape(-1)


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    p = Path(path)

    if p.is_absolute():
        return p

    if p.exists():
        return p

    return base_dir / p