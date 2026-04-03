import json
from pathlib import Path

import numpy as np
import pandas as pd


def to_intervals_json(intervals):
    return json.dumps([[int(start), int(end)] for start, end in intervals])


def make_base_signal(
    length: int,
    rng: np.random.Generator,
    mean: float = 0.0,
    std: float = 1.0,
    trend_strength: float = 0.0,
    sinus_freq: float = 0.0,
    sinus_amp: float = 0.0,
):
    x = rng.normal(mean, std, length).astype(np.float32)

    if trend_strength != 0.0:
        trend = np.linspace(0.0, trend_strength, length, dtype=np.float32)
        x = x + trend

    if sinus_freq > 0.0 and sinus_amp > 0.0:
        t = np.arange(length, dtype=np.float32)
        sinus = sinus_amp * np.sin(2.0 * np.pi * sinus_freq * t / length)
        x = x + sinus.astype(np.float32)

    return x.astype(np.float32)


def inject_spike_anomaly(
    x: np.ndarray,
    start: int,
    width: int,
    amplitude: float,
):
    y = x.copy()
    y[start:start + width] += amplitude
    return y.astype(np.float32)


def inject_burst_anomaly(
    x: np.ndarray,
    start: int,
    width: int,
    amplitude: float,
    cycles: float = 12.0,
):
    y = x.copy()
    t = np.arange(width, dtype=np.float32)
    burst = amplitude * np.sin(2.0 * np.pi * cycles * t / width)
    y[start:start + width] += burst.astype(np.float32)
    return y.astype(np.float32)


def save_signal(path: Path, x: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, x.astype(np.float32))


def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    rows = []

    length = 12000

    # ---------------------------
    # SOURCE TRAIN: normal + anomaly
    # ---------------------------
    for i in range(4):
        x = make_base_signal(
            length=length,
            rng=rng,
            mean=0.0,
            std=1.0,
            trend_strength=0.05,
            sinus_freq=3.0,
            sinus_amp=0.15,
        )
        path = raw_dir / f"source_train_normal_{i}.npy"
        save_signal(path, x)

        # для normal record
        rows.append(
            {
                "path": str(path).replace("\\", "/"),
                "label": 0,  # record-level label
                "domain": "source",
                "record_id": f"src_train_norm_{i}",
                "split": "train",
                "anomaly_intervals": to_intervals_json([]),
            }
        )

    x = make_base_signal(
        length=length,
        rng=rng,
        mean=0.0,
        std=1.0,
        trend_strength=0.05,
        sinus_freq=3.0,
        sinus_amp=0.15,
    )
    spike_start = 3000
    spike_width = 180
    amplitude = 4.5

    x = inject_spike_anomaly(x, start=spike_start, width=spike_width, amplitude=amplitude)
    path = raw_dir / "source_train_anomaly_spike_0.npy"
    save_signal(path, x)

    rows.append(
        {
            "path": str(path).replace("\\", "/"),
            "label": 1,
            "domain": "source",
            "record_id": "src_train_anom_spike_0",
            "split": "train",
            "anomaly_intervals": to_intervals_json([(spike_start, spike_start + spike_width)]),
        }
    )

    x = make_base_signal(
        length=length,
        rng=rng,
        mean=0.0,
        std=1.0,
        trend_strength=0.05,
        sinus_freq=3.0,
        sinus_amp=0.15,
    )

    # для burst anomaly
    burst_start = 7600
    burst_width = 320
    x = inject_burst_anomaly(x, start=burst_start, width=burst_width, amplitude=3.0, cycles=18.0)
    
    path = raw_dir / "source_train_anomaly_burst_0.npy"
    save_signal(path, x)

    rows.append(
        {
            "path": str(path).replace("\\", "/"),
            "label": 1,
            "domain": "source",
            "record_id": "src_train_anom_burst_0",
            "split": "train",
            "anomaly_intervals": to_intervals_json([(burst_start, burst_start + burst_width)]),
        }
    )

    # ---------------------------
    # SOURCE VAL: normal + anomaly
    # ---------------------------
    for i in range(2):
        x = make_base_signal(
            length=length,
            rng=rng,
            mean=0.0,
            std=1.0,
            trend_strength=0.05,
            sinus_freq=3.0,
            sinus_amp=0.15,
        )
        path = raw_dir / f"source_val_normal_{i}.npy"
        save_signal(path, x)

        # для normal record
        rows.append(
            {
                "path": str(path).replace("\\", "/"),
                "label": 0,  # record-level label
                "domain": "source",
                "record_id": f"src_val_norm_{i}",
                "split": "val",
                "anomaly_intervals": to_intervals_json([]),
            }
        )

    x = make_base_signal(
        length=length,
        rng=rng,
        mean=0.0,
        std=1.0,
        trend_strength=0.05,
        sinus_freq=3.0,
        sinus_amp=0.15,
    )
    
    spike_start = 4200
    spike_width = 180
    amplitude = 4.5

    x = inject_spike_anomaly(x, start=spike_start, width=spike_width, amplitude=amplitude)
    path = raw_dir / "source_val_anomaly_spike_0.npy"
    save_signal(path, x)

    rows.append(
        {
            "path": str(path).replace("\\", "/"),
            "label": 1,
            "domain": "source",
            "record_id": "src_val_anom_spike_0",
            "split": "val",
            "anomaly_intervals": to_intervals_json([(spike_start, spike_start + spike_width)]),
        }
    )

    x = make_base_signal(
        length=length,
        rng=rng,
        mean=0.0,
        std=1.0,
        trend_strength=0.05,
        sinus_freq=3.0,
        sinus_amp=0.15,
    )

    # для burst anomaly
    burst_start = 7000
    burst_width = 320
    cycles = 18.0
    amplitude = 3.0

    x = inject_burst_anomaly(x, start=burst_start, width=burst_width, amplitude=amplitude, cycles=cycles)
    
    path = raw_dir / "source_val_anomaly_burst_0.npy"
    save_signal(path, x)

    rows.append(
        {
            "path": str(path).replace("\\", "/"),
            "label": 1,
            "domain": "source",
            "record_id": "src_val_anom_burst_0",
            "split": "val",
            "anomaly_intervals": to_intervals_json([(burst_start, burst_start + burst_width)]),
        }
    )

    # ---------------------------
    # TARGET TEST: normal + anomaly with domain shift
    # Domain shift:
    # - mean shift
    # - larger variance
    # - stronger trend
    # - different sinusoid
    # ---------------------------
    for i in range(2):
        x = make_base_signal(
            length=length,
            rng=rng,
            mean=0.25,
            std=1.2,
            trend_strength=0.12,
            sinus_freq=5.0,
            sinus_amp=0.25,
        )
        path = raw_dir / f"target_test_normal_{i}.npy"
        save_signal(path, x)

        rows.append(
            {
                "path": str(path).replace("\\", "/"),
                "label": 0,
                "domain": "target",
                "record_id": f"tgt_test_norm_{i}",
                "split": "test",
            }
        )

    x = make_base_signal(
        length=length,
        rng=rng,
        mean=0.25,
        std=1.2,
        trend_strength=0.12,
        sinus_freq=5.0,
        sinus_amp=0.25,
    )

    spike_start = 3500
    spike_width = 220
    amplitude = 4.0

    x = inject_spike_anomaly(x, start=spike_start, width=spike_width, amplitude=amplitude)
    path = raw_dir / "target_test_anomaly_spike_0.npy"
    save_signal(path, x)

    rows.append(
        {
            "path": str(path).replace("\\", "/"),
            "label": 1,
            "domain": "target",
            "record_id": "tgt_test_anom_spike_0",
            "split": "test",
            "anomaly_intervals": to_intervals_json([(spike_start, spike_start + spike_width)]),
        }
    )

    x = make_base_signal(
        length=length,
        rng=rng,
        mean=0.25,
        std=1.2,
        trend_strength=0.12,
        sinus_freq=5.0,
        sinus_amp=0.25,
    )


    # для burst anomaly
    burst_start = 8200
    burst_width = 360
    cycles = 20.0
    amplitude = 2.8

    x = inject_burst_anomaly(x, start=burst_start, width=burst_width, amplitude=amplitude, cycles=cycles)
    
    path = raw_dir / "target_test_anomaly_burst_0.npy"
    save_signal(path, x)

    rows.append(
        {
            "path": str(path).replace("\\", "/"),
            "label": 1,
            "domain": "target",
            "record_id": "tgt_test_anom_burst_0",
            "split": "test",
            "anomaly_intervals": to_intervals_json([(burst_start, burst_start + burst_width)]),
        }
    )

    # ---------------------------
    # Optional target adaptation split for later SFDA
    # ---------------------------
    x = make_base_signal(
        length=length,
        rng=rng,
        mean=0.25,
        std=1.2,
        trend_strength=0.12,
        sinus_freq=5.0,
        sinus_amp=0.25,
    )
    path = raw_dir / "target_adapt_unlabeled_0.npy"
    save_signal(path, x)
    
    # для normal record
    rows.append(
        {
            "path": str(path).replace("\\", "/"),
            "label": 0,  # record-level label
            "domain": "target",
            "record_id": f"target_adapt_unlabeled_{i}",
            "split": "adapt",
            "anomaly_intervals": to_intervals_json([]),
        }
    )

    records = pd.DataFrame(rows)
    records_path = raw_dir / "records.csv"
    records.to_csv(records_path, index=False)

    print(f"Saved raw records to: {records_path}")
    print(records.groupby(["domain", "split", "label"]).size().reset_index(name="count"))
    print("\nFull records table:")
    print(records)


if __name__ == "__main__":
    main()