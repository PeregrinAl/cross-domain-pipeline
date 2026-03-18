from pathlib import Path

import numpy as np
import pandas as pd


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

        rows.append(
            {
                "path": str(path).replace("\\", "/"),
                "label": 0,
                "domain": "source",
                "record_id": f"src_train_norm_{i}",
                "split": "train",
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
    x = inject_spike_anomaly(x, start=3000, width=180, amplitude=4.5)
    path = raw_dir / "source_train_anomaly_spike_0.npy"
    save_signal(path, x)

    rows.append(
        {
            "path": str(path).replace("\\", "/"),
            "label": 1,
            "domain": "source",
            "record_id": "src_train_anom_spike_0",
            "split": "train",
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
    x = inject_burst_anomaly(x, start=7600, width=320, amplitude=3.0, cycles=18.0)
    path = raw_dir / "source_train_anomaly_burst_0.npy"
    save_signal(path, x)

    rows.append(
        {
            "path": str(path).replace("\\", "/"),
            "label": 1,
            "domain": "source",
            "record_id": "src_train_anom_burst_0",
            "split": "train",
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

        rows.append(
            {
                "path": str(path).replace("\\", "/"),
                "label": 0,
                "domain": "source",
                "record_id": f"src_val_norm_{i}",
                "split": "val",
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
    x = inject_spike_anomaly(x, start=4200, width=180, amplitude=4.5)
    path = raw_dir / "source_val_anomaly_spike.npy"
    save_signal(path, x)

    rows.append(
        {
            "path": str(path).replace("\\", "/"),
            "label": 1,
            "domain": "source",
            "record_id": "src_val_anom_spike",
            "split": "val",
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
    x = inject_burst_anomaly(x, start=7000, width=320, amplitude=3.0, cycles=18.0)
    path = raw_dir / "source_val_anomaly_burst.npy"
    save_signal(path, x)

    rows.append(
        {
            "path": str(path).replace("\\", "/"),
            "label": 1,
            "domain": "source",
            "record_id": "src_val_anom_burst",
            "split": "val",
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
    x = inject_spike_anomaly(x, start=3500, width=220, amplitude=4.0)
    path = raw_dir / "target_test_anomaly_spike.npy"
    save_signal(path, x)

    rows.append(
        {
            "path": str(path).replace("\\", "/"),
            "label": 1,
            "domain": "target",
            "record_id": "tgt_test_anom_spike",
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
    x = inject_burst_anomaly(x, start=8200, width=360, amplitude=2.8, cycles=20.0)
    path = raw_dir / "target_test_anomaly_burst.npy"
    save_signal(path, x)

    rows.append(
        {
            "path": str(path).replace("\\", "/"),
            "label": 1,
            "domain": "target",
            "record_id": "tgt_test_anom_burst",
            "split": "test",
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

    rows.append(
        {
            "path": str(path).replace("\\", "/"),
            "label": 0,
            "domain": "target",
            "record_id": "tgt_adapt_0",
            "split": "adapt",
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