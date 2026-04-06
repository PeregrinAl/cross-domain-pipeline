import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


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

def resample_to_length(x: np.ndarray, new_length: int) -> np.ndarray:
    old_idx = np.linspace(0.0, 1.0, num=len(x), dtype=np.float32)
    new_idx = np.linspace(0.0, 1.0, num=new_length, dtype=np.float32)
    y = np.interp(new_idx, old_idx, x).astype(np.float32)
    return y


def apply_amplitude_scale(x: np.ndarray, scale: float) -> np.ndarray:
    return (x * scale).astype(np.float32)


def apply_extra_noise(
    x: np.ndarray,
    rng: np.random.Generator,
    noise_std: float,
) -> np.ndarray:
    noise = rng.normal(0.0, noise_std, size=len(x)).astype(np.float32)
    return (x + noise).astype(np.float32)


def apply_extra_trend(x: np.ndarray, trend_strength: float) -> np.ndarray:
    trend = np.linspace(0.0, trend_strength, num=len(x), dtype=np.float32)
    return (x + trend).astype(np.float32)


def apply_frequency_shift(
    x: np.ndarray,
    rng: np.random.Generator,
    multiplier_min: float,
    multiplier_max: float,
) -> np.ndarray:
    factor = float(rng.uniform(multiplier_min, multiplier_max))
    warped = resample_to_length(x, max(8, int(len(x) / factor)))
    y = resample_to_length(warped, len(x))
    return y.astype(np.float32)


def apply_temporal_warp(
    x: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num=len(x), dtype=np.float32)
    anchors = np.linspace(0.0, 1.0, num=8, dtype=np.float32)
    jitter = rng.normal(0.0, sigma, size=len(anchors)).astype(np.float32)

    warped_anchors = np.clip(anchors + jitter, 0.0, 1.0)
    warped_anchors[0] = 0.0
    warped_anchors[-1] = 1.0
    warped_anchors = np.maximum.accumulate(warped_anchors)

    warped_t = np.interp(t, anchors, warped_anchors).astype(np.float32)
    y = np.interp(t, warped_t, x).astype(np.float32)
    return y


def apply_domain_shift(
    x: np.ndarray,
    rng: np.random.Generator,
    shift_cfg: dict,
) -> np.ndarray:
    y = x.astype(np.float32)

    scale = float(
        rng.uniform(
            shift_cfg["amplitude_scale_min"],
            shift_cfg["amplitude_scale_max"],
        )
    )
    y = apply_amplitude_scale(y, scale)
    y = apply_extra_noise(y, rng, shift_cfg["extra_noise_std"])
    y = apply_extra_trend(y, shift_cfg["extra_trend_strength"])
    # y = apply_frequency_shift(
    #     y,
    #     rng,
    #     shift_cfg["freq_multiplier_min"],
    #     shift_cfg["freq_multiplier_max"],
    # )
    # y = apply_temporal_warp(y, rng, shift_cfg["temporal_warp_sigma"])
    return y.astype(np.float32)

def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    rows = []

    length = 12000

    config_path = Path("configs/base.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    target_shift_cfg = config["target_shift"]

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
    # TARGET TEST: normal + anomaly with stronger domain shift
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
        x = apply_domain_shift(x, rng, target_shift_cfg)

        path = raw_dir / f"target_test_normal_{i}.npy"
        save_signal(path, x)
        rows.append(
            {
                "path": str(path).replace("\\", "/"),
                "label": 0,
                "domain": "target",
                "record_id": f"tgt_test_norm_{i}",
                "split": "test",
                "anomaly_intervals": to_intervals_json([]),
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
    x = apply_domain_shift(x, rng, target_shift_cfg)

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
    burst_start = 8200
    burst_width = 360
    cycles = 20.0
    amplitude = 2.8
    x = inject_burst_anomaly(
        x,
        start=burst_start,
        width=burst_width,
        amplitude=amplitude,
        cycles=cycles,
    )
    x = apply_domain_shift(x, rng, target_shift_cfg)

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
    # TARGET ADAPT: mixed unlabeled target split for SFDA
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
        x = apply_domain_shift(x, rng, target_shift_cfg)

        path = raw_dir / f"target_adapt_normal_{i}.npy"
        save_signal(path, x)
        rows.append(
            {
                "path": str(path).replace("\\", "/"),
                "label": 0,
                "domain": "target",
                "record_id": f"tgt_adapt_norm_{i}",
                "split": "adapt",
                "anomaly_intervals": to_intervals_json([]),
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
    adapt_spike_start = 5000
    adapt_spike_width = 220
    x = inject_spike_anomaly(
        x,
        start=adapt_spike_start,
        width=adapt_spike_width,
        amplitude=4.0,
    )
    x = apply_domain_shift(x, rng, target_shift_cfg)

    path = raw_dir / "target_adapt_anomaly_spike_0.npy"
    save_signal(path, x)
    rows.append(
        {
            "path": str(path).replace("\\", "/"),
            "label": 1,
            "domain": "target",
            "record_id": "tgt_adapt_anom_spike_0",
            "split": "adapt",
            "anomaly_intervals": to_intervals_json(
                [(adapt_spike_start, adapt_spike_start + adapt_spike_width)]
            ),
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
    adapt_burst_start = 7600
    adapt_burst_width = 360
    x = inject_burst_anomaly(
        x,
        start=adapt_burst_start,
        width=adapt_burst_width,
        amplitude=2.8,
        cycles=20.0,
    )
    x = apply_domain_shift(x, rng, target_shift_cfg)

    path = raw_dir / "target_adapt_anomaly_burst_0.npy"
    save_signal(path, x)
    rows.append(
        {
            "path": str(path).replace("\\", "/"),
            "label": 1,
            "domain": "target",
            "record_id": "tgt_adapt_anom_burst_0",
            "split": "adapt",
            "anomaly_intervals": to_intervals_json(
                [(adapt_burst_start, adapt_burst_start + adapt_burst_width)]
            ),
        }
    )

if __name__ == "__main__":
    main()