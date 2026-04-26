import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_FAULT_KEYWORDS = ["imbalance"]
TARGET_FAULT_KEYWORDS = ["horizontal-misalignment", "vertical-misalignment"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, default="data/external/mafaulda/raw")
    parser.add_argument("--out-records", type=str, default="data/raw/mafaulda_records.csv")
    parser.add_argument("--out-signals", type=str, default="data/raw/mafaulda_npy")
    parser.add_argument("--channel-index", type=int, default=1)
    parser.add_argument("--max-files-per-class", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--adapt-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_csv_signal(path: Path, channel_index: int) -> np.ndarray:
    try:
        df = pd.read_csv(path, header=None)
    except Exception:
        df = pd.read_csv(path, header=None, sep=";")

    if channel_index >= df.shape[1]:
        raise ValueError(
            f"channel_index={channel_index} is out of range for {path}, "
            f"columns={df.shape[1]}"
        )

    x = df.iloc[:, channel_index].to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


def infer_class(path: Path) -> str:
    parts = [p.lower() for p in path.parts]

    if any("normal" == p or "normal" in p for p in parts):
        return "normal"

    for key in SOURCE_FAULT_KEYWORDS + TARGET_FAULT_KEYWORDS:
        if any(key in p for p in parts):
            return key

    return "unknown_fault"


def class_to_domain(cls: str) -> str:
    if cls == "normal":
        return "normal_pool"

    if cls in SOURCE_FAULT_KEYWORDS:
        return "source"

    if cls in TARGET_FAULT_KEYWORDS:
        return "target"

    return "target"


def make_split(domain: str, idx: int, n: int, val_fraction: float, adapt_fraction: float) -> tuple[str, str]:
    if domain == "source":
        n_val = max(1, int(n * val_fraction))
        return "source", "val" if idx < n_val else "train"

    if domain == "target":
        n_adapt = max(1, int(n * adapt_fraction))
        return "target", "adapt" if idx < n_adapt else "test"

    raise ValueError(f"Unsupported domain: {domain}")


def main():
    args = parse_args()

    input_root = Path(args.input_root)
    out_records = Path(args.out_records)
    out_signals = Path(args.out_signals)

    out_records.parent.mkdir(parents=True, exist_ok=True)
    out_signals.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {input_root}")

    by_class = {}
    for path in csv_files:
        cls = infer_class(path)
        if cls == "unknown_fault":
            continue
        by_class.setdefault(cls, []).append(path)

    rng = np.random.default_rng(args.seed)

    for cls, files in by_class.items():
        files = sorted(files)
        rng.shuffle(files)
        if args.max_files_per_class > 0:
            files = files[: args.max_files_per_class]
        by_class[cls] = files

    rows = []

    # Split normal files between source and target, because both domains need normal examples.
    normal_files = by_class.get("normal", [])
    n_source_normal = len(normal_files) // 2
    source_normal_files = normal_files[:n_source_normal]
    target_normal_files = normal_files[n_source_normal:]

    class_groups = {
        "source_normal": ("source", source_normal_files, 0),
        "target_normal": ("target", target_normal_files, 0),
    }

    for cls in SOURCE_FAULT_KEYWORDS:
        class_groups[f"source_{cls}"] = ("source", by_class.get(cls, []), 1)

    for cls in TARGET_FAULT_KEYWORDS:
        class_groups[f"target_{cls}"] = ("target", by_class.get(cls, []), 1)

    for group_name, (domain, files, label) in class_groups.items():
        if not files:
            print(f"WARNING: no files for {group_name}")
            continue

        for idx, csv_path in enumerate(files):
            x = read_csv_signal(csv_path, channel_index=args.channel_index)

            split_domain, split = make_split(
                domain=domain,
                idx=idx,
                n=len(files),
                val_fraction=args.val_fraction,
                adapt_fraction=args.adapt_fraction,
            )

            record_id = f"mafaulda_{group_name}_{idx:04d}"
            npy_path = out_signals / f"{record_id}.npy"
            np.save(npy_path, x)

            if label == 1:
                anomaly_intervals = json.dumps([[0, int(len(x))]])
            else:
                anomaly_intervals = json.dumps([])

            rows.append(
                {
                    "record_id": record_id,
                    "path": str(npy_path).replace("\\", "/"),
                    "domain": split_domain,
                    "split": split,
                    "label": label,
                    "anomaly_intervals": anomaly_intervals,
                    "source_csv": str(csv_path).replace("\\", "/"),
                    "class_name": group_name,
                    "channel_index": args.channel_index,
                    "num_samples": int(len(x)),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_records, index=False)

    print(f"Saved records: {out_records}")
    print(f"Rows: {len(df)}")
    print()
    print(df.groupby(["domain", "split", "label"]).size())


if __name__ == "__main__":
    main()