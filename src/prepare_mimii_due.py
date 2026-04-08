import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to unpacked dev_data_<machine> directory")
    parser.add_argument("--machine", type=str, required=True, help="Machine type, for example: fan")
    parser.add_argument("--section", type=str, required=True, help="Section id, for example: 00")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation fraction taken from source-domain labeled test clips")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for .npy clips and records.csv. "
             "Default: data/raw/mimii_due/<machine>_section_<section>",
    )
    return parser.parse_args()


def normalize_section(section: str) -> str:
    s = str(section).strip()
    if not s.isdigit():
        raise ValueError(f"Section must be numeric, got: {section}")
    return f"{int(s):02d}"


def path_tokens(path: Path) -> List[str]:
    text = str(path).replace("\\", "/").lower()
    return [t for t in re.split(r"[^a-z0-9]+", text) if t]


def infer_label(path: Path) -> int:
    tokens = set(path_tokens(path))
    if "anomaly" in tokens or "abnormal" in tokens:
        return 1
    if "normal" in tokens:
        return 0
    raise ValueError(f"Could not infer label from path: {path}")


def infer_domain(path: Path) -> str:
    tokens = set(path_tokens(path))
    has_source = "source" in tokens
    has_target = "target" in tokens
    if has_source and has_target:
        raise ValueError(f"Ambiguous domain in path: {path}")
    if has_source:
        return "source"
    if has_target:
        return "target"
    raise ValueError(f"Could not infer domain from path: {path}")


def infer_stage(path: Path) -> str:
    tokens = set(path_tokens(path))
    has_train = "train" in tokens
    has_test = "test" in tokens
    if has_train and has_test:
        raise ValueError(f"Ambiguous stage in path: {path}")
    if has_train:
        return "train"
    if has_test:
        return "test"
    raise ValueError(f"Could not infer stage from path: {path}")


def infer_section(path: Path) -> Optional[str]:
    text = str(path).replace("\\", "/").lower()
    match = re.search(r"section[_\-]?0*([0-9]+)", text)
    if match:
        return f"{int(match.group(1)):02d}"

    tokens = path_tokens(path)
    for i, token in enumerate(tokens):
        if token == "section" and i + 1 < len(tokens) and tokens[i + 1].isdigit():
            return f"{int(tokens[i + 1]):02d}"

    return None


def infer_machine(path: Path) -> Optional[str]:
    tokens = set(path_tokens(path))
    for machine in ["fan", "gearbox", "pump", "slider", "slide", "rail", "valve"]:
        if machine in tokens:
            if machine == "slide" or machine == "rail":
                return "slider"
            return machine
    return None


def load_audio_mono(path: Path) -> np.ndarray:
    waveform, _sr = sf.read(str(path), dtype="float32", always_2d=True)
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform shape [T, C], got {waveform.shape} in {path}")
    mono = waveform[:, 0].astype(np.float32)
    if mono.ndim != 1:
        raise ValueError(f"Expected mono waveform shape [T], got {mono.shape} in {path}")
    return mono


def save_npy(signal: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, signal.astype(np.float32))


def safe_split_source_test(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("No source-domain labeled test clips found for val split")

    y = df["label"].astype(int).values
    label_counts = pd.Series(y).value_counts()

    stratify = None
    if len(label_counts) >= 2 and int(label_counts.min()) >= 2:
        stratify = y

    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    machine = args.machine.strip().lower()
    section = normalize_section(args.section)

    if args.output_dir is None:
        output_dir = Path(f"data/raw/mimii_due/{machine}_section_{section}")
    else:
        output_dir = Path(args.output_dir)

    wav_paths = sorted(root.rglob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found under: {root}")

    collected: List[Dict] = []

    for wav_path in wav_paths:
        rel = wav_path.relative_to(root)

        file_machine = infer_machine(rel)
        if file_machine is not None and file_machine != machine:
            continue

        file_section = infer_section(rel)
        if file_section is None:
            raise ValueError(f"Could not infer section from path: {wav_path}")
        if file_section != section:
            continue

        label = infer_label(rel)
        domain = infer_domain(rel)
        stage = infer_stage(rel)

        signal = load_audio_mono(wav_path)

        record_id = rel.with_suffix("").as_posix().replace("/", "__")
        npy_path = output_dir / "signals" / domain / stage / f"{record_id}.npy"
        save_npy(signal, npy_path)

        collected.append(
            {
                "path": str(npy_path).replace("\\", "/"),
                "label": int(label),
                "domain": domain,
                "stage": stage,
                "record_id": record_id,
            }
        )

    if not collected:
        raise ValueError(
            f"No clips collected for machine={machine}, section={section} under root={root}"
        )

    df = pd.DataFrame(collected)

    source_train_df = df[(df["domain"] == "source") & (df["stage"] == "train")].copy()
    source_test_df = df[(df["domain"] == "source") & (df["stage"] == "test")].copy()
    target_train_df = df[(df["domain"] == "target") & (df["stage"] == "train")].copy()
    target_test_df = df[(df["domain"] == "target") & (df["stage"] == "test")].copy()

    if source_train_df.empty:
        raise ValueError("No source/train clips found")
    if source_test_df.empty:
        raise ValueError("No source/test clips found")
    if target_train_df.empty:
        raise ValueError("No target/train clips found")
    if target_test_df.empty:
        raise ValueError("No target/test clips found")

    source_test_train_df, source_val_df = safe_split_source_test(
        source_test_df, val_ratio=args.val_ratio, seed=args.seed
    )

    source_train_df["split"] = "train"
    source_test_train_df["split"] = "train"
    source_val_df["split"] = "val"
    target_train_df["split"] = "adapt"
    target_test_df["split"] = "test"

    records_df = pd.concat(
        [
            source_train_df[["path", "label", "domain", "record_id", "split"]],
            source_test_train_df[["path", "label", "domain", "record_id", "split"]],
            source_val_df[["path", "label", "domain", "record_id", "split"]],
            target_train_df[["path", "label", "domain", "record_id", "split"]],
            target_test_df[["path", "label", "domain", "record_id", "split"]],
        ],
        axis=0,
        ignore_index=True,
    )

    records_df = records_df.sort_values(["domain", "split", "record_id"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    records_csv = output_dir / "records.csv"
    records_df.to_csv(records_csv, index=False)

    print(f"Saved records.csv to: {records_csv}")
    print(f"Total records: {len(records_df)}")
    print()
    print(
        records_df.groupby(["domain", "split", "label"])
        .size()
        .reset_index(name="count")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()