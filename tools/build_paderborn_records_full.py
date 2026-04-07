from pathlib import Path
import pandas as pd


NPY_DIR = Path("data/raw/paderborn_npy")
OUT_CSV = Path("data/raw/paderborn_records.csv")

# One source condition, one target condition
SOURCE_SETTING = "N15_M07_F10"
TARGET_SETTING = "N09_M07_F10"

# Fixed clean split by bearing code
SPLIT_ASSIGNMENT = {
    ("source", "train", 0): "K001",
    ("source", "train", 1): "KA01",
    ("source", "val",   0): "K002",
    ("source", "val",   1): "KA05",
    ("target", "adapt", 0): "K003",
    ("target", "adapt", 1): "KA03",
    ("target", "test",  0): "K004",
    ("target", "test",  1): "KA04",
}


def collect_rows_for_code(setting: str, bearing_code: str, label: int, domain: str, split: str):
    rows = []

    for measurement_idx in range(1, 21):
        fname = f"{setting}_{bearing_code}_{measurement_idx}.npy"
        path = NPY_DIR / fname

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        rows.append(
            {
                "path": str(path),
                "label": label,
                "domain": domain,
                "record_id": f"{setting}_{bearing_code}_{measurement_idx}",
                "split": split,
                "setting": setting,
                "bearing_code": bearing_code,
                "measurement_idx": measurement_idx,
                "modality": "vibration",
            }
        )

    return rows


def main():
    all_rows = []

    for (domain, split, label), bearing_code in SPLIT_ASSIGNMENT.items():
        setting = SOURCE_SETTING if domain == "source" else TARGET_SETTING
        rows = collect_rows_for_code(
            setting=setting,
            bearing_code=bearing_code,
            label=label,
            domain=domain,
            split=split,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    df = df.sort_values(
        ["domain", "split", "label", "bearing_code", "measurement_idx"]
    ).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}")
    print(f"Total rows: {len(df)}")
    print()
    print(df.groupby(["domain", "split", "label"]).size())
    print()
    print("All paths exist:", df["path"].map(lambda p: Path(p).exists()).all())


if __name__ == "__main__":
    main()