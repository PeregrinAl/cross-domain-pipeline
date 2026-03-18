from pathlib import Path
import numpy as np
import pandas as pd


def main():
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    rng = np.random.default_rng(42)

    # source normal
    for i in range(3):
        x = rng.normal(0, 1, 2048).astype(np.float32)
        path = out_dir / f"source_normal_{i}.npy"
        np.save(path, x)
        rows.append({
            "path": str(path).replace("\\", "/"),
            "label": 0,
            "domain": "source",
            "record_id": f"src_{i}",
            "split": "train",
        })

    # source anomaly
    for i in range(2):
        x = rng.normal(0, 1, 2048).astype(np.float32)
        x[500:550] += 5.0
        path = out_dir / f"source_anomaly_{i}.npy"
        np.save(path, x)
        rows.append({
            "path": str(path).replace("\\", "/"),
            "label": 1,
            "domain": "source",
            "record_id": f"src_anom_{i}",
            "split": "val",
        })

    # target normal
    for i in range(3):
        x = rng.normal(0.2, 1.2, 2048).astype(np.float32)
        path = out_dir / f"target_normal_{i}.npy"
        np.save(path, x)
        rows.append({
            "path": str(path).replace("\\", "/"),
            "label": 0,
            "domain": "target",
            "record_id": f"tgt_{i}",
            "split": "test",
        })

    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"Saved manifest to: {manifest_path}")
    print(manifest.head())


if __name__ == "__main__":
    main()