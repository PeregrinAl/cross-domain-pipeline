from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat


def extract_vibration_signal(mat_path: Path) -> np.ndarray:
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    top_keys = [k for k in data.keys() if not k.startswith("__")]
    if len(top_keys) != 1:
        raise ValueError(f"Expected exactly one top-level data key, got: {top_keys}")

    top = data[top_keys[0]]

    # Based on the observed structure:
    # top.Y[6].Data is the high-rate vibration signal
    y = top.Y
    signal = np.asarray(y[6].Data).squeeze()

    if signal.ndim != 1:
        raise ValueError(f"Expected 1D vibration signal, got shape {signal.shape}")

    return signal.astype(np.float32)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python tools/convert_one_paderborn_mat.py <input_mat> <output_npy>"
        )

    input_mat = Path(sys.argv[1])
    output_npy = Path(sys.argv[2])

    signal = extract_vibration_signal(input_mat)

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, signal)

    print(f"Saved: {output_npy}")
    print(f"Shape: {signal.shape}, dtype: {signal.dtype}")
    print(f"Min: {signal.min():.6f}, Max: {signal.max():.6f}, Mean: {signal.mean():.6f}")


if __name__ == "__main__":
    main()