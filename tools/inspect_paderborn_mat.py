from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat


def describe(obj, prefix="", depth=0, max_depth=6):
    if depth > max_depth:
        print(f"{prefix} <max depth reached>")
        return

    if isinstance(obj, np.ndarray):
        print(f"{prefix} ndarray shape={obj.shape} dtype={obj.dtype}")

        # If this is an object array, inspect its elements
        if obj.dtype == object:
            flat = obj.reshape(-1)
            for i, item in enumerate(flat):
                describe(item, prefix=f"{prefix}[{i}]", depth=depth + 1, max_depth=max_depth)
        return

    if hasattr(obj, "_fieldnames"):
        print(f"{prefix} matlab_struct fields={obj._fieldnames}")
        for field in obj._fieldnames:
            try:
                value = getattr(obj, field)
                describe(value, prefix=f"{prefix}.{field}", depth=depth + 1, max_depth=max_depth)
            except Exception as e:
                print(f"{prefix}.{field} -> error: {e}")
        return

    if isinstance(obj, (list, tuple)):
        print(f"{prefix} {type(obj).__name__} len={len(obj)}")
        for i, item in enumerate(obj):
            describe(item, prefix=f"{prefix}[{i}]", depth=depth + 1, max_depth=max_depth)
        return

    print(f"{prefix} type={type(obj)} value={repr(obj)[:120]}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python tools/inspect_paderborn_mat.py <path_to_mat>")

    mat_path = Path(sys.argv[1])
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    print(f"FILE: {mat_path}")
    print("TOP-LEVEL KEYS:")
    for key, value in data.items():
        if key.startswith("__"):
            continue
        print(f"\nKEY: {key}")
        describe(value, prefix=key)


if __name__ == "__main__":
    main()