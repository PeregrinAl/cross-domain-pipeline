import numpy as np

from src.preprocessing import build_preprocessor


def main():
    x = np.sin(np.linspace(0, 10, 2048)).astype(np.float32)

    for name in ["prep_base", "prep_filter", "prep_domain_norm"]:
        preprocessor = build_preprocessor(name)
        y = preprocessor(x)

        print(name)
        print("shape:", y.shape)
        print("mean:", float(y.mean()))
        print("std:", float(y.std()))
        print()


if __name__ == "__main__":
    main()