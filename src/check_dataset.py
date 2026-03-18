from src.data.dataset import SignalWindowDataset


def main():
    dataset = SignalWindowDataset(
        manifest_path="data/processed/manifest.csv",
        split="train",
        domain="source",
    )

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print("Keys:", sample.keys())
    print("x_raw shape:", sample["x_raw"].shape)
    print("label:", sample["label"])
    print("domain:", sample["domain"])
    print("record_id:", sample["record_id"])


if __name__ == "__main__":
    main()