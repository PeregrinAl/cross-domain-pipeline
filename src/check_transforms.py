import yaml

from src.data.dataset import SignalWindowDataset
from src.data.transforms import Compose, NormalizeRaw, AddSTFT


def build_transform(config):
    transforms = []

    if config["representation"].get("normalize_raw", True):
        transforms.append(NormalizeRaw())

    if config["representation"].get("use_tfr", True):
        tfr_type = config["representation"].get("tfr_type", "stft")
        if tfr_type != "stft":
            raise NotImplementedError(f"Only STFT is supported now, got {tfr_type}")

        transforms.append(
            AddSTFT(
                n_fft=config["representation"]["n_fft"],
                hop_length=config["representation"]["hop_length"],
                win_length=config["representation"]["win_length"],
                log_amplitude=config["representation"]["log_amplitude"],
                normalize_tfr=config["representation"]["normalize_tfr"],
            )
        )

    return Compose(transforms)


def main():
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    transform = build_transform(config)

    dataset = SignalWindowDataset(
        manifest_path="data/processed/manifest.csv",
        split="train",
        domain="source",
        transform=transform,
    )

    sample = dataset[0]

    print("Keys:", sample.keys())
    print("x_raw shape:", sample["x_raw"].shape)
    print("x_raw mean:", sample["x_raw"].mean().item())
    print("x_raw std:", sample["x_raw"].std().item())

    if "x_tfr" in sample:
        print("x_tfr shape:", sample["x_tfr"].shape)
        print("x_tfr min:", sample["x_tfr"].min().item())
        print("x_tfr max:", sample["x_tfr"].max().item())


if __name__ == "__main__":
    main()