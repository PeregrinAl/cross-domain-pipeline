import yaml
from torch.utils.data import DataLoader

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


def build_dataset(config, split: str, domain: str):
    transform = build_transform(config)

    manifest_path = config["data"].get("manifest_path", "data/processed/manifest.csv")

    dataset = SignalWindowDataset(
        manifest_path=manifest_path,
        split=split,
        domain=domain,
        transform=transform,
    )
    return dataset


def build_dataloader(config, split: str, domain: str, shuffle: bool):
    dataset = build_dataset(config, split=split, domain=domain)

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"].get("pin_memory", False),
    )
    return dataloader


def load_config(config_path: str = "configs/base.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)