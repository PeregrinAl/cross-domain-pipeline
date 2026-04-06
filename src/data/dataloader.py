import yaml
from torch.utils.data import DataLoader

from src.data.dataset import SignalWindowDataset
from src.data.transforms import Compose, NormalizeRaw, AddSTFT


def build_transform(config):
    transforms = []
    representation_cfg = config["representation"]

    # Backward-compatible raw normalization handling
    raw_normalization = representation_cfg.get("raw_normalization")
    if raw_normalization is None:
        raw_normalization = "zscore" if representation_cfg.get("normalize_raw", True) else "none"

    transforms.append(NormalizeRaw(mode=raw_normalization))

    if representation_cfg.get("use_tfr", True):
        tfr_type = representation_cfg.get("tfr_type", "stft")
        if tfr_type != "stft":
            raise NotImplementedError(f"Only STFT is supported now, got {tfr_type}")

        # Backward-compatible TFR normalization handling
        tfr_normalization = representation_cfg.get("tfr_normalization")
        if tfr_normalization is None:
            tfr_normalization = "zscore" if representation_cfg.get("normalize_tfr", False) else "none"

        transforms.append(
            AddSTFT(
                n_fft=representation_cfg["n_fft"],
                hop_length=representation_cfg["hop_length"],
                win_length=representation_cfg["win_length"],
                log_amplitude=representation_cfg["log_amplitude"],
                tfr_normalization=tfr_normalization,
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