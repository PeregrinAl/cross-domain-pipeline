from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SampleMeta:
    path: str
    label: int
    domain: str
    record_id: str
    split: str


class SignalWindowDataset(Dataset):
    """
    Dataset for pre-saved signal windows (.npy) described by a manifest .csv file.

    Expected manifest columns:
    - path
    - label
    - domain
    - record_id
    - split
    """

    REQUIRED_COLUMNS = {"path", "label", "domain", "record_id", "split"}

    def __init__(
        self,
        manifest_path: str,
        split: Optional[str] = None,
        domain: Optional[str] = None,
        transform: Optional[Callable] = None,
        return_meta: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.return_meta = return_meta

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)

        missing = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")

        if split is not None:
            self.df = self.df[self.df["split"] == split].copy()

        if domain is not None:
            self.df = self.df[self.df["domain"] == domain].copy()

        self.df = self.df.reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(
                f"No samples found for split={split}, domain={domain} in {manifest_path}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        window_path = Path(row["path"])
        if not window_path.exists():
            raise FileNotFoundError(f"Window file not found: {window_path}")

        x = np.load(window_path).astype(np.float32)

        # expected shape: [L]
        # convert to [1, L] for 1D CNN / TCN compatibility
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        elif x.ndim != 2:
            raise ValueError(
                f"Unexpected window shape {x.shape} in file {window_path}. "
                f"Expected [L] or [C, L]."
            )

        x = torch.from_numpy(x)
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        sample = {
            "x_raw": x,
            "label": label,
            "domain": row["domain"],
            "record_id": row["record_id"],
            "split": row["split"],
            "path": str(window_path),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_meta:
            return sample

        return sample["x_raw"], sample["label"]