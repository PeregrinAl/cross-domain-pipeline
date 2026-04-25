from typing import Dict, Optional

import numpy as np
import torch
import math


class NormalizeRaw:
    """
    Per-window normalization for raw signal.

    Supported modes:
    - "none"
    - "zscore"
    - "robust"

    Expects sample["x_raw"] with shape [C, L].
    """

    def __init__(self, mode: str = "zscore", eps: float = 1e-8):
        self.mode = mode
        self.eps = eps

        allowed = {"none", "zscore", "robust"}
        if self.mode not in allowed:
            raise ValueError(f"Unsupported raw normalization mode: {self.mode}")

    def __call__(self, sample: Dict) -> Dict:
        x = sample["x_raw"]  # [C, L]

        if self.mode == "none":
            return sample

        if self.mode == "zscore":
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            sample["x_raw"] = (x - mean) / (std + self.eps)
            return sample

        if self.mode == "robust":
            median = x.median(dim=-1, keepdim=True).values

            q1 = torch.quantile(x, 0.25, dim=-1, keepdim=True)
            q3 = torch.quantile(x, 0.75, dim=-1, keepdim=True)
            iqr = q3 - q1

            sample["x_raw"] = (x - median) / (iqr + self.eps)
            return sample

        return sample


class AddSTFT:
    """
    Builds STFT magnitude representation from sample["x_raw"].

    Input:
    sample["x_raw"] shape [1, L] or [C, L]

    Output:
    sample["x_tfr"] shape [1, F, T]
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: Optional[int] = None,
        log_amplitude: bool = True,
        power: float = 1.0,
        tfr_normalization: str = "none",
        normalize_tfr: Optional[bool] = None,
        eps: float = 1e-8,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.log_amplitude = log_amplitude
        self.power = power

        if normalize_tfr is not None:
            tfr_normalization = "zscore" if normalize_tfr else "none"

        self.tfr_normalization = tfr_normalization
        self.eps = eps

        allowed = {"none", "zscore", "robust"}
        if self.tfr_normalization not in allowed:
            raise ValueError(
                f"Unsupported TFR normalization mode: {self.tfr_normalization}"
            )

    def _normalize_tfr(self, mag: torch.Tensor) -> torch.Tensor:
        if self.tfr_normalization == "none":
            return mag

        if self.tfr_normalization == "zscore":
            mag_mean = mag.mean()
            mag_std = mag.std()
            return (mag - mag_mean) / (mag_std + self.eps)

        if self.tfr_normalization == "robust":
            mag_flat = mag.reshape(-1)
            mag_median = mag_flat.median()
            q1 = torch.quantile(mag_flat, 0.25)
            q3 = torch.quantile(mag_flat, 0.75)
            iqr = q3 - q1
            return (mag - mag_median) / (iqr + self.eps)

        return mag

    def __call__(self, sample: Dict) -> Dict:
        x = sample["x_raw"]  # [C, L]

        if x.ndim != 2:
            raise ValueError(f'x_raw must have shape [C, L], got {x.shape}')

        # For MVP take first channel
        x_1d = x[0]  # [L]

        window = torch.hann_window(
            self.win_length,
            device=x_1d.device,
            dtype=x_1d.dtype,
        )

        stft = torch.stft(
            x_1d,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )  # [F, T]

        mag = stft.abs()

        if self.power != 1.0:
            mag = mag.pow(self.power)

        if self.log_amplitude:
            mag = torch.log1p(mag)

        mag = self._normalize_tfr(mag)

        sample["x_tfr"] = mag.unsqueeze(0)  # [1, F, T]
        return sample

class AddCWT:
    """
    Builds a lightweight CWT-like scalogram from sample["x_raw"].

    Input:
        sample["x_raw"] shape [C, L]

    Output:
        sample["x_tfr"] shape [1, S, T]
    """

    def __init__(
        self,
        num_scales: int = 64,
        min_scale: float = 1.0,
        max_scale: float = 64.0,
        hop_length: int = 64,
        w0: float = 6.0,
        log_amplitude: bool = True,
        tfr_normalization: str = "none",
        eps: float = 1e-8,
    ):
        self.num_scales = num_scales
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.hop_length = hop_length
        self.w0 = w0
        self.log_amplitude = log_amplitude
        self.tfr_normalization = tfr_normalization
        self.eps = eps

        allowed = {"none", "zscore", "robust"}
        if self.tfr_normalization not in allowed:
            raise ValueError(
                f"Unsupported TFR normalization mode: {self.tfr_normalization}"
            )

    def _normalize_tfr(self, mag: torch.Tensor) -> torch.Tensor:
        if self.tfr_normalization == "none":
            return mag

        if self.tfr_normalization == "zscore":
            return (mag - mag.mean()) / (mag.std() + self.eps)

        if self.tfr_normalization == "robust":
            mag_flat = mag.reshape(-1)
            median = mag_flat.median()
            q1 = torch.quantile(mag_flat, 0.25)
            q3 = torch.quantile(mag_flat, 0.75)
            iqr = q3 - q1
            return (mag - median) / (iqr + self.eps)

        return mag

    def __call__(self, sample: Dict) -> Dict:
        x = sample["x_raw"]

        if x.ndim != 2:
            raise ValueError(f"x_raw must have shape [C, L], got {x.shape}")

        # MVP: use first channel, same as AddSTFT
        x_1d = x[0]
        n = x_1d.numel()

        x_fft = torch.fft.rfft(x_1d)
        freqs = torch.fft.rfftfreq(
            n,
            d=1.0,
            device=x_1d.device,
            dtype=x_1d.dtype,
        )

        omega = 2.0 * math.pi * freqs

        scales = torch.linspace(
            self.min_scale,
            self.max_scale,
            self.num_scales,
            device=x_1d.device,
            dtype=x_1d.dtype,
        )

        # Morlet-like frequency-domain filters
        filters = torch.exp(
            -0.5 * ((scales[:, None] * omega[None, :]) - self.w0).pow(2)
        )

        coeffs = torch.fft.irfft(
            x_fft.unsqueeze(0) * filters,
            n=n,
            dim=-1,
        )

        mag = coeffs.abs()

        if self.hop_length > 1:
            mag = mag[:, :: self.hop_length]

        if self.log_amplitude:
            mag = torch.log1p(mag)

        mag = self._normalize_tfr(mag)

        sample["x_tfr"] = mag.unsqueeze(0)
        return sample
    
    
class Compose:
    """
    Simple transform composition.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample: Dict) -> Dict:
        for transform in self.transforms:
            sample = transform(sample)
        return sample
    

class ApplyPreprocessor:
    """
    Applies a preprocessing method to sample["x_raw"] before representation transforms.
    Expects sample["x_raw"] with shape [C, L].
    """

    def __init__(self, name: str, params: dict | None = None):
        from src.preprocessing import build_preprocessor

        self.name = name
        self.preprocessor = build_preprocessor(name, params or {})

    def __call__(self, sample: Dict) -> Dict:
        x = sample["x_raw"]

        if not torch.is_tensor(x):
            raise TypeError(f"x_raw must be a torch.Tensor, got {type(x)}")

        if x.ndim != 2:
            raise ValueError(f"x_raw must have shape [C, L], got {x.shape}")

        dtype = x.dtype
        device = x.device

        x_np = x.detach().cpu().numpy()

        processed_channels = []
        metadata = {
            "domain": sample.get("domain"),
            "record_id": sample.get("record_id"),
            "split": sample.get("split"),
            "path": sample.get("path"),
        }

        for channel in x_np:
            processed = self.preprocessor(channel, metadata=metadata)
            processed_channels.append(processed)

        x_processed_np = np.stack(processed_channels, axis=0)

        x_processed = torch.as_tensor(
            x_processed_np,
            dtype=dtype,
            device=device,
        )

        sample["x_raw"] = x_processed
        return sample