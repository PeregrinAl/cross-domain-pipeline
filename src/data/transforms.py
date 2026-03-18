from typing import Dict, Optional

import torch


class NormalizeRaw:
    """
    Per-window z-normalization for raw signal.
    Expects sample["x_raw"] with shape [C, L].
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, sample: Dict) -> Dict:
        x = sample["x_raw"]  # [C, L]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        sample["x_raw"] = (x - mean) / (std + self.eps)
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
        normalize_tfr: bool = False,
        eps: float = 1e-8,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.log_amplitude = log_amplitude
        self.power = power
        self.normalize_tfr = normalize_tfr
        self.eps = eps

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

        if self.normalize_tfr:
            mag_mean = mag.mean()
            mag_std = mag.std()
            mag = (mag - mag_mean) / (mag_std + self.eps)

        sample["x_tfr"] = mag.unsqueeze(0)  # [1, F, T]
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