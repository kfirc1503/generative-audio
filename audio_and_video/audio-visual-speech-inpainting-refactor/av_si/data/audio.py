"""STFT + log-magnitude pipeline (Morrone §3.2)."""
from typing import Optional

import torch
import torchaudio


SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 384      # 24 ms
HOP_LENGTH = 192      # 12 ms
N_FREQ_BINS = N_FFT // 2 + 1   # 257
EPS = 1e-6


def _hann(device=None):
    return torch.hann_window(WIN_LENGTH, periodic=True, device=device)


def stft(waveform: torch.Tensor) -> torch.Tensor:
    """waveform: (..., samples) → complex STFT (..., T, F)."""
    spec = torch.stft(
        waveform,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        window=_hann(waveform.device),
        center=True,
        return_complex=True,
    )
    return spec.transpose(-1, -2)


def log_magnitude(complex_spec: torch.Tensor) -> torch.Tensor:
    return torch.log(complex_spec.abs() + EPS)


def inv_log_magnitude(log_mag: torch.Tensor) -> torch.Tensor:
    return torch.exp(log_mag) - EPS


def normalize(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (features - mean) / std


def denormalize(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return features * std + mean


def istft_with_phase(magnitude: torch.Tensor, phase: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
    """Reconstruct waveform from linear magnitude + phase (both (..., T, F))."""
    spec = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase)).transpose(-1, -2)
    return torch.istft(
        spec,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        window=_hann(magnitude.device),
        length=length,
        center=True,
    )


def griffin_lim(magnitude: torch.Tensor, n_iter: int = 32, length: Optional[int] = None) -> torch.Tensor:
    """Estimate phase from linear magnitude. Replaces upstream's LWS step."""
    transform = torchaudio.transforms.GriffinLim(
        n_fft=N_FFT,
        n_iter=n_iter,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        window_fn=lambda n: torch.hann_window(n, periodic=True),
        length=length,
    ).to(magnitude.device)
    return transform(magnitude.transpose(-1, -2))
