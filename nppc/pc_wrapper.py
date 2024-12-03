import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import numpy as np
import pydantic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class PCWrapperConfig(pydantic.BaseModel):
    base_net_configuration: nn.Module
    n_dirs: int
    base_net: nn.Module




class PCWrapper(nn.Module):
    def __init__(
            self,
            net,
            pre_net=None,
            n_dirs=5,
            offset=0.0,
            scale=1.0,
            project_func=None,
            n_fft=512,
            hop_length=256,
            win_length=512,
    ):
        """PC Wrapper for audio enhancement models.

        Args:
            net: Base network (FullSubNet+)
            pre_net: Optional preprocessing network
            n_dirs: Number of principal components
            offset: Data normalization offset
            scale: Data normalization scale
            project_func: Projection function
            n_fft: FFT size for STFT
            hop_length: Hop length for STFT
            win_length: Window length for STFT
        """
        super().__init__()
        self.net = net
        self.pre_net = pre_net
        self.n_dirs = n_dirs
        self.register_buffer('offset', torch.as_tensor(offset))
        self.register_buffer('scale', torch.as_tensor(scale))
        self.project_func = project_func if project_func is not None else lambda x: x

        # STFT parameters
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=None,
            return_complex=True
        )

        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length
        )

    def forward(self, x):
        """
        Forward pass through the PC wrapper.

        Args:
            x: Input audio waveform [batch, time]

        Returns:
            Tuple of (enhanced waveform, principal directions, projection weights)
        """
        # Convert to spectrogram and split real/imag
        spec = self.stft(x)
        x_spec = torch.stack([spec.real, spec.imag], dim=1)

        # Get batch size and normalize
        batch_size = x_spec.shape[0]
        x_norm = (x_spec - self.offset) / self.scale

        # Pre-network if available
        if self.pre_net is not None:
            x_pre = self.pre_net(x_norm)
        else:
            x_pre = x_norm

        # Get principal directions
        w = self.net(torch.cat([x_pre, x_norm], dim=1))
        w = w.view(batch_size, self.n_dirs, *x_norm.shape[1:])

        # Normalize directions
        w_norms = torch.norm(w.reshape(batch_size, self.n_dirs, -1), dim=2, keepdim=True)
        w = w / (w_norms.unsqueeze(-1).unsqueeze(-1) + 1e-8)

        # Project input onto directions
        x_flat = x_norm.reshape(batch_size, -1)
        w_flat = w.reshape(batch_size, self.n_dirs, -1)
        weights = torch.bmm(w_flat, x_flat.unsqueeze(-1)).squeeze(-1)

        # Reconstruct and apply projection
        x_proj = torch.bmm(weights.unsqueeze(1), w_flat).squeeze(1)
        x_proj = x_proj.view_as(x_norm)
        x_proj = self.project_func(x_proj)

        # Denormalize
        x_proj = x_proj * self.scale + self.offset

        # Convert back to waveform
        spec_proj = torch.complex(x_proj[:, 0], x_proj[:, 1])
        waveform = self.istft(spec_proj)

        return waveform, w, weights
