import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from FullSubNet_plus.speech_enhance.audio_zen.acoustics.mask import decompress_cIRM


def gram_schmidt_to_crm(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Gram-Schmidt orthogonalization to CRM directions.

    Args:
        x: Input tensor of shape [B, n_dirs, 2, F, T] where:
           B: batch size
           n_dirs: number of directions
           2: real and imaginary components
           F: frequency bins
           T: time steps

    Returns:
        Orthogonalized tensor of the same shape
    """
    # Combine real and imaginary into complex numbers for orthogonalization
    x_complex = torch.complex(x[:, :, 0], x[:, :, 1])  # [B, n_dirs, F, T]

    # Reshape to [B*F*T, n_dirs]
    x_flat = x_complex.permute(0, 2, 3, 1).reshape(-1, x_complex.shape[1])

    # Gram-Schmidt process
    basis = torch.zeros_like(x_flat)
    basis[:, 0] = x_flat[:, 0] / torch.norm(x_flat[:, 0:1] + 1e-8, dim=1)

    for i in range(1, x_flat.shape[1]):
        v = x_flat[:, i]
        # Project v onto previous basis vectors
        projections = torch.sum(basis[:, :i].conj() * v.unsqueeze(1), dim=1, keepdim=True) * basis[:, :i]
        # Subtract projections and normalize
        w = v.unsqueeze(1) - torch.sum(projections, dim=1, keepdim=True)
        basis[:, i] = w.squeeze() / (torch.norm(w + 1e-8, dim=1))

    # Reshape back and split into real/imaginary components
    basis = basis.reshape(x.shape[0], x.shape[3], x.shape[4], x.shape[1])  # [B, F, T, n_dirs]
    basis = basis.permute(0, 3, 1, 2)  # [B, n_dirs, F, T]

    return torch.stack([basis.real, basis.imag], dim=2)  # [B, n_dirs, 2, F, T]


class AudioPCWrapper(nn.Module):
    def __init__(
            self,
            net: nn.Module,
            pre_net: Optional[nn.Module] = None,
            project_func: Optional[callable] = None,
    ):
        """
        Wrapper for MultiDirectionFullSubNet_Plus that handles CRM directions.

        Args:
            net: MultiDirectionFullSubNet_Plus network
            pre_net: Optional preprocessing network
            project_func: Optional projection function
        """
        super().__init__()
        self.net = net
        self.n_dirs = net.n_directions
        self.pre_net = pre_net
        self.project_func = project_func

    def forward(self, noisy_mag, noisy_real, noisy_imag,
                enhanced_mag=None, enhanced_real=None, enhanced_imag=None):
        """
        Forward pass computing orthogonal CRM directions.

        Args:
            noisy_mag: Noisy magnitude spectrogram [B, F, T]
            noisy_real: Real part of noisy spectrogram [B, F, T]
            noisy_imag: Imaginary part of noisy spectrogram [B, F, T]
            enhanced_mag: Optional enhanced magnitude spectrogram
            enhanced_real: Optional real part of enhanced spectrogram
            enhanced_imag: Optional imaginary part of enhanced spectrogram

        Returns:
            w_mat: Orthogonalized CRM directions tensor [B, n_dirs, 2, F, T]
        """
        # Get predictions from base network (returns [B, 2*n_dirs, F, T])
        crm = self.net(noisy_mag, noisy_real, noisy_imag,
                       enhanced_mag, enhanced_real, enhanced_imag)

        # Decompress CRM if using compression
        crm = decompress_cIRM(crm)

        # Reshape to separate directions and real/imag components
        batch_size, _, freq_bins, time_steps = crm.shape
        crm = crm.reshape(batch_size, self.n_dirs, 2, freq_bins, time_steps)

        # Apply optional projection
        if self.project_func is not None:
            crm = self.project_func(crm)

        # Apply Gram-Schmidt orthogonalization
        w_mat = gram_schmidt_to_crm(crm)

        return w_mat

