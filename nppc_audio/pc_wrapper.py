import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from FullSubNet_plus.speech_enhance.audio_zen.acoustics.mask import decompress_cIRM


def gram_schmidt_to_crm(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Gram-Schmidt orthogonalization to CRM directions.
    Direct adaptation of NPPC's implementation for complex CRMs.

    Args:
        x: Input tensor of shape [B, n_dirs, 2, F, T]

    Returns:
        Orthogonalized tensor of the same shape
    """
    # Combine real and imaginary into complex numbers
    x_complex = torch.complex(x[:, :, 0], x[:, :, 1])  # [B, n_dirs, F, T]

    # Reshape to [B, n_dirs, F*T]
    B, n_dirs, F, T = x_complex.shape
    x = x_complex.reshape(B, n_dirs, -1)  # [B, n_dirs, F*T]

    x_orth = []
    proj_vec_list = []

    for i in range(n_dirs):
        w = x[:, i]  # [B, F*T]

        # Project onto all previous vectors
        for w2 in proj_vec_list:
            w = w - w2 * torch.sum(w.conj() * w2, dim=1, keepdim=True)

        # Normalize
        w_hat = w.detach() / w.detach().norm(dim=1, keepdim=True)

        x_orth.append(w)
        proj_vec_list.append(w_hat)

    # Stack and reshape back
    out = torch.stack(x_orth, dim=1).reshape(B, n_dirs, F, T)
    return torch.stack([out.real, out.imag], dim=2)


class AudioPCWrapper(nn.Module):
    def __init__(
            self,
            net: nn.Module,
            project_func: Optional[callable] = None,
    ):
        """
        Wrapper for MultiDirectionFullSubNet_Plus that handles CRM directions.

        Args:
            net: MultiDirectionFullSubNet_Plus network
            project_func: Optional projection function
        """
        super().__init__()
        self.net = net
        self.n_dirs = net.n_directions
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

