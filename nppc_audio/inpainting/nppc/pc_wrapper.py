import pydantic
import torch
import torch.nn as nn
from nppc_audio.inpainting.networks.unet import UNetConfig,UNet,RestorationWrapper


def gram_schmidt_to_spec_mag(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Gram-Schmidt orthogonalization to spectrogram magnitudes.

    Args:
        x: Input tensor of shape [B, n_dirs, F, T] (real-valued spectrogram magnitudes)

    Returns:
        Orthogonalized tensor of the same shape
    """
    # Reshape to [B, n_dirs, F*T]
    B, n_dirs, F, T = x.shape
    x = x.reshape(B, n_dirs, -1)  # [B, n_dirs, F*T]

    x_orth = []
    proj_vec_list = []

    for i in range(n_dirs):
        w = x[:, i]  # [B, F*T]

        # Project onto all previous vectors
        for w2 in proj_vec_list:
            # Projection formula: w = w - (w · w2) * w2
            w = w - w2 * torch.sum(w * w2, dim=1, keepdim=True) / (torch.sum(w2 * w2, dim=1, keepdim=True) + 1e-8)

        # Normalize
        w_hat = w / (w.norm(dim=1, keepdim=True) + 1e-8)  # Add small epsilon for numerical stability

        x_orth.append(w)
        proj_vec_list.append(w_hat)

    # Stack and reshape back
    out = torch.stack(x_orth, dim=1).reshape(B, n_dirs, F, T)
    return out

class AudioInpaintingPCWrapperConfig(pydantic.BaseModel):
    model_configuration: UNetConfig
    n_dirs: int

class AudioInpaintingPCWrapper(nn.Module):
    def __init__(self, pc_wrapper_config: AudioInpaintingPCWrapperConfig):
        super().__init__()
        self.config  = pc_wrapper_config
        self.net = RestorationWrapper(self.config.model_configuration)



    def forward(self, mag_spec: torch.Tensor, mask: torch.Tensor):
        # Get predictions from base network (returns [B, n_dirs, F, T])
        alternatives_pred = self.net(mag_spec, mask)
        # Apply Gram-Schmidt orthogonalization
        w_mat = gram_schmidt_to_spec_mag(alternatives_pred)
        return w_mat