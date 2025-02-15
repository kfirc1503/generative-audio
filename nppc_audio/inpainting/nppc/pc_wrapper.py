import pydantic
import torch
import torch.nn as nn
from nppc_audio.inpainting.networks.unet import UNetConfig,UNet,RestorationWrapper

#
# def gram_schmidt_to_spec_mag(x: torch.Tensor) -> torch.Tensor:
#     """
#     Apply Gram-Schmidt orthogonalization to spectrogram magnitudes.
#
#     Args:
#         x: Input tensor of shape [B, n_dirs, F, T] (real-valued spectrogram magnitudes)
#
#     Returns:
#         Orthogonalized tensor of the same shape
#     """
#     # Reshape to [B, n_dirs, F*T]
#     B, n_dirs, F, T = x.shape
#     x = x.reshape(B, n_dirs, -1)  # [B, n_dirs, F*T]
#
#     x_orth = []
#     proj_vec_list = []
#
#     for i in range(n_dirs):
#         w = x[:, i]  # [B, F*T]
#
#         # Project onto all previous vectors
#         for w2 in proj_vec_list:
#             # Projection formula: w = w - (w · w2) * w2
#             w = w - w2 * torch.sum(w * w2, dim=1, keepdim=True) / (torch.sum(w2 * w2, dim=1, keepdim=True) + 1e-8)
#
#         # Normalize
#         w_hat = w / (w.norm(dim=1, keepdim=True) + 1e-8)  # Add small epsilon for numerical stability
#
#         x_orth.append(w)
#         proj_vec_list.append(w_hat)
#
#     # Stack and reshape back
#     out = torch.stack(x_orth, dim=1).reshape(B, n_dirs, F, T)
#     return out


def gram_schmidt_to_spec_mag(x):
    x_shape = x.shape
    x = x.flatten(2)

    x_orth = []
    proj_vec_list = []
    for i in range(x.shape[1]):
        w = x[:, i, :]
        for w2 in proj_vec_list:
            w = w - w2 * torch.sum(w * w2, dim=-1, keepdim=True)
        w_hat = w.detach() / w.detach().norm(dim=-1, keepdim=True)

        x_orth.append(w)
        proj_vec_list.append(w_hat)

    x_orth = torch.stack(x_orth, dim=1).view(*x_shape)
    return x_orth

class AudioInpaintingPCWrapperConfig(pydantic.BaseModel):
    model_configuration: UNetConfig
    n_dirs: int

class AudioInpaintingPCWrapper(nn.Module):
    def __init__(self, pc_wrapper_config: AudioInpaintingPCWrapperConfig):
        super().__init__()
        self.config  = pc_wrapper_config
        base_net = UNet(self.config.model_configuration)
        # self.net = RestorationWrapper(base_net)
        self.net = base_net



    def forward(self, mag_spec: torch.Tensor, mask: torch.Tensor):
        # Get predictions from base network (returns [B, n_dirs, F, T])
        alternatives_pred = self.net(mag_spec)
        mask_broadcasted = mask
        if alternatives_pred.shape[1] > 1:  # If x_in has more than 1 channel (K > 1)
            mask_broadcasted = mask_broadcasted.expand(-1, alternatives_pred.shape[1], -1,-1)  # Broadcast along the channel dimension
        # Apply inpainting
        alternatives_pred = alternatives_pred * (1 - mask_broadcasted)
        tmp = alternatives_pred.detach().cpu().numpy()
        # Apply Gram-Schmidt orthogonalization
        w_mat = gram_schmidt_to_spec_mag(alternatives_pred)
        tmp2 = w_mat.detach().cpu().numpy()

        return w_mat