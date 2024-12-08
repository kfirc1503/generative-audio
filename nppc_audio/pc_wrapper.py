import pydantic
import torch
import torch.nn as nn
from FullSubNet_plus.speech_enhance.audio_zen.acoustics.mask import decompress_cIRM
from nppc_audio.networks import MultiDirectionConfig , MultiDirectionFullSubNet_Plus


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

class AudioPCWrapperConfig(pydantic.BaseModel):
    multi_direction_configuration:MultiDirectionConfig

    def make_instance(self):
        # Create and return an instance of Model using this config
        return AudioPCWrapper(self)


class AudioPCWrapper(nn.Module):
    def __init__(
            self,
            audio_pc_wrapper_config: AudioPCWrapperConfig,
    ):
        """
        Wrapper for MultiDirectionFullSubNet_Plus that handles CRM directions.

        Args:
            net: MultiDirectionFullSubNet_Plus network
        """
        super().__init__()
        #self.net = audio_pc_wrapper_config.multi_direction_configuration.make_instance()
        self.net = MultiDirectionFullSubNet_Plus(audio_pc_wrapper_config.multi_direction_configuration)
        self.n_dirs = self.net.n_directions

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

        # Permute to match FullSubNet+ format [B, F, T, 2*n_dirs]
        crm = crm.permute(0, 2, 3, 1)

        # Decompress CRM
        crm = decompress_cIRM(crm)

        # Permute back to our working format [B, 2*n_dirs, F, T]
        crm = crm.permute(0, 3, 1, 2)

        # Reshape to separate directions and real/imag components
        batch_size, _, freq_bins, time_steps = crm.shape
        crm = crm.reshape(batch_size, self.n_dirs, 2, freq_bins, time_steps)

        # Apply Gram-Schmidt orthogonalization
        w_mat = gram_schmidt_to_crm(crm)

        return w_mat



