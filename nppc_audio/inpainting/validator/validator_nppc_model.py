import torch
import matplotlib.pyplot as plt
import pydantic
from pathlib import Path
from nppc_audio.inpainting.nppc.nppc_model import NPPCModel, NPPCModelConfig
from utils import preprocess_log_magnitude


def plot_pc_spectrograms(masked_spec, clean_spec, pred_spec_mag, pc_directions_mag, sample_len_seconds,
                         max_dirs=None):
    """
    Plot spectrograms, error, and PC directions

    Args:
        masked_spec: Preprocessed masked spectrogram
        mask: Binary mask
        clean_spec: Clean spectrogram
        pred_spec_mag: Predicted magnitude spectrogram
        pc_directions_mag: PC directions tensor [B, n_dirs, F, T]
        sample_len_seconds: Length of audio sample in seconds
        max_dirs: Maximum number of PC directions to plot (None for all)
    """
    n_dirs = pc_directions_mag.shape[1]
    if max_dirs is not None:
        n_dirs = min(n_dirs, max_dirs)
        pc_directions_mag = pc_directions_mag[:, :n_dirs]

    fig, axs = plt.subplots(4 + n_dirs, 1, figsize=(15, 5 * (4 + n_dirs)))

    # Set dB range for spectrograms
    vmin, vmax = -120, 20

    # Clean spectrogram
    clean_mag = torch.abs(clean_spec[0, 0, :, :] + 1j * clean_spec[0, 1, :, :])
    clean_mag_db = 20 * torch.log10(clean_mag + 1e-8)
    im = axs[0].imshow(clean_mag_db.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                       extent=[0, sample_len_seconds, 0, clean_mag.shape[0]])
    axs[0].set_title('Clean Spectrogram')
    plt.colorbar(im, ax=axs[0])

    # Masked spectrogram
    masked_mag = torch.abs(masked_spec[0, 0, :, :] + 1j * masked_spec[0, 1, :, :])
    masked_mag_db = 20 * torch.log10(masked_mag + 1e-8)
    im = axs[1].imshow(masked_mag_db.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                       extent=[0, sample_len_seconds, 0, masked_mag.shape[0]])
    axs[1].set_title('Masked Spectrogram')
    plt.colorbar(im, ax=axs[1])

    # Model output spectrogram
    output_mag_db = 20 * torch.log10(pred_spec_mag[0, 0, :, :] + 1e-8)
    im = axs[2].imshow(output_mag_db.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                       extent=[0, sample_len_seconds, 0, pred_spec_mag.shape[2]])
    axs[2].set_title('Model Output Spectrogram')
    plt.colorbar(im, ax=axs[2])

    # Plot error (difference between clean and predicted)
    error = torch.abs(clean_mag - pred_spec_mag[0, 0, :, :])
    error_db = 20 * torch.log10(error + 1e-8)
    im = axs[3].imshow(error_db.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                       extent=[0, sample_len_seconds, 0, error.shape[0]])
    axs[3].set_title('Reconstruction Error (dB)')
    plt.colorbar(im, ax=axs[3])

    # Plot each PC direction in dB scale
    for i in range(n_dirs):
        pc_dir_db = 20 * torch.log10(pc_directions_mag[0, i] + 1e-8)
        im = axs[i + 4].imshow(pc_dir_db.cpu().numpy(), origin='lower', aspect='auto',
                               vmin=vmin, vmax=vmax,
                               extent=[0, sample_len_seconds, 0, pc_directions_mag.shape[-2]])
        axs[i + 4].set_title(f'PC Direction {i + 1} (dB)')
        plt.colorbar(im, ax=axs[i + 4])

    plt.tight_layout()
    return fig


class NPPCModelValidatorConfig(pydantic.BaseModel):
    checkpoint_path: str
    device: str = "cuda"
    save_dir: str = "validation_nppc_results"
    model_configuration: NPPCModelConfig
    max_dirs_to_plot: int = None  # New parameter


class NPPCModelValidator:
    def __init__(self, config: NPPCModelValidatorConfig):
        self.config = config
        self.device = config.device
        if config.device == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint_path = Path(config.checkpoint_path).absolute()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Initialize model
        self.model = NPPCModel(config.model_configuration)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def validate_sample(self, masked_spec, mask, clean_spec, sample_len_seconds):
        """Validate model on a single sample"""
        with torch.no_grad():
            # Move inputs to device
            masked_spec = masked_spec.to(self.device)
            mask = mask.to(self.device)
            clean_spec = clean_spec.to(self.device)
            mask_broadcast = mask.unsqueeze(1).unsqueeze(2)
            mask_broadcast = mask_broadcast.expand(-1, 1, clean_spec.shape[2], -1)
            clean_spec_mag = torch.sqrt(clean_spec[:, 0, :, :] ** 2 + clean_spec[:, 1, :, :] ** 2)
            clean_spec_mag = clean_spec_mag.unsqueeze(1)
            clean_spec_mag_norm_log, mean, std = preprocess_log_magnitude(clean_spec_mag)
            masked_spec_mag_norm_log = clean_spec_mag_norm_log * mask_broadcast
            masked_spec_mag_log = masked_spec_mag_norm_log * std + mean
            clean_spec_mag_log = clean_spec_mag_norm_log * std + mean
            # Preprocess log magnitude

            # Get PC directions from model
            pc_directions = self.model(masked_spec_mag_norm_log, mask_broadcast)
            pred_spec_mag_log = self.model.get_pred_spec_mag_norm(masked_spec_mag_norm_log, mask_broadcast)
            # unnormalized:
            tmp = pc_directions.detach().cpu().numpy()
            pc_directions = pc_directions * std + mean
            pred_spec_mag_log = pred_spec_mag_log * std + mean
            pred_spec_mag = torch.exp(pred_spec_mag_log)
            pc_directions_mag = torch.exp(pc_directions)
            tmp2 = pc_directions_mag.detach().cpu().numpy()

            # Plot results
            fig = plot_pc_spectrograms(
                masked_spec.cpu(),
                clean_spec.cpu(),
                pred_spec_mag.cpu(),
                pc_directions_mag.cpu(),
                sample_len_seconds,
                max_dirs=self.config.max_dirs_to_plot
            )

            return {
                'figure': fig,
                'pc_directions': pc_directions.cpu()
            }
