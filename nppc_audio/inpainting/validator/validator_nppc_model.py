import torch
import matplotlib.pyplot as plt
import pydantic
from pathlib import Path
from nppc_audio.inpainting.nppc.nppc_model import NPPCModel, NPPCModelConfig
from utils import preprocess_log_magnitude
import utils
import numpy as np
import torchaudio


def save_pc_audio_variations(clean_spec_mag_norm_log, pred_spec_mag, pc_directions_mag, clean_spec, mask, alphas,
                             save_dir, mean, std, sample_idx,
                             n_fft=255, hop_length=128, sample_rate=16000):
    """
    Save audio variations for each PC direction and alpha value.

    Args:
        clean_spec_mag_norm_log: Normalized log magnitude of clean spectrogram [1, 1, F, T]
        pred_spec_mag: Predicted magnitude spectrogram [1, 1, F, T]
        pc_directions_mag: PC directions tensor [1, n_dirs, F, T]
        clean_spec: Clean complex spectrogram [1, 2, F, T]
        mask: Binary mask [1, 1, F, T]
        alphas: Tensor of alpha values
        save_dir: Directory to save audio files
        mean: Mean value used in log magnitude normalization
        std: Standard deviation used in log magnitude normalization
        sample_idx: Sample identifier for unique path
        n_fft: FFT size
        hop_length: Hop length for STFT
        sample_rate: Audio sample rate
    """
    save_dir = Path(save_dir) / f"sample_{sample_idx}"
    save_dir.mkdir(parents=True, exist_ok=True)
    clean_spec_ref_complex = torch.complex(clean_spec[0,0], clean_spec[0,1])
    # Get phase from clean spectrogram
    clean_phase = torch.angle(clean_spec_ref_complex)
    window = torch.hann_window(n_fft).to(clean_phase.device)

    # Save clean audio using normalized log magnitude
    clean_mag_log = clean_spec_mag_norm_log[0, 0] * std + mean
    clean_mag_linear = torch.exp(clean_mag_log) -1e-6
    real_part = clean_mag_linear * torch.cos(clean_phase)
    imag_part = clean_mag_linear * torch.sin(clean_phase)
    complex_clean_spec = torch.complex(real_part, imag_part)

    clean_audio = torch.istft(complex_clean_spec,
                              n_fft=n_fft,
                              hop_length=hop_length,
                              win_length=n_fft,
                              window=window)
    clean_audio_ref = torch.istft(clean_spec_ref_complex,
                                  n_fft=n_fft,
                                  hop_length=hop_length,
                                  win_length=n_fft,
                                  window=window)

    clean_path = save_dir / "clean.wav"
    torchaudio.save(clean_path, clean_audio.unsqueeze(0), sample_rate=sample_rate)

    clean_ref_path = save_dir / "clean_ref.wav"
    torchaudio.save(clean_ref_path, clean_audio_ref.unsqueeze(0), sample_rate=sample_rate)

    # For each PC direction
    for i in range(pc_directions_mag.shape[1]):
        pc_dir = save_dir / f"pc_{i + 1}"
        pc_dir.mkdir(exist_ok=True)

        pc_dir_db = pc_directions_mag[0, i]

        # For each alpha value
        for alpha in alphas:
            # Modify the predicted magnitude with PC direction
            modified_mag = pred_spec_mag[0, 0] + alpha * pc_dir_db

            # Denormalize the log magnitude
            modified_mag_log = modified_mag * std + mean

            # Convert to linear scale
            modified_mag_linear = torch.exp(modified_mag_log)

            # Create complex spectrogram using magnitude and clean phase
            real_part = modified_mag_linear * torch.cos(clean_phase)
            imag_part = modified_mag_linear * torch.sin(clean_phase)
            modified_complex = torch.complex(real_part, imag_part)

            # Convert to audio using inverse STFT
            audio = torch.istft(modified_complex,
                                n_fft=n_fft,
                                hop_length=hop_length,
                                win_length=n_fft,
                                window=window)

            # Save audio file
            audio_path = pc_dir / f"alpha_{alpha:.1f}.wav"
            torchaudio.save(audio_path, audio.unsqueeze(0), sample_rate=sample_rate)



def plot_pc_spectrograms(masked_spec, clean_spec, pred_spec_mag, pc_directions_mag, mask, sample_len_seconds,
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

    # Define alpha values for PC direction scaling
    alphas = torch.arange(-3, 3.5, 0.5)
    n_alphas = len(alphas)

    # Create figure with subplots for each alpha value
    n_cols = n_alphas + 1
    fig, axs = plt.subplots(1 + n_dirs, n_cols, figsize=(3 * n_cols, 3 * (1 + n_dirs)))

    # Set dB range for spectrograms
    vmin, vmax = -3, 3
    vmin_err, vmax_err = 0, 3

    # Clean spectrogram
    clean_mag_db = clean_spec[0, 0, :, :]
    mask = mask.squeeze(1).squeeze(0)
    im = axs[0, 0].imshow(clean_mag_db.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[0, sample_len_seconds, 0, clean_mag_db.shape[0]])
    axs[0, 0].set_title('Clean Spectrogram')
    plt.colorbar(im, ax=axs[0, 0])

    # Masked spectrogram
    masked_mag_db = masked_spec[0, 0, :, :]
    im = axs[0, 1].imshow(masked_mag_db.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[0, sample_len_seconds, 0, masked_mag_db.shape[0]])
    axs[0, 1].set_title('Masked Spectrogram')
    plt.colorbar(im, ax=axs[0, 1])

    # Model output spectrogram
    output_mag_db = pred_spec_mag[0, 0, :, :]
    im = axs[0, 2].imshow(output_mag_db.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[0, sample_len_seconds, 0, pred_spec_mag.shape[2]])
    axs[0, 2].set_title('Model Output Spectrogram')
    plt.colorbar(im, ax=axs[0, 2])

    # Plot error (difference between clean and predicted)
    error_db = torch.abs(clean_mag_db - output_mag_db)
    error_db_relevant_part = error_db[mask == 0]
    error_db_relevant_part = error_db_relevant_part.reshape(error_db_relevant_part.shape[-1] // error_db.shape[0],
                                                            error_db.shape[0])
    im = axs[0, 3].imshow(error_db_relevant_part.numpy(), origin='lower', aspect='auto', vmin=vmin_err, vmax=vmax_err,
                          extent=[0.4, 0.528, 0, error_db.shape[0]])
    axs[0, 3].set_title('Reconstruction Error (dB)')
    plt.colorbar(im, ax=axs[0, 3])

    clean_spec_mag_zoom = clean_mag_db[mask == 0]
    clean_spec_mag_zoom = clean_spec_mag_zoom.reshape(clean_spec_mag_zoom.shape[-1] // clean_mag_db.shape[0],
                                                      clean_mag_db.shape[0])
    im = axs[0, 4].imshow(clean_spec_mag_zoom.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[0.4, 0.528, 0, clean_mag_db.shape[0]])
    axs[0, 4].set_title('inpainting part clean spec')
    plt.colorbar(im, ax=axs[0, 4])

    output_spec_mag_zoom = output_mag_db[mask == 0]
    output_spec_mag_zoom = output_spec_mag_zoom.reshape(output_spec_mag_zoom.shape[-1] // output_mag_db.shape[0],
                                                        output_mag_db.shape[0])
    im = axs[0, 5].imshow(output_spec_mag_zoom.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[0.4, 0.528, 0, output_mag_db.shape[0]])
    axs[0, 5].set_title('inpainting part predicted spec')
    plt.colorbar(im, ax=axs[0, 5])

    # Remove remaining subplots in first row
    for j in range(6, n_cols):
        axs[0, j].remove()

    # Plot each PC direction and its variations
    for i in range(n_dirs):
        row_idx = i + 1

        # Get PC direction
        pc_dir_db = pc_directions_mag[0, i]
        pc_dir_db_relevant_part = pc_dir_db[mask == 0]
        pc_dir_db_relevant_part = pc_dir_db_relevant_part.reshape(
            pc_dir_db_relevant_part.shape[-1] // pc_dir_db.shape[0],
            pc_dir_db.shape[0])

        # Plot PC direction in first column
        im = axs[row_idx, 0].imshow(pc_dir_db_relevant_part.cpu().numpy(), origin='lower', aspect='auto',
                                    vmin=vmin, vmax=vmax,
                                    extent=[0.4, 0.528, 0, pc_directions_mag.shape[-2]])
        axs[row_idx, 0].set_title(f'PC Direction {i + 1} (dB)')
        plt.colorbar(im, ax=axs[row_idx, 0])

        # Plot variations for each alpha
        for j, alpha in enumerate(alphas):
            modified_error = torch.abs(error_db_relevant_part + alpha * pc_dir_db_relevant_part)
            im = axs[row_idx, j + 1].imshow(modified_error.cpu().numpy(), origin='lower', aspect='auto',
                                            vmin=vmin_err, vmax=vmax_err,
                                            extent=[0.4, 0.528, 0, modified_error.shape[0]])
            axs[row_idx, j + 1].set_title(f'α={alpha:.1f}')
            plt.colorbar(im, ax=axs[row_idx, j + 1])

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

    def validate_sample(self, masked_spec, mask, clean_spec, sample_len_seconds, sample_idx):
        """Validate model on a single sample"""
        with torch.no_grad():
            # Move inputs to device
            masked_spec = masked_spec.to(self.device)
            mask = mask.to(self.device)
            clean_spec = clean_spec.to(self.device)
            clean_spec_mag_norm_log, mask, masked_spec_mag_log, mean, std = utils.preprocess_data(clean_spec,
                                                                                                  masked_spec, mask,
                                                                                                  plot_mean_std=True)

            # Get PC directions from model
            pc_directions = self.model(masked_spec_mag_log, mask)
            pred_spec_mag_log = self.model.get_pred_spec_mag_norm(masked_spec_mag_log, mask)
            # Plot results
            fig = plot_pc_spectrograms(
                masked_spec_mag_log.cpu(),
                clean_spec_mag_norm_log.cpu(),
                pred_spec_mag_log.cpu(),
                pc_directions.cpu(),
                mask.cpu(),
                sample_len_seconds,
                max_dirs=self.config.max_dirs_to_plot
            )
            # Save audio variations
            audio_save_path = Path(self.config.save_dir) / "audio_variations"
            alphas = torch.arange(-3, 3.5, 0.5)
            save_pc_audio_variations(
                clean_spec_mag_norm_log,
                pred_spec_mag_log.cpu(),
                pc_directions.cpu(),
                clean_spec.cpu(),
                mask.cpu(),
                alphas,
                audio_save_path,
                mean,
                std,
                sample_idx
            )

            return {
                'figure': fig,
                'pc_directions': pc_directions.cpu()
            }
