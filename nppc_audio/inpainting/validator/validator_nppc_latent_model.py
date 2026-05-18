import torch
import matplotlib.pyplot as plt
import pydantic
from pathlib import Path
from nppc_audio.inpainting.nppc.nppc_model import NPPCModel, NPPCModelConfig
from nppc_audio.inpainting.networks.unet import LatentEncoder, LatentEncoderConfig
from nppc_audio.inpainting.nppc.pc_wrapper import gram_schmidt_to_spec_mag
import utils
import numpy as np
import torch.nn.functional as F


def plot_pc_spectrograms_latent(masked_spec, clean_spec, pred_spec_mag, pc_variations_spec, mask, sample_len_seconds,
                                metadata, save_dir, sample_idx, max_dirs=None):
    """
    Plot spectrograms, error, and PC variations for latent space model
    
    Args:
        pc_variations_spec: Pre-computed variations [n_dirs, n_alphas, B, 1, F, T]
    """
    # Create directory for individual spectrograms
    sample_dir = Path(save_dir) / f"sample_{sample_idx}" / "spectrograms"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # pc_variations_spec shape: [n_dirs, n_alphas, B, 1, F, T]
    n_dirs = pc_variations_spec.shape[0]
    n_alphas = pc_variations_spec.shape[1]
    
    if max_dirs is not None:
        n_dirs = min(n_dirs, max_dirs)
        pc_variations_spec = pc_variations_spec[:n_dirs]

    alphas = torch.arange(-3, 3.5, 0.5)
    n_cols = n_alphas + 1
    fig, axs = plt.subplots(1 + n_dirs, n_cols, figsize=(3 * n_cols, 3 * (1 + n_dirs)))

    vmin, vmax = -3, 3
    vmin_err, vmax_err = 0, 3

    # Get mask indices in spec domain from metadata
    spec_start_idx = metadata['mask_start_frame_idx'][0]
    spec_end_idx = metadata['mask_end_frame_idx'][0] + 1

    # Calculate context window (same duration as mask on each side)
    mask_duration = spec_end_idx - spec_start_idx
    context_start_idx = max(0, spec_start_idx - mask_duration)
    context_end_idx = min(mask.shape[-1], spec_end_idx + mask_duration)

    # Calculate time extent for plotting
    time_per_column = sample_len_seconds / mask.shape[-1]
    plot_start_time = context_start_idx * time_per_column
    plot_end_time = context_end_idx * time_per_column

    def save_individual_spectrogram(data, title, filename):
        """Helper function to save individual spectrograms"""
        plt.rcParams.update({'font.size': 14})

        fig_single, ax = plt.subplots(figsize=(10, 6))

        # Calculate frequency values for y-axis
        sample_rate = 16000  # Hz
        n_fft = 255  # FFT size
        n_freq_bins = data.shape[0]
        freqs = np.linspace(0, sample_rate / 2, n_freq_bins)

        # Plot spectrogram with frequency y-axis
        im = ax.imshow(data, origin='lower', aspect='auto',
                      vmin=vmin if 'error' not in filename else vmin_err,
                      vmax=vmax if 'error' not in filename else vmax_err,
                      extent=[plot_start_time, plot_end_time, freqs[0], freqs[-1]])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=12)

        # Add vertical lines to show mask region
        ax.axvline(x=spec_start_idx * time_per_column, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=spec_end_idx * time_per_column, color='r', linestyle='--', alpha=0.5)

        # Set axis labels
        ax.set_xlabel('Time (s)', fontsize=18)
        ax.set_ylabel('Frequency (kHz)', fontsize=18)

        # Set y-ticks at meaningful frequencies
        yticks = np.arange(0, sample_rate / 2 + 1, 2000)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{int(f / 1000)}' for f in yticks], fontsize=12)

        # Set x-ticks font size
        ax.tick_params(axis='x', labelsize=12)

        plt.tight_layout()
        fig_single.savefig(sample_dir / filename)
        plt.close(fig_single)

    # Clean spectrogram
    clean_mag_db = clean_spec[0, 0, :, :]
    mask = mask.squeeze(1).squeeze(0)
    im = axs[0, 0].imshow(clean_mag_db[:, context_start_idx:context_end_idx].numpy(),
                          origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[plot_start_time, plot_end_time, 0, clean_mag_db.shape[0]])
    axs[0, 0].set_title('Clean Spectrogram')
    plt.colorbar(im, ax=axs[0, 0])
    save_individual_spectrogram(
        clean_mag_db[:, context_start_idx:context_end_idx].numpy(),
        '',
        'clean_spec.png'
    )

    # Masked spectrogram
    masked_mag_db = masked_spec[0, 0, :, :]
    im = axs[0, 1].imshow(masked_mag_db[:, context_start_idx:context_end_idx].numpy(),
                          origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[plot_start_time, plot_end_time, 0, masked_mag_db.shape[0]])
    axs[0, 1].set_title('Masked Spectrogram')
    plt.colorbar(im, ax=axs[0, 1])
    save_individual_spectrogram(
        masked_mag_db[:, context_start_idx:context_end_idx].numpy(),
        '',
        'masked_spec.png'
    )

    # Model output spectrogram
    output_mag_db = pred_spec_mag[0, 0, :, :]
    im = axs[0, 2].imshow(output_mag_db[:, context_start_idx:context_end_idx].numpy(),
                          origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[plot_start_time, plot_end_time, 0, pred_spec_mag.shape[2]])
    axs[0, 2].set_title('Model Output Spectrogram')
    plt.colorbar(im, ax=axs[0, 2])
    save_individual_spectrogram(
        output_mag_db[:, context_start_idx:context_end_idx].numpy(),
        '',
        'output_spec.png'
    )

    # Plot error
    error_db = torch.abs(clean_mag_db - output_mag_db)
    im = axs[0, 3].imshow(error_db[:, context_start_idx:context_end_idx].numpy(),
                          origin='lower', aspect='auto', vmin=vmin_err, vmax=vmax_err,
                          extent=[plot_start_time, plot_end_time, 0, error_db.shape[0]])
    axs[0, 3].set_title('Reconstruction Error (dB)')
    plt.colorbar(im, ax=axs[0, 3])
    save_individual_spectrogram(
        error_db[:, context_start_idx:context_end_idx].numpy(),
        '',
        'error_spec.png'
    )

    # Plot clean and output specs with context
    im = axs[0, 4].imshow(clean_mag_db[:, context_start_idx:context_end_idx].numpy(),
                          origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[plot_start_time, plot_end_time, 0, clean_mag_db.shape[0]])
    axs[0, 4].set_title('Clean Spec (Inpainting Region)')
    plt.colorbar(im, ax=axs[0, 4])
    axs[0, 4].axvline(x=spec_start_idx * time_per_column, color='r', linestyle='--', alpha=0.5)
    axs[0, 4].axvline(x=spec_end_idx * time_per_column, color='r', linestyle='--', alpha=0.5)

    im = axs[0, 5].imshow(output_mag_db[:, context_start_idx:context_end_idx].numpy(),
                          origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[plot_start_time, plot_end_time, 0, output_mag_db.shape[0]])
    axs[0, 5].set_title('Output Spec (Inpainting Region)')
    plt.colorbar(im, ax=axs[0, 5])
    axs[0, 5].axvline(x=spec_start_idx * time_per_column, color='r', linestyle='--', alpha=0.5)
    axs[0, 5].axvline(x=spec_end_idx * time_per_column, color='r', linestyle='--', alpha=0.5)

    # Remove remaining subplots in first row
    for j in range(6, n_cols):
        axs[0, j].remove()

    # Plot PC variations (pre-computed in latent space)
    for i in range(n_dirs):
        row_idx = i + 1
        
        # Get the base prediction (alpha=0, which is at index 6 for alphas from -3 to 3)
        # alphas = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
        base_idx = 6  # alpha = 0
        base_variation = pc_variations_spec[i, base_idx, 0, 0]  # [F, T]
        
        # Show the base variation (alpha=0) as the "PC direction reference"
        im = axs[row_idx, 0].imshow(base_variation[:, context_start_idx:context_end_idx].numpy(),
                                    origin='lower', aspect='auto',
                                    vmin=vmin, vmax=vmax,
                                    extent=[plot_start_time, plot_end_time, 0, base_variation.shape[0]])
        axs[row_idx, 0].set_title(f'PC Direction {i + 1} (α=0)')
        plt.colorbar(im, ax=axs[row_idx, 0])
        axs[row_idx, 0].axvline(x=spec_start_idx * time_per_column, color='r', linestyle='--', alpha=0.5)
        axs[row_idx, 0].axvline(x=spec_end_idx * time_per_column, color='r', linestyle='--', alpha=0.5)

        # Save individual PC direction spectrogram
        save_individual_spectrogram(
            base_variation[:, context_start_idx:context_end_idx].numpy(),
            '',
            f'pc_direction_{i + 1}.png'
        )

        # Plot all variations for this PC direction
        for j, alpha in enumerate(alphas):
            # Get pre-computed variation
            modified_spec = pc_variations_spec[i, j, 0, 0]  # [F, T]
            im = axs[row_idx, j + 1].imshow(modified_spec[:, context_start_idx:context_end_idx].numpy(),
                                            origin='lower', aspect='auto',
                                            vmin=vmin, vmax=vmax,
                                            extent=[plot_start_time, plot_end_time, 0, modified_spec.shape[0]])
            axs[row_idx, j + 1].set_title(f'PC{i + 1} (α={alpha:.1f})')
            plt.colorbar(im, ax=axs[row_idx, j + 1])
            axs[row_idx, j + 1].axvline(x=spec_start_idx * time_per_column, color='r', linestyle='--', alpha=0.5)
            axs[row_idx, j + 1].axvline(x=spec_end_idx * time_per_column, color='r', linestyle='--', alpha=0.5)

            # Save individual PC variation spectrogram
            save_individual_spectrogram(
                modified_spec[:, context_start_idx:context_end_idx].numpy(),
                f'PC{i + 1} (α={alpha:.1f})',
                f'pc{i + 1}_alpha_{alpha:.1f}.png'
            )

    plt.tight_layout()
    return fig


class NPPCLatentModelValidatorConfig(pydantic.BaseModel):
    checkpoint_path: str
    device: str = "cuda"
    save_dir: str = "validation_nppc_latent_results"
    nppc_latent_model_configuration: LatentEncoderConfig
    nppc_model_configuration: NPPCModelConfig
    max_dirs_to_plot: int = None


class NPPCLatentModelValidator:
    def __init__(self, config: NPPCLatentModelValidatorConfig):
        self.config = config
        self.device = config.device
        if config.device == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load latent model checkpoint
        checkpoint_path = Path(config.checkpoint_path).absolute()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Initialize latent space NPPC model (LatentEncoder)
        self.nppc_latent_model = LatentEncoder(config.nppc_latent_model_configuration)
        self.nppc_latent_model.load_state_dict(checkpoint["model_state_dict"])
        self.nppc_latent_model.to(self.device)
        self.nppc_latent_model.eval()

        # Initialize the NPPC model (which loads restoration model from wandb)
        self.nppc_model = NPPCModel(config.nppc_model_configuration)
        self.nppc_model.to(self.device)
        self.nppc_model.eval()

        # Get the restoration model for encoding/decoding
        self.restoration_model = self.nppc_model.pretrained_restoration_model
        self.restoration_model.eval()

    def validate_sample(self, masked_spec, mask, clean_spec, masked_audio, metadata, sample_len_seconds, sample_idx):
        """Validate latent space model on a single sample"""
        with torch.no_grad():
            # Move inputs to device
            masked_spec = masked_spec.to(self.device)
            mask = mask.to(self.device)
            clean_spec = clean_spec.to(self.device)

            # Preprocess data
            clean_spec_mag_norm_log, mask, masked_spec_mag_log, mean, std = utils.preprocess_data(
                clean_spec, masked_spec, mask, plot_mean_std=True
            )

            # Get restoration model prediction
            pred_spec_mag_log = self.restoration_model(masked_spec_mag_log, 1-mask)
            #pred_spec_mag_log = self.restoration_model(masked_spec_mag_log, mask)

            # transfer pred spec mag log to the cpu for debug reasons:
            pred_spec_mag_log_cpu = pred_spec_mag_log.cpu()
            masked_spec_mag_log_cpu = masked_spec_mag_log.cpu()
            mask_cpu = mask.cpu()


            # Encode to latent space using the restoration model's encoder
            latent_restored, skip_connections_restored = self.restoration_model.net.encoder(pred_spec_mag_log)
            latent_distorted, skip_connections_distorted = self.restoration_model.net.encoder(masked_spec_mag_log)

            # Prepare input for latent NPPC model (concatenate distorted and restored)
            masked_with_pred = torch.cat((masked_spec_mag_log, pred_spec_mag_log), dim=1)
            broadcasted_mask = mask.view(1, 1, mask.shape[-2], mask.shape[-1]).expand(
                masked_spec_mag_log.shape[0], 1, mask.shape[-2], mask.shape[-1]
            )

            # Get PC directions in latent space
            # Note: The latent model expects inverted mask (1 - mask)
            w_mat_latent, latent_mask = self.nppc_latent_model(masked_with_pred, 1 - broadcasted_mask)
            # w_mat_latent shape: [B, n_dirs, C, H, W] where C is latent channels (e.g., 512)
            # latent_mask shape: [B, 1, H, W]
            
            print(f"w_mat_latent from model - min: {w_mat_latent.min():.4f}, max: {w_mat_latent.max():.4f}, mean: {w_mat_latent.mean():.4f}")

            # For validation, we DON'T apply masking - we want to see the full PC directions
            # Just apply Gram-Schmidt normalization
            B, n_dirs, C, H, W = w_mat_latent.shape
            w_latent_flat = w_mat_latent.flatten(start_dim=2)  # [B, n_dirs, C*H*W]
            
            # Apply Gram-Schmidt normalization
            w_mat_latent_flat = gram_schmidt_to_spec_mag(w_latent_flat)
            w_mat_latent = w_mat_latent_flat.view(B, n_dirs, C, H, W)
            
            print(f"w_mat_latent after gram-schmidt - min: {w_mat_latent.min():.4f}, max: {w_mat_latent.max():.4f}, mean: {w_mat_latent.mean():.4f}")
            
            # Check the magnitude of latent_restored for comparison
            print(f"latent_restored - min: {latent_restored.min():.4f}, max: {latent_restored.max():.4f}, mean: {latent_restored.mean():.4f}, std: {latent_restored.std():.4f}")
            print(f"Ratio of PC direction magnitude to latent magnitude: {w_mat_latent.std():.4f} / {latent_restored.std():.4f} = {(w_mat_latent.std() / latent_restored.std()):.4f}")

            # Generate variations by adding PC directions in LATENT space, then decoding
            # This is the CORRECT approach (same as MNIST validator):
            # 1. Add scaled direction to latent representation
            # 2. Decode the modified latent
            # NOT: decode directions separately and add them (mathematically incorrect!)
            
            # Generate variations for visualization at different scales
            alphas = torch.arange(-3, 3.5, 0.5).to(self.device)
            
            # Store all variations: [n_dirs, n_alphas, B, 1, F, T]
            pc_variations_spec = []
            
            for dir_idx in range(w_mat_latent.shape[1]):
                direction_variations = []
                for alpha_idx, alpha in enumerate(alphas):
                    # CORRECT: Add scaled direction to latent representation FIRST
                    latent_var = latent_restored + alpha * w_mat_latent[:, dir_idx, :, :]
                    
                    # Debug: print latent changes for first direction
                    if dir_idx == 0 and alpha_idx % 3 == 0:  # Print every 3rd alpha
                        latent_diff = (latent_var - latent_restored).abs().mean()
                        print(f"  PC{dir_idx+1}, α={alpha:.1f}: latent change = {latent_diff:.6f}")
                    if alpha == 0 :
                        latent_diff = (latent_var - latent_restored).abs().mean()
                        print(latent_diff)

                    
                    # Then decode the modified latent
                    # Use skip connections from the DISTORTED input (like MNIST validator does!)
                    # decoded_output = self.restoration_model.net.decoder(latent_var, skip_connections_distorted)
                    decoded_output = self.restoration_model.net.decoder(latent_var, skip_connections_restored)

                    # IMPORTANT: The restoration model does: output = input + network_output * mask
                    # The decoder gives us the network_output, so we need to add the residual like the model does
                    # But we inverted the mask convention, so we need to use (1-mask) here
                    # spec_var = masked_spec_mag_log + decoded_output * (1 - mask)
                    spec_var = decoded_output
                    
                    # Debug: check decoded output changes for first direction
                    if dir_idx == 0 and alpha_idx % 3 == 0:
                        spec_diff = (spec_var - pred_spec_mag_log).abs().mean()
                        print(f"    Final spec change from pred = {spec_diff:.6f}")
                    
                    direction_variations.append(spec_var)
                
                pc_variations_spec.append(torch.stack(direction_variations))
            
            # Stack all directions: [n_dirs, n_alphas, B, 1, F, T]
            pc_variations_spec = torch.stack(pc_variations_spec)
            
            print(f"Generated {pc_variations_spec.shape[0]} PC directions with {pc_variations_spec.shape[1]} variations each")

            # Plot results
            spec_save_path = Path(self.config.save_dir) / "spec_variations"
            fig = plot_pc_spectrograms_latent(
                masked_spec_mag_log.cpu(),
                clean_spec_mag_norm_log.cpu(),
                pred_spec_mag_log.cpu(),
                pc_variations_spec.cpu(),
                mask.cpu(),
                sample_len_seconds,
                metadata,
                spec_save_path,
                sample_idx,
                max_dirs=self.config.max_dirs_to_plot
            )

            return {
                'figure': fig,
                'pc_variations': pc_variations_spec.cpu()
            }

