import torch
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from pathlib import Path
import pydantic
from nppc_audio.inpainting.networks.unet import RestorationWrapper, UNetConfig, UNet
from utils import preprocess_log_magnitude
from dataset.audio_dataset_inpainting import AudioInpaintingDataset
# from nppc_audio.inpainting.validator.config.schema import ModelValidatorConfig
import utils



def restore_pred_spec_using_clean(pred_norm_log_mag, mean, std, clean_spec):
    """
    Restore the predicted spectrogram using the phase of the clean spectrogram.

    Args:
        pred_norm_log_mag: Normalized log-magnitude spectrogram [B, 1, F, T]
        mean: Mean used for normalization
        std: Standard deviation used for normalization
        clean_spec: Clean complex spectrogram [B, F, T]

    Returns:
        pred_spec: Reconstructed spectrogram [B, F, T]
    """
    # Unnormalize the log-magnitude
    pred_log_mag = pred_norm_log_mag * std + mean

    # Convert from log-magnitude (dB) to magnitude
    pred_mag = 10 ** (pred_log_mag / 20.0)

    # Extract the phase from the clean spectrogram
    clean_phase = torch.angle(clean_spec)  # [B, F, T]

    # Create the complex-valued spectrogram
    pred_spec = pred_mag.squeeze(1) * torch.exp(1j * clean_phase)

    return pred_spec


def plot_spectrograms_and_error(clean_spec, masked_spec, output_mag, mask, sample_len_seconds,mean,std):
    """Plot spectrograms and reconstruction error"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Set dB range for spectrograms
    vmin, vmax = -3, 3
    vmin_err , vmax_err = 0,3

    # Clean spectrogram
    # clean_mag = torch.abs(clean_spec[0, 0, :, :] + 1j * clean_spec[0, 1, :, :])
    # clean_mag_db = 20 * torch.log10(clean_mag + 1e-8)
    clean_mag_db = clean_spec[0,0,:,:]
    im = axs[0, 0].imshow(clean_mag_db.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[0, sample_len_seconds, 0, clean_mag_db.shape[0]])
    axs[0, 0].set_title('Clean Spectrogram')
    plt.colorbar(im, ax=axs[0, 0])

    # Masked spectrogram
    # masked_mag = torch.abs(masked_spec[0, 0, :, :] + 1j * masked_spec[0, 1, :, :])
    # masked_mag_db = 20 * torch.log10(masked_mag + 1e-
    masked_mag_db = masked_spec[0,0,:,:]
    im = axs[0, 1].imshow(masked_mag_db.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[0, sample_len_seconds, 0, masked_mag_db.shape[0]])
    axs[0, 1].set_title('Masked Spectrogram')
    plt.colorbar(im, ax=axs[0, 1])

    # Model output spectrogram
    # output_mag = torch.abs(output_spec[0, 0, :, :] + 1j * output_spec[0, 1, :, :])
    mask = mask[0,0,:,:]

    output_mag_db = output_mag[0,0,:,:]
    im = axs[1, 0].imshow(output_mag_db.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[0, sample_len_seconds, 0, output_mag_db.shape[0]])
    axs[1, 0].set_title('Model Output Spectrogram')
    plt.colorbar(im, ax=axs[1, 0])

    # Error plot (difference between clean and output)
    # vmin_error = -3
    # vmax_error = 3
    error = torch.abs(clean_mag_db - output_mag_db)
    error_db_relevant_part = error[mask == 0]
    error_db_relevant_part = error_db_relevant_part.reshape(error_db_relevant_part.shape[-1]//error.shape[0],error.shape[0])
    im = axs[1, 1].imshow(error_db_relevant_part.numpy(), origin='lower', aspect='auto', vmin=vmin_err, vmax=vmax_err,
                          extent=[0, sample_len_seconds, 0, error.shape[0]])
    axs[1, 1].set_title('Reconstruction Error (dB)')
    plt.colorbar(im, ax=axs[1, 1])

    real = clean_spec[:, 0, :, :]  # Real part
    imag = clean_spec[:, 1, :, :]  # Imaginary part
    clean_complex_spec = torch.complex(real, imag)  # Combine to complex-valued spectrogram
    pred_spec = restore_pred_spec_using_clean(output_mag, mean, std, clean_complex_spec)
    err_spec = clean_spec - pred_spec



    plt.tight_layout()
    return fig


class InpaintingModelValidatorConfig(pydantic.BaseModel):
    checkpoint_path: str
    device: str = "cuda"
    save_dir: str = "validation_results"
    model_configuration: UNetConfig


class InpaintingModelValidator:
    def __init__(self, config: InpaintingModelValidatorConfig):
        self.config = config
        self.device = config.device
        if config.device == 'cuda':
            # check if gpu is exist
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint_path = Path(config.checkpoint_path).absolute()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # self.model = RestorationWrapper(checkpoint['model'])
        base_net = UNet(config.model_configuration)
        base_net.load_state_dict(checkpoint["model_state_dict"])
        base_net.to(self.device)
        base_net.eval()
        self.model = RestorationWrapper(base_net)
        # # might not be needed, but just in case
        # self.model.to(self.device)
        # self.model.eval()

    def validate_sample(self, masked_spec, mask, clean_spec, sample_len_seconds):
        """Validate model on a single sample"""
        with torch.no_grad():
            # Move inputs to device
            masked_spec = masked_spec.to(self.device)
            mask = mask.to(self.device)
            clean_spec = clean_spec.to(self.device)
            # turn it into normalized log amplitude
            # clean_spec_mag = torch.sqrt(clean_spec[:, 0, :, :] ** 2 + clean_spec[:, 1, :, :] ** 2)
            # clean_spec_mag = clean_spec_mag.unsqueeze(1)
            clean_spec_mag = torch.sqrt(clean_spec[:, 0, :, :] ** 2 + clean_spec[:, 1, :, :] ** 2)
            clean_spec_mag = clean_spec_mag.unsqueeze(1)
            _, mean, std = preprocess_log_magnitude(clean_spec_mag)


            clean_spec_mag_norm_log, mask, masked_spec_mag_log = utils.preprocess_data(clean_spec, masked_spec, mask)
            output_log_mag_normalized = self.model(masked_spec_mag_log, mask)

            # Get model output
            # output = self.model(masked_spec, mask)

            # Calculate errors
            # TODO calculate the mse and the mae only on the gap area like the loss training
            omask = 1 - mask
            mse_gap = torch.sum(((output_log_mag_normalized - clean_spec_mag_norm_log) ** 2) * omask)
            mse_gap = mse_gap / torch.sum(omask)
            mse_gap = mse_gap.item()

            # Plot results
            fig = plot_spectrograms_and_error(clean_spec_mag_norm_log.cpu(), masked_spec_mag_log.cpu(),
                                              output_log_mag_normalized.cpu(), mask.cpu(), sample_len_seconds)

            return {
                'mse': mse_gap,
                'figure': fig,
                'output': output_log_mag_normalized.cpu()
            }

# @hydra.main(version_base=None, config_path="../scripts/config", config_name="config")
# def main(cfg: DictConfig):
#     # Create validator config
#     validator_config = ModelValidatorConfig(
#         checkpoint_path=str(Path(cfg.checkpoint_dir) / "your_checkpoint.pt"),  # Update with actual checkpoint name
#         device=cfg.inpainting_training_configuration.device,
#         save_dir="validation_results"
#     )
#
#     # Create validator
#     validator = ModelValidator(validator_config)
#
#     # Create dataset
#     dataset = AudioInpaintingDataset(cfg.inpainting_training_configuration.data_configuration)
#
#     # Get a sample
#     masked_spec, mask, clean_spec = dataset[0]
#     masked_spec = masked_spec.unsqueeze(0)  # Add batch dimension
#     mask = mask.unsqueeze(0)
#     clean_spec = clean_spec.unsqueeze(0)
#
#     # Validate
#     results = validator.validate_sample(
#         masked_spec,
#         mask,
#         clean_spec,
#         cfg.inpainting_training_configuration.data_configuration.sub_sample_length_seconds
#     )
#
#     print(f"MSE: {results['mse']:.6f}")
#     print(f"MAE: {results['mae']:.6f}")
#
#     # Save figure
#     save_dir = Path(validator_config.save_dir)
#     save_dir.mkdir(exist_ok=True)
#     results['figure'].savefig(save_dir / "spectrogram_comparison.png")
#     plt.close()
#
#
# if __name__ == "__main__":
#     main()
