import torch
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from pathlib import Path
import pydantic
from nppc_audio.inpainting.networks.unet import RestorationWrapper , UNetConfig
from dataset.audio_dataset_inpainting import AudioInpaintingDataset
#from nppc_audio.inpainting.validator.config.schema import ModelValidatorConfig


def plot_spectrograms_and_error(clean_spec, masked_spec, output_spec, sample_len_seconds):
    """Plot spectrograms and reconstruction error"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Set dB range for spectrograms
    vmin, vmax = -120, 20

    # Clean spectrogram
    clean_mag = torch.abs(clean_spec[0, 0, :, :] + 1j * clean_spec[0, 1, :, :])
    clean_mag_db = 20 * torch.log10(clean_mag + 1e-8)
    im = axs[0, 0].imshow(clean_mag_db.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[0, sample_len_seconds, 0, clean_mag.shape[0]])
    axs[0, 0].set_title('Clean Spectrogram')
    plt.colorbar(im, ax=axs[0, 0])

    # Masked spectrogram
    masked_mag = torch.abs(masked_spec[0, 0, :, :] + 1j * masked_spec[0, 1, :, :])
    masked_mag_db = 20 * torch.log10(masked_mag + 1e-8)
    im = axs[0, 1].imshow(masked_mag_db.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[0, sample_len_seconds, 0, masked_mag.shape[0]])
    axs[0, 1].set_title('Masked Spectrogram')
    plt.colorbar(im, ax=axs[0, 1])

    # Model output spectrogram
    output_mag = torch.abs(output_spec[0, 0, :, :] + 1j * output_spec[0, 1, :, :])
    output_mag_db = 20 * torch.log10(output_mag + 1e-8)
    im = axs[1, 0].imshow(output_mag_db.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[0, sample_len_seconds, 0, output_mag.shape[0]])
    axs[1, 0].set_title('Model Output Spectrogram')
    plt.colorbar(im, ax=axs[1, 0])

    # Error plot (difference between clean and output)
    error = torch.abs(clean_mag - output_mag)
    error_db = 20 * torch.log10(error + 1e-8)
    im = axs[1, 1].imshow(error_db.numpy(), origin='lower', aspect='auto',
                          extent=[0, sample_len_seconds, 0, error.shape[0]])
    axs[1, 1].set_title('Reconstruction Error (dB)')
    plt.colorbar(im, ax=axs[1, 1])

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
        checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
        # self.model = RestorationWrapper(checkpoint['model'])
        self.model = RestorationWrapper(self.config.model_configuration)
        # self.model = RestorationWrapper(checkpoint['model_config']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def validate_sample(self, masked_spec, mask, clean_spec, sample_len_seconds):
        """Validate model on a single sample"""
        with torch.no_grad():
            # Move inputs to device
            masked_spec = masked_spec.to(self.device)
            mask = mask.to(self.device)
            clean_spec = clean_spec.to(self.device)

            # Get model output
            output = self.model(masked_spec, mask)

            # Calculate errors
            mse = torch.nn.functional.mse_loss(output, clean_spec).item()
            mae = torch.nn.functional.l1_loss(output, clean_spec).item()

            # Plot results
            fig = plot_spectrograms_and_error(clean_spec.cpu(), masked_spec.cpu(),
                                              output.cpu(), sample_len_seconds)

            return {
                'mse': mse,
                'mae': mae,
                'figure': fig,
                'output': output.cpu()
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