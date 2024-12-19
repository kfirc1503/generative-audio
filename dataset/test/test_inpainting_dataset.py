import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import torch
import torchaudio
from dataset.audio_dataset_inpainting import AudioInpaintingConfig, AudioInpaintingDataset


def plot_spectrograms(clean_audio: torch.Tensor, masked_audio: torch.Tensor, mask: torch.Tensor):
    """Plot spectrograms of clean and masked audio, along with the mask."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Compute spectrograms
    n_fft = 512
    hop_length = 256

    # Plot clean spectrogram
    spec_clean = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2
    )(clean_audio)
    axes[0].imshow(torch.log10(spec_clean.squeeze() + 1e-8),
                   aspect='auto', origin='lower')
    axes[0].set_title('Clean Audio Spectrogram')
    axes[0].set_ylabel('Frequency Bin')

    # Plot masked spectrogram
    spec_masked = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2
    )(masked_audio)
    axes[1].imshow(torch.log10(spec_masked.squeeze() + 1e-8),
                   aspect='auto', origin='lower')
    axes[1].set_title('Masked Audio Spectrogram')
    axes[1].set_ylabel('Frequency Bin')

    # Plot mask
    axes[2].plot(mask.squeeze().numpy())
    axes[2].set_title('Mask')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.show()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Test the AudioInpaintingDataset and visualize results"""

    # Convert Hydra config to Pydantic model
    config = AudioInpaintingConfig(**cfg.inpainting_dataset)

    # Create dataset
    dataset = AudioInpaintingDataset(config)
    print(f"Dataset size: {len(dataset)}")

    # Get a sample
    masked_audio, mask, clean_audio = dataset[0]

    # Plot spectrograms
    plot_spectrograms(
        clean_audio=clean_audio,
        masked_audio=masked_audio,
        mask=mask
    )


if __name__ == "__main__":
    main()