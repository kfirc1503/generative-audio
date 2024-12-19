import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import torch
import torchaudio
import numpy as np
from dataset.audio_dataset_inpainting import AudioInpaintingConfig, AudioInpaintingDataset


def plot_spectrograms(clean_audio: torch.Tensor, masked_audio: torch.Tensor, mask: torch.Tensor):
    """Plot spectrograms of clean and masked audio, along with the mask."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Compute spectrograms
    n_fft = 512
    hop_length = 256

    # Calculate time axis (in seconds)
    audio_len_samples = clean_audio.shape[1]
    duration = audio_len_samples / 16000  # assuming 16kHz sample rate

    # For spectrograms
    spec_time = np.linspace(0, duration, num=int(np.ceil(audio_len_samples / hop_length)))
    spec_freq = np.linspace(0, 8000, n_fft // 2 + 1)  # Frequency axis up to Nyquist (8kHz)

    # Plot clean spectrogram
    spec_clean = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2
    )(clean_audio)
    axes[0].imshow(torch.log10(spec_clean.squeeze() + 1e-8),
                   aspect='auto', origin='lower',
                   extent=[0, duration, 0, 8000])
    axes[0].set_title('Clean Audio Spectrogram')
    axes[0].set_ylabel('Frequency (Hz)')

    # Plot masked spectrogram
    spec_masked = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2
    )(masked_audio)
    axes[1].imshow(torch.log10(spec_masked.squeeze() + 1e-8),
                   aspect='auto', origin='lower',
                   extent=[0, duration, 0, 8000])
    axes[1].set_title('Masked Audio Spectrogram')
    axes[1].set_ylabel('Frequency (Hz)')

    # Plot mask
    time_axis = np.linspace(0, duration, mask.shape[1])
    axes[2].plot(time_axis, mask.squeeze().numpy())
    axes[2].set_title('Mask')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True)

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