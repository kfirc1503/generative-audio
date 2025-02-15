import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import torch
from dataset.audio_dataset_inpainting import AudioInpaintingConfig, AudioInpaintingDataset


def plot_all_spectrograms(masked_spec, spec_mask, time_mask, clean_spec, clean_audio, masked_audio, stft_config,
                          sample_len_seconds):
    """Plot spectrograms and masks for comparison"""
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))

    # Create STFT window
    window = torch.hann_window(stft_config.win_length)

    # Set dB range
    vmin, vmax = -120, 20

    # 1. Direct STFT output spectrograms
    # Clean spectrogram magnitude
    clean_mag = torch.abs(clean_spec[0, :, :] + 1j * clean_spec[1, :, :])
    clean_mag_db = 20 * torch.log10(clean_mag + 1e-8)
    im = axs[0, 0].imshow(clean_mag_db.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[0, sample_len_seconds, 0, clean_mag.shape[0]])
    axs[0, 0].set_title('Clean Spectrogram Magnitude (Direct)')
    plt.colorbar(im, ax=axs[0, 0])

    # Masked spectrogram magnitude
    masked_mag = torch.abs(masked_spec[0, :, :] + 1j * masked_spec[1, :, :])
    masked_mag_db = 20 * torch.log10(masked_mag + 1e-8)
    im = axs[0, 1].imshow(masked_mag_db.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[0, sample_len_seconds, 0, masked_mag.shape[0]])
    axs[0, 1].set_title('Masked Spectrogram Magnitude (Direct)')
    plt.colorbar(im, ax=axs[0, 1])

    # 2. Time-domain to STFT spectrograms
    # Compute spectrograms from time-domain signals using same STFT params
    clean_spec_from_audio = torch.stft(
        clean_audio,
        n_fft=stft_config.nfft,
        hop_length=stft_config.hop_length,
        win_length=stft_config.win_length,
        window=window,
        return_complex=False
    )
    clean_spec_from_audio = clean_spec_from_audio.squeeze(0)
    clean_spec_from_audio = clean_spec_from_audio.permute(2, 0, 1)
    clean_mag_from_audio = torch.abs(clean_spec_from_audio[0] + 1j * clean_spec_from_audio[1])
    clean_mag_from_audio_db = 20 * torch.log10(clean_mag_from_audio + 1e-8)

    im = axs[1, 0].imshow(clean_mag_from_audio_db.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[0, sample_len_seconds, 0, clean_mag_from_audio.shape[0]])
    axs[1, 0].set_title('Clean Spectrogram (from time-domain)')
    plt.colorbar(im, ax=axs[1, 0])

    masked_spec_from_audio = torch.stft(
        masked_audio,
        n_fft=stft_config.nfft,
        hop_length=stft_config.hop_length,
        win_length=stft_config.win_length,
        window=window,
        return_complex=False
    )
    masked_spec_from_audio = masked_spec_from_audio.squeeze(0)
    masked_spec_from_audio = masked_spec_from_audio.permute(2, 0, 1)
    masked_mag_from_audio = torch.abs(masked_spec_from_audio[0] + 1j * masked_spec_from_audio[1])
    masked_mag_from_audio_db = 20 * torch.log10(masked_mag_from_audio + 1e-8)

    im = axs[1, 1].imshow(masked_mag_from_audio_db.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                          extent=[0, sample_len_seconds, 0, masked_mag_from_audio.shape[0]])
    axs[1, 1].set_title('Masked Spectrogram (from time-domain)')
    plt.colorbar(im, ax=axs[1, 1])

    # 3. Masks
    time_points = torch.linspace(0, sample_len_seconds, spec_mask.shape[0])
    axs[2, 0].plot(time_points, spec_mask.numpy())
    axs[2, 0].set_title('STFT Domain Mask')
    axs[2, 0].set_ylim(-0.1, 1.1)
    axs[2, 0].grid(True)

    time_points = torch.linspace(0, sample_len_seconds, time_mask.shape[1])
    axs[2, 1].plot(time_points, time_mask.squeeze(0).numpy())
    axs[2, 1].set_title('Time Domain Mask')
    axs[2, 1].set_ylim(-0.1, 1.1)
    axs[2, 1].grid(True)

    plt.tight_layout()
    return fig


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Test the AudioInpaintingDataset and visualize results"""

    # Convert Hydra config to Pydantic model
    config = AudioInpaintingConfig(**cfg.inpainting_dataset)

    # Create dataset
    dataset = AudioInpaintingDataset(config)
    print(f"Dataset size: {len(dataset)}")

    # Get spectrograms from dataset
    stft_masked, mask_frames, stft_clean, masked_audio = dataset[0]

    masked_audio, mask, clean_audio = dataset.get_audio_and_time_mask(0)
    # Print shapes
    print("\nTensor shapes:")
    print(f"Clean audio: {clean_audio.shape}")
    print(f"Masked audio: {masked_audio.shape}")
    print(f"Masked spectrogram: {mask_frames.shape}")
    print(f"STFT mask: {stft_masked.shape}")
    print(f"Time mask: {mask.shape}")
    print(f"Clean spectrogram: {stft_clean.shape}")

    # Plot spectrograms and masks
    fig = plot_all_spectrograms(
        stft_masked,
        mask_frames,
        mask,
        stft_clean,
        clean_audio,
        masked_audio,
        config.stft_configuration,
        config.sub_sample_length_seconds  # Pass STFT config
    )
    plt.show()

    # # Verify mask properties
    # assert torch.all((spec_mask >= 0) & (spec_mask <= 1)), "Spec mask values should be between 0 and 1"
    # assert torch.all((time_mask >= 0) & (time_mask <= 1)), "Time mask values should be between 0 and 1"
    #
    # # Verify masking is working
    # masked_regions = (spec_mask == 0)
    # assert torch.all(masked_spec[:, masked_regions[0]] == 0), "Masked regions should be zero"
    #
    # print("\nAll tests passed!")


if __name__ == "__main__":
    main()
