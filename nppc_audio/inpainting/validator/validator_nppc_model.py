import torch
import matplotlib.pyplot as plt
import pydantic
from pathlib import Path
from nppc_audio.inpainting.nppc.nppc_model import NPPCModel, NPPCModelConfig
from utils import preprocess_log_magnitude, enable_dropout, compute_pca_and_importance_weights, calculate_unet_baseline
import json
import utils
import numpy as np
import torchaudio
import librosa
import whisper
from itertools import groupby

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, \
    Wav2Vec2PhonemeCTCTokenizer  # Added for phoneme recognition


def plot_pitch_comparison(audio_variations: dict, n_dirs: int = 5, sample_rate: int = 16000, save_dir=None,
                          sample_idx=None):
    """
    Plot pitch contours comparing clean audio with PC variations.
    First subplot shows clean reference, followed by one subplot per PC direction.

    Args:
        audio_variations: Dictionary containing clean audio and PC variations
        n_dirs: Number of PC directions to plot
        sample_rate: Audio sample rate in Hz
        save_dir: Directory to save individual plots (optional)
        sample_idx: Sample index for directory naming (optional)
    """
    # Create directory for individual pitch plots if save_dir is provided
    if save_dir is not None and sample_idx is not None:
        sample_dir = Path(save_dir) / f"sample_{sample_idx}" / "pitch_contours"
        sample_dir.mkdir(parents=True, exist_ok=True)

    # Get clean audio as reference
    clean_audio = audio_variations['clean']

    # Create figure with n_dirs + 1 subplots (clean + PC variations)
    fig, axes = plt.subplots(n_dirs + 1, 1, figsize=(15, 4 * (n_dirs + 1)))

    # Calculate clean pitch contour once
    clean_np = clean_audio.squeeze().numpy()
    if clean_np.ndim > 1:
        clean_np = clean_np[0]
    f0_clean, voiced_flag_clean, _ = librosa.pyin(
        clean_np,
        fmin=80,
        fmax=400,
        sr=sample_rate
    )
    times = librosa.times_like(f0_clean)

    # Plot clean reference in first subplot
    axes[0].plot(times, f0_clean, color='black', label='Clean', linewidth=2)
    axes[0].set_title('Clean Audio Pitch Contour')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_xlabel('Time (s)')
    axes[0].grid(True)
    axes[0].legend()

    # Create colormap for alpha variations
    unique_alphas = sorted(set(float(k.split('alpha')[-1]) for k in audio_variations.keys() if 'alpha' in k))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_alphas)))

    # Plot each PC direction
    for i in range(n_dirs):
        pc_num = i + 1
        ax = axes[i + 1]  # +1 because first subplot is clean reference

        # Plot clean reference
        ax.plot(times, f0_clean, color='black', label='Clean', linewidth=2)

        # Plot each alpha variation for this PC
        for alpha_idx, alpha in enumerate(unique_alphas):
            variation_key = f'pc{pc_num}_alpha{alpha:.1f}'
            if variation_key in audio_variations:
                audio = audio_variations[variation_key]
                audio_np = audio.squeeze().numpy()
                if audio_np.ndim > 1:
                    audio_np = audio_np[0]

                f0, voiced_flag, _ = librosa.pyin(
                    audio_np,
                    fmin=80,
                    fmax=400,
                    sr=sample_rate
                )

                ax.plot(times, f0, color=colors[alpha_idx],
                        label=f'α={alpha:.1f}', alpha=0.7)

        ax.set_title(f'PC Direction {pc_num} Pitch Contours')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save individual PC plot if save_dir is provided
        if save_dir is not None:
            fig_pc = plt.figure(figsize=(15, 4))  # Changed from (15, 4)
            ax_pc = fig_pc.add_subplot(111)

            # Plot clean reference
            ax_pc.plot(times, f0_clean, color='black', label='Clean', linewidth=2)

            # Plot all alpha variations for this PC
            legend_alphas = unique_alphas[::2]
            # legend_alphas = unique_alphas
            # First plot all lines without labels
            for alpha_idx, alpha in enumerate(unique_alphas):
                variation_key = f'pc{pc_num}_alpha{alpha:.1f}'
                if variation_key in audio_variations:
                    audio = audio_variations[variation_key]
                    audio_np = audio.squeeze().numpy()
                    if audio_np.ndim > 1:
                        audio_np = audio_np[0]

                    f0, voiced_flag, _ = librosa.pyin(
                        audio_np,
                        fmin=80,
                        fmax=400,
                        sr=sample_rate
                    )

                    # Use _nolegend_ for alphas we don't want in legend
                    if alpha in legend_alphas:
                        label = f'α={alpha:.1f}'
                    else:
                        label = '_nolegend_'

                    ax_pc.plot(times, f0, color=colors[alpha_idx],
                               label=label, alpha=0.7)

            # Set empty title and larger fonts
            ax_pc.set_title('')
            ax_pc.set_ylabel('Frequency (Hz)', fontsize=20)
            ax_pc.set_xlabel('Time (s)', fontsize=20)
            ax_pc.tick_params(axis='both', labelsize=18)
            ax_pc.grid(True)

            # # Add legend on the right side
            ax_pc.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

            # Adjust the plot to fill the figure properly
            plt.subplots_adjust(right=0.85, top=0.98, bottom=0.15, left=0.1)

            # Save figure
            fig_pc.savefig(sample_dir / f'pc_{pc_num}_pitch.png', bbox_inches='tight')
            plt.close(fig_pc)

    plt.tight_layout()
    return fig


# def plot_pc_spectrograms(masked_spec, clean_spec, pred_spec_mag, pc_directions_mag, mask, sample_len_seconds,
#                          metadata, max_dirs=None):
#     """
#     Plot spectrograms, error, and PC directions, focusing on the inpainting area and its surroundings
#     """
#     n_dirs = pc_directions_mag.shape[1]
#     if max_dirs is not None:
#         n_dirs = min(n_dirs, max_dirs)
#         pc_directions_mag = pc_directions_mag[:, :n_dirs]
#
#     alphas = torch.arange(-3, 3.5, 0.5)
#     n_alphas = len(alphas)
#     n_cols = n_alphas + 1
#     fig, axs = plt.subplots(1 + n_dirs, n_cols, figsize=(3 * n_cols, 3 * (1 + n_dirs)))
#
#     vmin, vmax = -3, 3
#     vmin_err, vmax_err = 0, 3
#
#     # Get mask indices in spec domain from metadata
#     spec_start_idx = metadata['mask_start_frame_idx'][0]
#     spec_end_idx = metadata['mask_end_frame_idx'][0]
#
#     # Calculate context window (same duration as mask on each side)
#     mask_duration = spec_end_idx - spec_start_idx
#     context_start_idx = max(0, spec_start_idx - mask_duration)
#     context_end_idx = min(mask.shape[-1], spec_end_idx + mask_duration)
#
#     # Calculate time extent for plotting
#     time_per_column = sample_len_seconds / mask.shape[-1]
#     plot_start_time = context_start_idx * time_per_column
#     plot_end_time = context_end_idx * time_per_column
#
#     # Clean spectrogram
#     clean_mag_db = clean_spec[0, 0, :, :]
#     mask = mask.squeeze(1).squeeze(0)
#     im = axs[0, 0].imshow(clean_mag_db[:, context_start_idx:context_end_idx].numpy(),
#                           origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
#                           extent=[plot_start_time, plot_end_time, 0, clean_mag_db.shape[0]])
#     axs[0, 0].set_title('Clean Spectrogram')
#     plt.colorbar(im, ax=axs[0, 0])
#
#     # Masked spectrogram
#     masked_mag_db = masked_spec[0, 0, :, :]
#     im = axs[0, 1].imshow(masked_mag_db[:, context_start_idx:context_end_idx].numpy(),
#                           origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
#                           extent=[plot_start_time, plot_end_time, 0, masked_mag_db.shape[0]])
#     axs[0, 1].set_title('Masked Spectrogram')
#     plt.colorbar(im, ax=axs[0, 1])
#
#     # Model output spectrogram
#     output_mag_db = pred_spec_mag[0, 0, :, :]
#     im = axs[0, 2].imshow(output_mag_db[:, context_start_idx:context_end_idx].numpy(),
#                           origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
#                           extent=[plot_start_time, plot_end_time, 0, pred_spec_mag.shape[2]])
#     axs[0, 2].set_title('Model Output Spectrogram')
#     plt.colorbar(im, ax=axs[0, 2])
#
#     # Plot error
#     error_db = torch.abs(clean_mag_db - output_mag_db)
#     im = axs[0, 3].imshow(error_db[:, context_start_idx:context_end_idx].numpy(),
#                           origin='lower', aspect='auto', vmin=vmin_err, vmax=vmax_err,
#                           extent=[plot_start_time, plot_end_time, 0, error_db.shape[0]])
#     axs[0, 3].set_title('Reconstruction Error (dB)')
#     plt.colorbar(im, ax=axs[0, 3])
#
#     # Plot zoomed spectrograms
#     clean_spec_mag_zoom = clean_mag_db[mask == 0]
#     clean_spec_mag_zoom = clean_spec_mag_zoom.reshape(clean_spec_mag_zoom.shape[-1] // clean_mag_db.shape[0],
#                                                       clean_mag_db.shape[0])
#     im = axs[0, 4].imshow(clean_spec_mag_zoom.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
#                           extent=[plot_start_time + mask_duration * time_per_column,
#                                   plot_start_time + 2 * mask_duration * time_per_column,
#                                   0, clean_mag_db.shape[0]])
#     axs[0, 4].set_title('inpainting part clean spec')
#     plt.colorbar(im, ax=axs[0, 4])
#
#     output_spec_mag_zoom = output_mag_db[mask == 0]
#     output_spec_mag_zoom = output_spec_mag_zoom.reshape(output_spec_mag_zoom.shape[-1] // output_mag_db.shape[0],
#                                                         output_mag_db.shape[0])
#     im = axs[0, 5].imshow(output_spec_mag_zoom.numpy(), origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
#                           extent=[plot_start_time + mask_duration * time_per_column,
#                                   plot_start_time + 2 * mask_duration * time_per_column,
#                                   0, output_mag_db.shape[0]])
#     axs[0, 5].set_title('inpainting part predicted spec')
#     plt.colorbar(im, ax=axs[0, 5])
#
#     # Remove remaining subplots in first row
#     for j in range(6, n_cols):
#         axs[0, j].remove()
#
#     # Plot PC directions and variations
#     for i in range(n_dirs):
#         row_idx = i + 1
#         pc_dir_db = pc_directions_mag[0, i]
#
#         # Show the PC direction (zoomed)
#         im = axs[row_idx, 0].imshow(pc_dir_db[:, context_start_idx:context_end_idx].cpu().numpy(),
#                                     origin='lower', aspect='auto',
#                                     vmin=vmin, vmax=vmax,
#                                     extent=[plot_start_time, plot_end_time, 0, pc_directions_mag.shape[-2]])
#         axs[row_idx, 0].set_title(f'PC Direction {i + 1} (dB)')
#         plt.colorbar(im, ax=axs[row_idx, 0])
#
#         for j, alpha in enumerate(alphas):
#             # Add PC direction to base prediction (zoomed)
#             modified_spec = output_mag_db + alpha * pc_dir_db
#             im = axs[row_idx, j + 1].imshow(modified_spec[:, context_start_idx:context_end_idx].cpu().numpy(),
#                                             origin='lower', aspect='auto',
#                                             vmin=vmin, vmax=vmax,
#                                             extent=[plot_start_time, plot_end_time, 0, modified_spec.shape[0]])
#             axs[row_idx, j + 1].set_title(f'Base + PC{i + 1} (α={alpha:.1f})')
#             plt.colorbar(im, ax=axs[row_idx, j + 1])
#
#     plt.tight_layout()
#     return fig

def plot_pc_spectrograms(masked_spec, clean_spec, pred_spec_mag, pc_directions_mag, mask, sample_len_seconds,
                         metadata, save_dir, sample_idx, max_dirs=None):
    """
    Plot spectrograms, error, and PC directions, focusing on the inpainting area and its surroundings
    """
    # Create directory for individual spectrograms
    sample_dir = Path(save_dir) / f"sample_{sample_idx}" / "spectrograms"
    sample_dir.mkdir(parents=True, exist_ok=True)

    n_dirs = pc_directions_mag.shape[1]
    if max_dirs is not None:
        n_dirs = min(n_dirs, max_dirs)
        pc_directions_mag = pc_directions_mag[:, :n_dirs]

    alphas = torch.arange(-3, 3.5, 0.5)
    n_alphas = len(alphas)
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
        # Set font sizes
        plt.rcParams.update({'font.size': 14})  # Base font size

        fig_single, ax = plt.subplots(figsize=(10, 6))

        # Calculate frequency values for y-axis
        sample_rate = 16000  # Hz
        n_fft = 255  # FFT size
        n_freq_bins = data.shape[0]  # Should be 128 (n_fft//2 + 1)
        freqs = np.linspace(0, sample_rate / 2, n_freq_bins)  # Array of frequency values

        # Plot spectrogram with frequency y-axis
        im = ax.imshow(data, origin='lower', aspect='auto',
                       vmin=vmin if 'error' not in filename else vmin_err,
                       vmax=vmax if 'error' not in filename else vmax_err,
                       extent=[plot_start_time, plot_end_time, freqs[0], freqs[-1]])

        # Add colorbar with larger font
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=12)  # Colorbar tick font size

        # # Add vertical lines to show mask region if needed
        # if not filename.startswith('pc_direction'):
        #     ax.axvline(x=spec_start_idx * time_per_column, color='r', linestyle='--', alpha=0.5)
        #     ax.axvline(x=spec_end_idx * time_per_column, color='r', linestyle='--', alpha=0.5)

        ax.axvline(x=spec_start_idx * time_per_column, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=spec_end_idx * time_per_column, color='r', linestyle='--', alpha=0.5)

        # Set axis labels with larger font
        ax.set_xlabel('Time (s)', fontsize=18)
        ax.set_ylabel('Frequency (kHz)', fontsize=18)

        # Set y-ticks at meaningful frequencies with larger font
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
    # Save individual clean spectrogram
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
    # Save individual masked spectrogram
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
    # Save individual output spectrogram
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
    # Save individual error spectrogram
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

    # Plot PC directions and variations
    for i in range(n_dirs):
        row_idx = i + 1
        pc_dir_db = pc_directions_mag[0, i]

        # Show the PC direction (zoomed)
        im = axs[row_idx, 0].imshow(pc_dir_db[:, context_start_idx:context_end_idx].cpu().numpy(),
                                    origin='lower', aspect='auto',
                                    vmin=vmin, vmax=vmax,
                                    extent=[plot_start_time, plot_end_time, 0, pc_directions_mag.shape[-2]])
        axs[row_idx, 0].set_title(f'PC Direction {i + 1} (dB)')
        plt.colorbar(im, ax=axs[row_idx, 0])
        axs[row_idx, 0].axvline(x=spec_start_idx * time_per_column, color='r', linestyle='--', alpha=0.5)
        axs[row_idx, 0].axvline(x=spec_end_idx * time_per_column, color='r', linestyle='--', alpha=0.5)

        # Save individual PC direction spectrogram
        save_individual_spectrogram(
            pc_dir_db[:, context_start_idx:context_end_idx].cpu().numpy(),
            '',
            f'pc_direction_{i + 1}.png'
        )

        for j, alpha in enumerate(alphas):
            # Add PC direction to base prediction (zoomed)
            modified_spec = output_mag_db + alpha * pc_dir_db
            im = axs[row_idx, j + 1].imshow(modified_spec[:, context_start_idx:context_end_idx].cpu().numpy(),
                                            origin='lower', aspect='auto',
                                            vmin=vmin, vmax=vmax,
                                            extent=[plot_start_time, plot_end_time, 0, modified_spec.shape[0]])
            axs[row_idx, j + 1].set_title(f'Base + PC{i + 1} (α={alpha:.1f})')
            plt.colorbar(im, ax=axs[row_idx, j + 1])
            axs[row_idx, j + 1].axvline(x=spec_start_idx * time_per_column, color='r', linestyle='--', alpha=0.5)
            axs[row_idx, j + 1].axvline(x=spec_end_idx * time_per_column, color='r', linestyle='--', alpha=0.5)

            # Save individual PC variation spectrogram
            save_individual_spectrogram(
                modified_spec[:, context_start_idx:context_end_idx].cpu().numpy(),
                f'Base + PC{i + 1} (α={alpha:.1f})',
                f'pc{i + 1}_alpha_{alpha:.1f}.png'
            )

    plt.tight_layout()
    return fig


def process_audio_for_phonemes(audio_tensor, processor, phoneme_model, sample_rate=16000):
    """Process audio tensor for phoneme recognition"""

    def decode_phonemes(ids, processor, ignore_stress=False):
        """CTC-like decoding with consecutive duplicates removal"""
        # Remove consecutive duplicates
        ids = [id_ for id_, _ in groupby(ids.tolist())]

        # Get special token IDs to skip
        special_token_ids = processor.tokenizer.all_special_ids + [
            processor.tokenizer.word_delimiter_token_id
        ]

        # Convert IDs to phonemes, skipping special tokens
        phonemes = [processor.decode(id_) for id_ in ids if id_ not in special_token_ids]

        # Join phonemes
        prediction = " ".join(phonemes)

        # Optionally remove stress marks
        if ignore_stress:
            prediction = prediction.replace("ˈ", "").replace("ˌ", "")

        return prediction

    with torch.no_grad():
        # Process audio
        inputs = processor(audio_tensor.numpy(), sampling_rate=sample_rate, return_tensors="pt")
        logits = phoneme_model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode phonemes
        phoneme_sequence = decode_phonemes(predicted_ids[0], processor, ignore_stress=True)

    return phoneme_sequence


def get_with_full_audio(clean_audio_full, pred_subsample_audio, metadata):
    subsample_start_idx = metadata['subsample_start_idx'][0]
    mask_start_idx = metadata['mask_start_idx'][0]
    mask_end_idx = metadata['mask_end_idx'][0]
    pred_audio_full = clean_audio_full
    pred_audio_full[subsample_start_idx + mask_start_idx: subsample_start_idx + mask_end_idx] = pred_subsample_audio[
                                                                                                mask_start_idx: mask_end_idx]
    return pred_audio_full


def save_pc_audio_variations(clean_spec_mag_norm_log, pred_spec_mag, pc_directions_mag, clean_spec, mask, masked_audio,
                             metadata, alphas, save_dir, pitch_save_path, mean, std, sample_idx,
                             n_fft=255, hop_length=128, sample_rate=16000, analyze_phonemes=False):
    """
    Save audio variations, their pitch analyses, and transcriptions

    """
    # Create sample-specific directory
    sample_dir = Path(save_dir) / f"sample_{sample_idx}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Initialize whisper model (using base model for faster inference)
    whisper_model = whisper.load_model("base")

    # Initialize phoneme model only if needed
    if analyze_phonemes:
        model_name = "bookbot/wav2vec2-ljspeech-gruut"
        phoneme_model = Wav2Vec2ForCTC.from_pretrained(model_name, weights_only=True)
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        phoneme_model.eval()

    transcriptions = {}
    phonemes = {} if analyze_phonemes else None

    # Process clean audio and get transcription
    clean_spec_ref_complex = torch.complex(clean_spec[0, 0], clean_spec[0, 1])
    clean_phase = torch.angle(clean_spec_ref_complex)
    window = torch.hann_window(n_fft).to(clean_phase.device)

    clean_mag_log = clean_spec_mag_norm_log[0, 0] * std + mean
    clean_mag_linear = torch.exp(clean_mag_log) - 1e-6
    real_part = clean_mag_linear * torch.cos(clean_phase)
    imag_part = clean_mag_linear * torch.sin(clean_phase)
    complex_clean_spec = torch.complex(real_part, imag_part)

    clean_audio = torch.istft(complex_clean_spec,
                              n_fft=n_fft,
                              hop_length=hop_length,
                              win_length=n_fft,
                              window=window)

    # Initialize audio variations dictionary
    audio_variations = {
        'clean': clean_audio,
        'masked': masked_audio
    }

    # Get ground truth transcription from metadata
    ground_truth_transcription = metadata['transcriptions'][0]
    clean_audio_path_ref = metadata['clean_audio_paths'][0]
    clean_audio_full = torchaudio.load(clean_audio_path_ref)[0].squeeze(0)
    clean_path_full = sample_dir / "clean_full.wav"
    torchaudio.save(clean_path_full, clean_audio_full.unsqueeze(0), sample_rate=sample_rate)

    # Save reference audio files and get transcriptions and phonemes
    clean_path = sample_dir / "clean.wav"
    torchaudio.save(clean_path, clean_audio.unsqueeze(0), sample_rate=sample_rate)
    # Use whisper for clean audio transcription
    transcriptions['clean'] = whisper_model.transcribe(clean_path_full.as_posix(), language="en")['text']
    transcriptions['ground_truth'] = ground_truth_transcription
    if analyze_phonemes:
        phonemes['clean'] = process_audio_for_phonemes(clean_audio_full, processor, phoneme_model)

    masked_audio_path_full = sample_dir / "masked_audio_full.wav"
    masked_audio_full = get_with_full_audio(clean_audio_full, masked_audio.squeeze(0).squeeze(0), metadata)
    torchaudio.save(masked_audio_path_full, masked_audio_full.unsqueeze(0), sample_rate=sample_rate)

    masked_audio_path = sample_dir / "masked_audio.wav"
    torchaudio.save(masked_audio_path, masked_audio.squeeze(0), sample_rate=sample_rate)
    transcriptions['masked'] = whisper_model.transcribe(masked_audio_path_full.as_posix(), language="en")['text']
    if analyze_phonemes:
        phonemes['masked'] = process_audio_for_phonemes(masked_audio_full, processor, phoneme_model)

    # Process PC directions
    for i in range(pc_directions_mag.shape[1]):
        pc_dir = sample_dir / f"pc_{i + 1}"
        pc_dir.mkdir(exist_ok=True)
        pc_dir_db = pc_directions_mag[0, i]

        for alpha in alphas:
            modified_mag = pred_spec_mag[0, 0] + alpha * pc_dir_db
            modified_mag_log = modified_mag * std + mean
            modified_mag_linear = torch.exp(modified_mag_log)
            real_part = modified_mag_linear * torch.cos(clean_phase)
            imag_part = modified_mag_linear * torch.sin(clean_phase)
            modified_complex = torch.complex(real_part, imag_part)

            audio = torch.istft(modified_complex,
                                n_fft=n_fft,
                                hop_length=hop_length,
                                win_length=n_fft,
                                window=window)

            # Add to variations dictionary
            variation_name = f'pc{i + 1}_alpha{alpha:.1f}'
            audio_variations[variation_name] = audio

            # Create full audio
            curr_full_audio = get_with_full_audio(clean_audio_full, audio, metadata)

            # Save the full audio file
            audio_path_full = pc_dir / f"alpha_{alpha:.1f}_full.wav"
            torchaudio.save(audio_path_full, curr_full_audio.unsqueeze(0), sample_rate=sample_rate)

            # Save audio file and get transcription and phonemes
            audio_path = pc_dir / f"alpha_{alpha:.1f}.wav"
            torchaudio.save(audio_path, audio.unsqueeze(0), sample_rate=sample_rate)
            transcriptions[variation_name] = whisper_model.transcribe(audio_path_full.as_posix(), language="en")['text']
            if analyze_phonemes:
                phonemes[variation_name] = process_audio_for_phonemes(curr_full_audio, processor, phoneme_model)

    # Save transcriptions and phonemes to a text file
    with open(sample_dir / "transcriptions_and_phonemes.txt", "w") as f:
        # First write the ground truth
        f.write("Ground Truth Transcription:\n")
        f.write(f"{transcriptions['ground_truth']}\n\n")

        for name in ['clean', 'masked'] + [f'pc{i + 1}_alpha{alpha:.1f}' for i in range(pc_directions_mag.shape[1]) for
                                           alpha in alphas]:
            f.write(f"{name}:\n")
            f.write(f"Transcription: {transcriptions[name]}\n")
            if analyze_phonemes and name in phonemes:
                f.write(f"Phonemes: {phonemes[name]}\n")
            f.write("\n")

    # Generate and save pitch analysis
    n_dirs = pc_directions_mag.shape[1]
    pitch_fig = plot_pitch_comparison(audio_variations, n_dirs, sample_rate, pitch_save_path, sample_idx)
    pitch_fig.savefig(sample_dir / f"pitch_comparison.png")
    plt.close(pitch_fig)

    return {'transcriptions': transcriptions, 'phonemes': phonemes}

#
# def calculate_unet_baseline(model, masked_spec, mask, n_mc_samples=50, n_components=5):
#     """
#     Calculate U-Net baseline with MC Dropout and PCA analysis
#
#     Args:
#         model: The U-Net model
#         masked_spec: Masked spectrogram input [B, 1, F, T]
#         mask: Binary mask [B, 1, F, T] (1 for known regions, 0 for inpainting area)
#         n_mc_samples: Number of MC dropout samples
#         n_components: Number of principal components to extract
#     Returns:
#         dict containing:
#         - mean_prediction: [1, 1, F, T]
#         - principal_components: [1, n_components, F, T]  # Exactly this shape
#         - importance_weights: [n_components]
#     """
#     # Enable dropout
#     enable_dropout(model)
#
#     # Collect MC samples
#     mc_predictions = []
#     for _ in range(n_mc_samples):
#         with torch.no_grad():
#             pred = model(masked_spec, mask)  # Shape: [B, 1, F, T]
#             # Extract only the inpainting area
#             pred_inpaint = pred[mask == 0]  # Shape: [N_masked_elements]
#             mc_predictions.append(pred_inpaint)
#
#     # Stack predictions: [n_mc, N_masked_elements]
#     mc_predictions = torch.stack(mc_predictions)
#
#     # Reshape for PCA: [n_mc, B, N_masked_per_batch]
#     B = masked_spec.shape[0]
#     N_masked_per_batch = (~mask.bool()).sum() // B  # Number of masked elements per batch item
#     predictions_flat = mc_predictions.reshape(n_mc_samples, B, N_masked_per_batch)
#
#     # Apply PCA analysis
#     principal_components, importance_weights, mean_prediction = compute_pca_and_importance_weights(predictions_flat)
#     # turn to torch, move back to the device:
#     device = masked_spec.device
#     principal_components = torch.from_numpy(principal_components).to(device)
#     importance_weights = torch.from_numpy(importance_weights).to(device)
#     mean_prediction = mean_prediction.to(device)
#
#     # Reconstruct full spectrograms with zeros in known regions
#     _, F, T = masked_spec.shape[1:]
#
#     # Helper function to reconstruct full spectrogram
#     def reconstruct_full_spec(inpaint_values):
#         full_spec = torch.zeros((F, T), device=masked_spec.device)
#         full_spec.reshape(-1)[mask[0, 0].reshape(-1) == 0] = inpaint_values
#         return full_spec
#
#     # Reshape principal components back to full spectrograms
#     principal_components = principal_components.reshape(n_components, N_masked_per_batch)
#
#     # Reconstruct each PC and stack them
#     full_pcs = torch.stack([
#         reconstruct_full_spec(pc) for pc in principal_components
#     ])  # [n_components, F, T]
#
#     # Reshape to desired output shape [1, n_components, F, T]
#     full_pcs = full_pcs.unsqueeze(0)  # [1, n_components, F, T]
#
#     # Reshape mean prediction to full spectrogram [1, 1, F, T]
#     mean_prediction = mean_prediction.reshape(N_masked_per_batch)
#     full_mean = reconstruct_full_spec(mean_prediction).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
#
#     # Add shape assertions to verify
#     assert full_pcs.shape == (
#         1, n_components, F, T), f"PC shape is {full_pcs.shape}, expected (1, {n_components}, {F}, {T})"
#     assert full_mean.shape == (1, 1, F, T), f"Mean shape is {full_mean.shape}, expected (1, 1, {F}, {T})"
#
#     return {
#         'mean_prediction': full_mean,
#         'principal_components': full_pcs,
#         'importance_weights': importance_weights
#     }


def compute_metrics(nppc_directions, mc_dropout_directions, pred_spec_mag, mean_prediction, clean_spec_mag, mask):
    """
    Compute comparison metrics between NPPC and MC-Dropout PCA

    Args:
        nppc_directions: NPPC directions [1, n_components, F, T]
        mc_dropout_directions: MC-Dropout PCA directions [1, n_components, F, T]
        pred_spec_mag: Predicted spectrogram [1, 1, F, T]
        mean_prediction: Predicted mean spectrogram of the mc dropout method [1, 1, F, T]
        clean_spec_mag: Ground truth spectrogram [1, 1, F, T]
        mask: Binary mask [1, 1, F, T] (1 for known regions, 0 for inpainting area)

    Returns:
        dict containing metrics for both methods
    """

    def compute_rmse(pred, target, mask):
        """Compute RMSE only in the inpainting region"""
        error = pred - target
        masked_error = error[mask == 0]
        return torch.norm(masked_error).item()

    def compute_residual_error_magnitude(error, directions):
        """
        Compute ||e - WW^T e||_2 where e is the error and W are the principal directions
        """
        # Flatten the error and directions for the masked region
        # error_flat = error[mask == 0].reshape(1, -1)  # [1, N]
        error_flat = error.reshape(error.shape[1], -1)
        directions_flat = directions.reshape(directions.shape[1], -1)  # [n_components, N]

        directions_flat_normalized = directions_flat
        directions_flat_norms = directions_flat.norm(dim=1) + 1e-6
        directions_flat_normalized = directions_flat / directions_flat_norms[:, None]

        # Compute W(W^T e)
        wt_e = torch.matmul(directions_flat_normalized, error_flat.T)  # [n_components, 1]
        w_wt_e = torch.matmul(directions_flat_normalized.T, wt_e)  # [N, 1]

        # Compute ||e - WW^T e||_2
        residual = error_flat.T - w_wt_e
        return torch.norm(residual).item()

    def compute_principal_angle(nppc_dirs, mc_dirs):
        """
        Compute principal angles between two subspaces
        Returns angles in degrees

        Args:
            nppc_dirs: NPPC directions [1, n_components, F, T]
            mc_dirs: MC-Dropout directions [1, n_components, F, T]
        Returns:
            list: Principal angles in degrees
        """
        # Flatten directions for masked region
        nppc_flat = nppc_dirs.reshape(nppc_dirs.shape[1], -1)  # [n_components, N]
        mc_flat = mc_dirs.reshape(mc_dirs.shape[1], -1)  # [n_components, N]

        # Orthonormalize the bases using QR decomposition
        nppc_q, _ = torch.linalg.qr(nppc_flat.T)
        mc_q, _ = torch.linalg.qr(mc_flat.T)

        # Compute singular values using svdvals
        s = torch.linalg.svdvals(torch.matmul(nppc_q.T, mc_q))

        # Convert to angles in degrees
        angles = torch.arccos(torch.clamp(s, -1, 1)) * 180 / np.pi

        return angles.cpu().tolist()

    # Compute error
    error = pred_spec_mag - clean_spec_mag

    # Compute metrics
    metrics = {
        'nppc': {
            'rmse': compute_rmse(pred_spec_mag, clean_spec_mag, mask),
            'residual_error': compute_residual_error_magnitude(error, nppc_directions)
        },
        'mc_dropout': {
            'rmse': compute_rmse(mean_prediction, clean_spec_mag, mask),
            'residual_error': compute_residual_error_magnitude(error, mc_dropout_directions)
        },
        'principal_angles': compute_principal_angle(nppc_directions, mc_dropout_directions)
    }

    return metrics


def save_metrics_to_json(metrics, save_dir, sample_idx):
    """
    Save metrics to a JSON file with organized directory structure

    Args:
        metrics: Dictionary of metrics
        save_dir: Base directory for saving validation results
        sample_idx: Index of the current sample
    """
    # Convert numpy arrays and tensors to lists
    json_metrics = {}
    for method, values in metrics.items():
        if method == 'principal_angles':
            # Convert numpy array of angles from radians to degrees
            json_metrics[method] = [float(angle) for angle in values]
        else:
            json_metrics[method] = {
                k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                for k, v in values.items()
            }

    # Create validation directory structure
    metrics_dir = Path(save_dir) / "validation_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON file
    json_path = metrics_dir / f"sample_{sample_idx}.json"
    with open(json_path, 'w') as f:
        json.dump(json_metrics, f, indent=4)


# def calculate_unet_baseline(model, masked_spec, mask, n_mc_samples=50, n_components=5):
#     """
#     Calculate U-Net baseline with MC Dropout and PCA analysis
#
#     Args:
#         model: The U-Net model
#         masked_spec: Masked spectrogram input [B, 1, F, T]
#         mask: Binary mask [B, 1, F, T]
#         n_mc_samples: Number of MC dropout samples
#         n_components: Number of principal components to extract
#     """
#     # Enable dropout for MC sampling
#     enable_dropout(model)
#
#     # Collect MC samples
#     mc_predictions = []
#     for _ in range(n_mc_samples):
#         with torch.no_grad():
#             pred = model(masked_spec, mask)  # Shape: [B, 1, F, T]
#             mc_predictions.append(pred)
#
#     # Stack predictions: [n_mc, B, 1, F, T]
#     mc_predictions = torch.stack(mc_predictions)
#
#     # Reshape for PCA: [n_mc, B, -1]
#     B = masked_spec.shape[0]
#     predictions_flat = mc_predictions.reshape(n_mc_samples, B, -1)  # Flatten all dimensions after batch
#
#     # Apply PCA analysis
#     principal_components, importance_weights, mean_prediction = compute_pca_and_importance_weights(predictions_flat)
#
#     # Reshape components back to spectrogram shape: [n_components, B, 1, F, T]
#     _, F, T = masked_spec.shape[1:]  # Get F, T from input shape
#     principal_components = torch.from_numpy(principal_components).reshape(n_components, B, 1, F, T)
#     mean_prediction = mean_prediction.reshape(B, 1, F, T)
#
#     return {
#         'mean_prediction': mean_prediction,
#         'principal_components': principal_components,
#         'importance_weights': importance_weights
#     }


class NPPCModelValidatorConfig(pydantic.BaseModel):
    checkpoint_path: str
    device: str = "cuda"
    save_dir: str = "validation_nppc_results"
    model_configuration: NPPCModelConfig
    max_dirs_to_plot: int = None


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

    def validate_sample(self, masked_spec, mask, clean_spec, masked_audio, metadata, sample_len_seconds, sample_idx):
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
            # Calculate mc dropout + pca
            restoration_model = self.model.pretrained_restoration_model
            mc_dropout_after_pca = calculate_unet_baseline(restoration_model, masked_spec_mag_log, mask)
            mc_pc_directions = mc_dropout_after_pca['scaled_principal_components']
            self.model.eval()  # just in case
            self.model.to(self.device)
            pred_spec_mag_log = self.model.get_pred_spec_mag_norm(masked_spec_mag_log, mask)
            self._validate_with_baseline(masked_spec_mag_log,mask,clean_spec_mag_norm_log,pred_spec_mag_log,sample_idx)
            self.model.eval()  # just in case
            self.model.to(self.device)

            # Plot results
            spec_save_path = Path(self.config.save_dir) / "spec_variations"
            fig = plot_pc_spectrograms(
                masked_spec_mag_log.cpu(),
                clean_spec_mag_norm_log.cpu(),
                pred_spec_mag_log.cpu(),
                pc_directions.cpu(),
                mask.cpu(),
                sample_len_seconds,
                metadata,
                spec_save_path,
                sample_idx,
                max_dirs=self.config.max_dirs_to_plot
            )

            # Save audio variations with pitch analysis and transcriptions
            audio_save_path = Path(self.config.save_dir) / "audio_variations"
            pitch_save_path = Path(self.config.save_dir) / "pitch_variations"
            alphas = torch.arange(-3, 3.5, 0.5)
            save_pc_audio_variations(
                clean_spec_mag_norm_log.cpu(),
                pred_spec_mag_log.cpu(),
                pc_directions.cpu(),
                clean_spec.cpu(),
                mask.cpu(),
                masked_audio.cpu(),
                metadata,
                alphas,
                audio_save_path,
                pitch_save_path,
                mean.cpu(),
                std.cpu(),
                sample_idx
            )

            return {
                'figure': fig,
                'pc_directions': pc_directions.cpu()
            }

    def _validate_with_baseline(self, masked_spec, mask, clean_spec,pred_mask_spec, sample_idx,
                               n_mc_samples=50, n_components=5):
        """
        Validate NPPC model and compare with U-Net baseline
        """
        # Get NPPC results
        self.model.eval()

        with torch.no_grad():
            nppc_directions = self.model(masked_spec, mask)

        # Get MC-Dropout baseline using the restoration model from NPPC
        mc_dropout_results = calculate_unet_baseline(
            self.model.pretrained_restoration_model,
            masked_spec,
            mask,
            n_mc_samples=n_mc_samples,
            n_components=n_components
        )
        self.model.pretrained_restoration_model.eval()

        # Compute metrics
        metrics = compute_metrics(
            nppc_directions=nppc_directions,
            mc_dropout_directions=mc_dropout_results['scaled_principal_components'],
            pred_spec_mag=pred_mask_spec,
            mean_prediction=mc_dropout_results['mean_prediction'],
            clean_spec_mag=clean_spec,
            mask=mask
        )

        # Save metrics to JSON
        if self.config.save_dir is not None:
            save_metrics_to_json(metrics, self.config.save_dir, sample_idx)
