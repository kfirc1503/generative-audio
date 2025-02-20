import torch
import matplotlib.pyplot as plt
import pydantic
from pathlib import Path
from nppc_audio.inpainting.nppc.nppc_model import NPPCModel, NPPCModelConfig
from utils import preprocess_log_magnitude
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
            fig_pc = plt.figure(figsize=(15, 4))
            ax_pc = fig_pc.add_subplot(111)

            # Plot clean reference
            ax_pc.plot(times, f0_clean, color='black', label='Clean', linewidth=2)

            # Plot all alpha variations for this PC
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
                    ax_pc.plot(times, f0, color=colors[alpha_idx],
                               label=f'α={alpha:.1f}', alpha=0.7)

            ax_pc.set_title(f'PC Direction {pc_num} Pitch Contours')
            ax_pc.set_ylabel('Frequency (Hz)')
            ax_pc.set_xlabel('Time (s)')
            ax_pc.grid(True)
            ax_pc.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            fig_pc.tight_layout()
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
    spec_end_idx = metadata['mask_end_frame_idx'][0]

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
        fig_single, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data, origin='lower', aspect='auto',
                       vmin=vmin if 'error' not in filename else vmin_err,
                       vmax=vmax if 'error' not in filename else vmax_err,
                       extent=[plot_start_time, plot_end_time, 0, data.shape[0]])
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

        # Add vertical lines to show mask region if needed
        if not filename.startswith('pc_direction'):
            ax.axvline(x=spec_start_idx * time_per_column, color='r', linestyle='--', alpha=0.5)
            ax.axvline(x=spec_end_idx * time_per_column, color='r', linestyle='--', alpha=0.5)

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
        'Clean Spectrogram',
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
        'Masked Spectrogram',
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
        'Model Output Spectrogram',
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
        'Reconstruction Error (dB)',
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
            f'PC Direction {i + 1} (dB)',
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
                             metadata, alphas, save_dir,pitch_save_path, mean, std, sample_idx,
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
            pred_spec_mag_log = self.model.get_pred_spec_mag_norm(masked_spec_mag_log, mask)

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
