import torch
import os
import json
from tqdm import tqdm
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pydantic
from typing import Optional, Literal, Union, Tuple, List
from nppc_audio.nppc_model import NPPCModel, NPPCModelConfig
from FullSubNet_plus.speech_enhance.audio_zen.acoustics.mask import decompress_cIRM
import utils
from pathlib import Path


class NPPCAudioValidatorConfig(pydantic.BaseModel):
    nppc_audio_model_configuration: NPPCModelConfig
    checkpoint_path: str
    metrics_path: Optional[str] = None
    device: Literal["cpu", "cuda"] = "cuda"


class NPPCAudioValidator:
    def __init__(self, config: NPPCAudioValidatorConfig):
        """
        Initialize validator with model checkpoint and optional metrics

        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.device)

        # Create NPPC model instance
        self.nppc_model = NPPCModel(config.nppc_audio_model_configuration)

        # Load checkpoint
        model_path = Path(config.checkpoint_path)
        model_path = model_path.expanduser().absolute()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            checkpoint = torch.load(model_path.as_posix(), map_location=self.device)
            # Load model state
            self.nppc_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from step {checkpoint['step']}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

        self.nppc_model.to(self.device)
        self.nppc_model.eval()

    def _crm_directions_to_spectograms(self, noisy_audio):
        """
        Validate the n_dir CRM masks by converting them to spectrograms

        Args:
            noisy_audio: Noisy audio tensor [1, T]
        output:
            list of the spectrograms of the curr_pc_crm directions
        """
        with torch.no_grad():
            noisy = noisy_audio.to(self.device)
            # Get w_mat (CRM directions)
            w_mat = self.nppc_model(noisy_audio)  # [1, n_dirs, 2, F, T]

            # Get STFT of noisy audio
            stft_config = self.config.nppc_audio_model_configuration.stft_configuration
            window = torch.hann_window(stft_config.win_length).to(self.device)

            noisy_complex = torch.stft(
                noisy,
                stft_config.nfft,
                hop_length=stft_config.hop_length,
                win_length=stft_config.win_length,
                window=window,
                return_complex=True
            )

            # Convert each direction to spectrogram
            B, n_dirs, _, F, T = w_mat.shape
            specs = []

            for dir_idx in range(n_dirs):
                # Get CRM for this direction
                curr_pc_crm = w_mat[:, dir_idx]  # [1, 2, F, T]
                # need to decompress the curr_pc_crm mask
                curr_pc_crm = decompress_cIRM(curr_pc_crm)
                curr_pc_crm = curr_pc_crm.permute(0, 2, 3, 1)  # [1,F,T,2]

                # Apply mask and reconstruct
                enhanced_complex = utils.crm_to_spectogram(curr_pc_crm, noisy_complex)

                # Convert to magnitude spectrogram (in dB)
                # mag_spec = torch.abs(enhanced_complex)
                # mag_spec_db = 20 * torch.log10(mag_spec + 1e-8)
                #
                # specs.append(mag_spec_db)
                specs.append(enhanced_complex)
            return specs

    def save_audio_files(self, audio_dir: Path,
                        noisy_audio: torch.Tensor,  # [1,T]
                        clean_audio: Optional[torch.Tensor],  # [1,T]
                        enhanced_complex: torch.Tensor,
                        stft_config,
                        window: torch.Tensor):
        """
        Helper function to save audio files with normalization

        Args:
            noisy_audio: Noisy audio tensor [1,T]
            clean_audio: Clean audio tensor [1,T] (optional)
            enhanced_complex: Complex STFT of enhanced audio
            stft_config: STFT configuration
            window: STFT window
        """

        # Create audio directory if it doesn't exist
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Save noisy audio (already tensor)
        noisy_np = noisy_audio.cpu().numpy()[0]  # [T]
        noisy_np = noisy_np / (np.max(np.abs(noisy_np)) + 1e-8)  # Normalize
        sf.write(audio_dir / "noisy.wav", noisy_np, samplerate=16000)

        # Save clean audio if provided
        if clean_audio is not None:
            clean_np = clean_audio.cpu().numpy()[0]  # [T]
            clean_np = clean_np / (np.max(np.abs(clean_np)) + 1e-8)  # Normalize
            sf.write(audio_dir / "clean.wav", clean_np, samplerate=16000)

        # Save enhanced audio
        enhanced_audio = torch.istft(
            enhanced_complex,
            stft_config.nfft,
            hop_length=stft_config.hop_length,
            win_length=stft_config.win_length,
            window=window,
            length=noisy_audio.shape[1]  # Use exact length from input tensor
        )
        enhanced_np = enhanced_audio.cpu().numpy()[0]  # [T]
        enhanced_np = enhanced_np / (np.max(np.abs(enhanced_np)) + 1e-8)  # Normalize
        sf.write(audio_dir / "enhanced.wav", enhanced_np, samplerate=16000)

    def visualize_pc_spectrograms(self, noisy_audio: torch.Tensor, clean_audio: Optional[torch.Tensor] = None,
                                save_dir=None):
        """
        Visualize PC spectrograms and save corresponding audio files

        Args:
            noisy_audio: Noisy audio tensor [1,T]
            clean_audio: Clean audio tensor [1,T] (optional)
            save_dir: Directory to save outputs
        """
        assert noisy_audio.dim() == 2 and noisy_audio.shape[
            0] == 1, f"Expected noisy_audio shape [1,T], got {noisy_audio.shape}"
        if clean_audio is not None:
            assert clean_audio.dim() == 2 and clean_audio.shape[
                0] == 1, f"Expected clean_audio shape [1,T], got {clean_audio.shape}"
            assert clean_audio.shape[1] == noisy_audio.shape[1], "Clean and noisy audio must have same length"

        with torch.no_grad():
            noisy = noisy_audio.to(self.device)
            if clean_audio is not None:
                clean_audio = clean_audio.to(self.device)

            save_dir = Path(save_dir)
            save_dir = save_dir.absolute()
            audio_dir = save_dir / "audio"
            spec_dir = save_dir / "spectrograms"
            audio_dir.mkdir(parents=True, exist_ok=True)
            spec_dir.mkdir(parents=True, exist_ok=True)

            # Get spectrograms for each PC direction
            pc_specs = self._crm_directions_to_spectograms(noisy)  # already in db

            # Get STFT of noisy
            stft_config = self.config.nppc_audio_model_configuration.stft_configuration
            window = torch.hann_window(stft_config.win_length).to(self.device)

            noisy_complex = torch.stft(
                noisy,
                stft_config.nfft,
                hop_length=stft_config.hop_length,
                win_length=stft_config.win_length,
                window=window,
                return_complex=True
            )

            # Get the pred_crm and enhanced spectogram
            pred_crm = self.nppc_model.get_pred_crm(noisy)  # [1,2,F,T]
            pred_crm = pred_crm.permute(0, 2, 3, 1)  # [1,F,T,2]
            pred_crm = decompress_cIRM(pred_crm)
            enhanced_complex = utils.crm_to_spectogram(pred_crm, noisy_complex)

            # Save base audio files
            self.save_audio_files(audio_dir, noisy, clean_audio, enhanced_complex, stft_config, window)

            # Create visualization grid
            n_rows = len(pc_specs) + 1
            n_cols = 9  # noisy, clean, enhanced + 6 alpha variations
            fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

            # Plot and save base spectrograms
            plt.subplot(n_rows, n_cols, 1)
            noisy_spec_db = 20 * torch.log10(torch.abs(noisy_complex) + 1e-8)
            plt.imshow(noisy_spec_db[0].cpu(), origin='lower', aspect='auto')
            plt.colorbar()
            plt.title('Noisy')
            plt.ylabel('Base\nSpectrograms')

            if clean_audio is not None:
                clean_complex = torch.stft(
                    clean_audio,
                    stft_config.nfft,
                    hop_length=stft_config.hop_length,
                    win_length=stft_config.win_length,
                    window=window,
                    return_complex=True
                )
                plt.subplot(n_rows, n_cols, 2)
                clean_spec_db = 20 * torch.log10(torch.abs(clean_complex) + 1e-8)
                plt.imshow(clean_spec_db[0].cpu(), origin='lower', aspect='auto')
                plt.colorbar()
                plt.title('Clean')

                # Error (Enhanced - Clean)
                plt.subplot(n_rows, n_cols, 4)
                error_complex = enhanced_complex - clean_complex
                error_spec_db = 20 * torch.log10(torch.abs(error_complex) + 1e-8)
                im = plt.imshow(error_spec_db[0].cpu(), origin='lower', aspect='auto',
                                vmin=-60, vmax=0)
                plt.colorbar(im)
                plt.title('Error (Enh - Clean)')

            plt.subplot(n_rows, n_cols, 3)
            enhanced_spec_db = 20 * torch.log10(torch.abs(enhanced_complex) + 1e-8)
            plt.imshow(enhanced_spec_db[0].cpu(), origin='lower', aspect='auto')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Enhanced')

            # Create alpha values for variations
            alphas = torch.linspace(-3, 3, 6)

            # For each PC direction
            for pc_idx, pc_spec in enumerate(pc_specs):
                row_start = (pc_idx + 1) * n_cols + 1
                # Convert to magnitude spectrogram (in dB)
                mag_spec = torch.abs(pc_spec)
                mag_spec_db = 20 * torch.log10(mag_spec + 1e-8)
                # Plot base PC spectrogram
                plt.subplot(n_rows, n_cols, row_start)
                im = plt.imshow(mag_spec_db[0].cpu(), origin='lower', aspect='auto',
                                vmin=-60, vmax=0)
                plt.colorbar(im)
                plt.title(f'PC {pc_idx + 1}')
                plt.ylabel(f'PC {pc_idx + 1}\nVariations')

                # For each alpha value
                for alpha_idx, alpha in enumerate(alphas):
                    # Create variation spectrogram

                    variation_complex = enhanced_complex + alpha * pc_spec

                    # Plot spectrogram
                    plt.subplot(n_rows, n_cols, row_start + alpha_idx + 1)
                    variation_spec_db = 20 * torch.log10(torch.abs(variation_complex) + 1e-8)
                    plt.imshow(variation_spec_db[0].cpu(), origin='lower', aspect='auto')
                    plt.colorbar(format='%+2.0f dB')
                    plt.title(f'α={alpha:.1f}')

                    # Save audio for this variation
                    variation_audio = torch.istft(
                        variation_complex,
                        stft_config.nfft,
                        hop_length=stft_config.hop_length,
                        win_length=stft_config.win_length,
                        window=window,
                        length=noisy_audio.shape[1]  # Use exact length from input tensor
                    )
                    variation_np = variation_audio.cpu().numpy()[0]
                    variation_np = variation_np / (np.max(np.abs(variation_np)) + 1e-8)
                    sf.write(
                        audio_dir / f"pc{pc_idx}_alpha{alpha:.1f}.wav",
                        variation_np,
                        samplerate=16000
                    )

            plt.tight_layout()
            plt.savefig(spec_dir / 'pc_spectrograms_variations.png')
            plt.close()

            print("\nAudio Information:")
            print(f"Input shape: {noisy_audio.shape}")
            print(f"Sample rate: 16000 Hz")
            print(f"Duration: {noisy_audio.shape[1] / 16000:.2f} seconds")
            print(f"\nSaved files to: {audio_dir}")

            return pc_specs, enhanced_spec_db

    def audio_to_stft(self, input_audio):
        stft_config = self.config.nppc_audio_model_configuration.stft_configuration
        window = torch.hann_window(stft_config.win_length).to(self.device)
        return torch.stft(
            input_audio,
            stft_config.nfft,
            hop_length=stft_config.hop_length,
            win_length=stft_config.win_length,
            window=window,
            return_complex=True
        )