import torch
import os
import json
from tqdm import tqdm
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pydantic
from typing import Optional
from nppc_audio.nppc_model import NPPCModel, NPPCModelConfig
from FullSubNet_plus.speech_enhance.audio_zen.acoustics.mask import decompress_cIRM
import utils


class NPPCAudioValidatorConfig(pydantic.BaseModel):
    nppc_audio_model_configuration: NPPCModelConfig
    checkpoint_path: str
    metrics_path: Optional[str] = None


class NPPCAudioValidator:
    def __init__(self, config, metrics_path=None):
        """
        Initialize validator with model checkpoint and optional metrics

        Args:
            config: Configuration object
            checkpoint_path: Path to model checkpoint
            metrics_path: Path to training metrics JSON file (optional)
        """
        self.config = config
        self.device = config.device

        # Load checkpoint
        self.checkpoint = torch.load(config.checkpoint_path, map_location=self.device)

        # Initialize model
        self.model = NPPCModel(config.nppc_model_configuration)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Load training metrics if provided
        if config.metrics_path and os.path.exists(metrics_path):
            with open(config.metrics_path, 'r') as f:
                self.training_metrics = json.load(f)
            print(f"Loaded training metrics from {config.metrics_path}")
            print(f"Model was trained for {self.training_metrics['total_steps']} steps")
            print(f"Final training loss: {self.training_metrics['final_loss']:.4f}")
        else:
            self.training_metrics = None
            print("No training metrics provided or file not found.")

    def _crm_directions_to_spectograms(self, noisy_audio):
        """
        Validate the n_dir CRM masks by converting them to spectrograms

        Args:
            noisy_audio: Noisy audio tensor [1, T]
        output:
            list of the spectrograms of the curr_pc_crm directions
        """
        self.model.eval()
        with torch.no_grad():
            noisy = noisy_audio.to(self.device)
            # Get w_mat (CRM directions)
            w_mat = self.model(noisy_audio)  # [1, n_dirs, 2, F, T]

            # Get STFT of noisy audio
            stft_config = self.config.nppc_model_configuration.stft_configuration
            window = torch.hann_window(stft_config.win_length).to(self.device)

            noisy_complex = torch.stft(
                noisy_audio,
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
                mag_spec = torch.abs(enhanced_complex)
                mag_spec_db = 20 * torch.log10(mag_spec + 1e-8)

                specs.append(mag_spec_db)
            return specs

    def visualize_pc_spectrograms(self, noisy_audio, clean_audio=None, save_dir=None):
        #TODO : need to change this function 
        """
        Visualize PC spectrograms along with noisy, clean, and enhanced spectrograms
        Similar to NPPC article visualization

        Args:
            noisy_audio: Noisy audio tensor [1, T]
            clean_audio: Clean audio tensor [1, T] (optional)
            save_dir: Directory to save visualization (optional)
        """
        self.model.eval()
        with torch.no_grad():
            noisy = noisy_audio.to(self.device)

            # Get spectrograms for each PC direction using our existing function
            pc_specs = self._crm_directions_to_spectograms(noisy)

            # Get STFT of noisy and clean
            noisy_complex, stft_config, window = self.audio_to_stft(noisy_audio)

            # Get enhanced audio spectrogram
            enhanced = self.model.enhance(noisy)
            enhanced_complex = torch.stft(
                enhanced,
                stft_config.nfft,
                hop_length=stft_config.hop_length,
                win_length=stft_config.win_length,
                window=window,
                return_complex=True
            )

            # Create visualization
            n_rows = 2
            n_cols = 2 + len(pc_specs)  # noisy, clean, enhanced + PC directions
            fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

            # Plot noisy spectrogram
            plt.subplot(n_rows, n_cols, 1)
            noisy_spec_db = 20 * torch.log10(torch.abs(noisy_complex) + 1e-8)
            plt.imshow(noisy_spec_db[0].cpu(), origin='lower', aspect='auto')
            plt.colorbar()
            plt.title('Noisy')
            plt.ylabel('Frequency')

            # Plot clean spectrogram if available
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

            # Plot enhanced spectrogram
            plt.subplot(n_rows, n_cols, 3)
            enhanced_spec_db = 20 * torch.log10(torch.abs(enhanced_complex) + 1e-8)
            plt.imshow(enhanced_spec_db[0].cpu(), origin='lower', aspect='auto')
            plt.colorbar()
            plt.title('Enhanced')

            # Plot PC spectrograms
            for i, spec in enumerate(pc_specs):
                plt.subplot(n_rows, n_cols, i + 4)
                plt.imshow(spec[0].cpu(), origin='lower', aspect='auto')
                plt.colorbar()
                plt.title(f'PC {i + 1}')
                if i == 0:
                    plt.ylabel('Frequency')

            plt.tight_layout()

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(save_dir, f'pc_spectrograms_{timestamp}.png')
                plt.savefig(save_path)
                print(f"Saved visualization to {save_path}")
                plt.close()
            else:
                plt.show()

            return pc_specs

    def audio_to_stft(self, input_audio):
        stft_config = self.config.nppc_model_configuration.stft_configuration
        window = torch.hann_window(stft_config.win_length).to(self.device)
        input_complex = torch.stft(
            input_audio,
            stft_config.nfft,
            hop_length=stft_config.hop_length,
            win_length=stft_config.win_length,
            window=window,
            return_complex=True
        )
        return input_complex, stft_config, window