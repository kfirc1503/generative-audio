import torch
import numpy as np
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
import json
from typing import Dict, Literal
import pydantic
from FullSubNet_plus.speech_enhance.fullsubnet_plus.model.fullsubnet_plus import FullSubNetPlusConfig
import utils

from FullSubNet_plus.speech_enhance.audio_zen.acoustics.mask import decompress_cIRM
import scipy.signal as signal


class ModelValidatorConfig(pydantic.BaseModel):
    model_path: str
    model_configuration: FullSubNetPlusConfig
    device: Literal['cpu', 'cuda'] = 'cuda'


class ModelValidator:
    def __init__(self, config: ModelValidatorConfig):
        self.config = config
        self.model = utils.load_pretrained_model(config.model_path, config.model_configuration)
        self.device = config.device
        if config.device == 'cuda':
            # check if gpu is exist
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def calculate_metrics(self, clean: np.ndarray, enhanced: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        """Calculate PESQ and STOI metrics"""
        # Ensure correct shape and type
        clean = clean.squeeze()
        enhanced = enhanced.squeeze()

        # Calculate metrics
        try:
            wb_pesq = pesq(sr, clean, enhanced, 'wb')

            # Narrow-band PESQ (8kHz)
            if sr != 8000:
                # Downsample to 8000 Hz for NB-PESQ
                nb_clean = signal.resample_poly(clean, up=1, down=2)  # 16000 -> 8000
                nb_enhanced = signal.resample_poly(enhanced, up=1, down=2)
            else:
                nb_clean = clean
                nb_enhanced = enhanced
            nb_pesq = pesq(8000, nb_clean, nb_enhanced, 'nb')  # Note: using 8000 as sr here

            stoi_score = stoi(clean, enhanced, sr, extended=False)

            # Calculate SI-SDR
            enhanced_norm = enhanced - np.mean(enhanced)
            clean_norm = clean - np.mean(clean)
            alpha = np.dot(enhanced_norm, clean_norm) / (np.linalg.norm(clean_norm) ** 2 + 1e-6)
            si_sdr = 20 * np.log10(
                np.linalg.norm(alpha * clean_norm) / (np.linalg.norm(alpha * clean_norm - enhanced_norm) + 1e-6))

            metrics = {
                'WB_PESQ': float(wb_pesq),
                'NB_PESQ': float(nb_pesq),
                'STOI': float(stoi_score),
                'SI_SDR': float(si_sdr)
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics = {
                'WB_PESQ': -1,
                'NB_PESQ': -1,
                'STOI': -1,
                'SI_SDR': -1
            }

        return metrics

    def enhance_audio(self, noisy: torch.Tensor) -> np.ndarray:
        """Enhance a single audio sample"""
        with torch.no_grad():
            noisy = noisy.to(self.device).unsqueeze(0)
            # Use the mag_complex_full_band_crm_mask inference type
            noisy_complex = torch.stft(
                noisy,
                n_fft=512,
                hop_length=256,
                win_length=512,
                window=torch.hann_window(512).to(self.device),
                return_complex=True
            )

            noisy_mag = noisy_complex.abs().unsqueeze(1)
            noisy_real = noisy_complex.real.unsqueeze(1)
            noisy_imag = noisy_complex.imag.unsqueeze(1)

            pred_crm = self.model(noisy_mag, noisy_real, noisy_imag)
            pred_crm = pred_crm.permute(0, 2, 3, 1)
            # dont know if necessary yet
            pred_crm = decompress_cIRM(pred_crm)

            # Apply mask and reconstruct
            enhanced_real = pred_crm[..., 0] * noisy_complex.real - pred_crm[..., 1] * noisy_complex.imag
            enhanced_imag = pred_crm[..., 1] * noisy_complex.real + pred_crm[..., 0] * noisy_complex.imag
            enhanced_complex = torch.complex(enhanced_real, enhanced_imag)

            enhanced = torch.istft(
                enhanced_complex,
                n_fft=512,
                hop_length=256,
                win_length=512,
                window=torch.hann_window(512).to(self.device),
                length=noisy.size(-1)
            )

            return enhanced.cpu().numpy().squeeze()

    def validate_dataloader(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Validate model using a dataloader and return average metrics"""
        all_metrics = []

        # Process batches
        for batch in tqdm(dataloader, desc="Validating"):
            noisy, clean = batch
            batch_size = noisy.size(0)

            # Process each item in batch
            for i in range(batch_size):
                # Enhance audio
                enhanced = self.enhance_audio(noisy[i])

                # Calculate metrics
                metrics = self.calculate_metrics(
                    clean[i].numpy(),
                    enhanced,
                    sr=16000  # You might want to make this configurable
                )
                all_metrics.append(metrics)

        # Calculate average metrics
        avg_metrics = {
            metric: np.mean([m[metric] for m in all_metrics])
            for metric in all_metrics[0].keys()
        }

        # Print results
        print("\nValidation Results:")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.4f}")

        return avg_metrics

    def save_metrics(self, metrics: Dict[str, float], save_path: str):
        """Save metrics to a JSON file"""
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
