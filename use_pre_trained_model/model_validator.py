import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
import json
from dataset import AudioDataset, AudioDataSetConfig
from typing import Dict, List


class ModelValidator:
    def __init__(self, model_path: str, config_path: str, device: str = "cuda"):
        self.device = device
        # Load model and config
        self.model, self.config = self.load_model_and_config(model_path, config_path)
        self.model.to(device)
        self.model.eval()

    def load_model_and_config(self, model_path: str, config_path: str):
        # Load the pre-trained model
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint['model_config']

        # Import the model class dynamically
        from fullsubnet_plus.model.fullsubnet_plus import FullSubNet_Plus
        model = FullSubNet_Plus(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model, model_config

    def calculate_metrics(self, clean: np.ndarray, enhanced: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        """Calculate PESQ and STOI metrics"""
        # Ensure correct shape and type
        clean = clean.squeeze()
        enhanced = enhanced.squeeze()

        # Calculate metrics
        try:
            wb_pesq = pesq(sr, clean, enhanced, 'wb')
            nb_pesq = pesq(sr, clean, enhanced, 'nb')
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

    def validate_dataset(self, dataset: AudioDataset, num_samples: int = 100) -> Dict[str, float]:
        """Validate model on dataset and return average metrics"""
        all_metrics = []

        for i in tqdm(range(num_samples)):
            noisy, clean = dataset[i % len(dataset)]

            # Enhance audio
            enhanced = self.enhance_audio(noisy)

            # Calculate metrics
            metrics = self.calculate_metrics(clean.numpy(), enhanced)
            all_metrics.append(metrics)

        # Calculate average metrics
        avg_metrics = {
            metric: np.mean([m[metric] for m in all_metrics])
            for metric in all_metrics[0].keys()
        }

        return avg_metrics


def main():
    # Setup paths and configuration
    model_path = "path/to/pretrained/model.pth"
    config_path = "path/to/inference.toml"

    # Setup dataset
    dataset_config = AudioDataSetConfig(
        clean_path="path/to/clean/wavs",
        noisy_path="path/to/noisy/wavs",
        sample_rate=16000,
        sub_sample_length_seconds=3.0
    )

    dataset = AudioDataset(dataset_config)

    # Initialize validator
    validator = ModelValidator(model_path, config_path)

    # Run validation
    metrics = validator.validate_dataset(dataset, num_samples=100)

    # Print and save results
    print("\nValidation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save metrics to file
    with open("validation_results.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()