import random
from pathlib import Path
from typing import Union, Tuple
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(
            self,
            clean_path: Union[str, Path],
            noisy_path: Union[str, Path],
            sample_rate: int = 16000,
            snr_range: Tuple[int, int] = (-5, 15),
            silence_length: float = 0.2,
            sub_sample_length: float = 3.0,
            target_dB_FS: float = -25.0,
            target_dB_FS_floating_value: float = 0.0,
    ):
        """
        Dataset for speech enhancement training that dynamically mixes clean and noisy audio.

        Args:
            clean_path (Union[str, Path]): Path to directory containing clean audio files.
            noisy_path (Union[str, Path]): Path to directory containing noise audio files.
            sample_rate (int): Target sample rate for all audio.
            snr_range (Tuple[int, int]): Range of Signal-to-Noise ratios for mixing.
            silence_length (float): Length of silence between noise segments in seconds.
            sub_sample_length (float): Length of audio samples in seconds.
            target_dB_FS (float): Central target RMS level in dBFS for normalization.
            target_dB_FS_floating_value (float): Range for random variation of target dBFS.
        """
        self.clean_path = Path(clean_path)
        self.noisy_path = Path(noisy_path)
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self.silence_length = int(silence_length * sample_rate)
        self.sub_sample_length = int(sub_sample_length * sample_rate)
        self.target_dB_FS = target_dB_FS
        self.target_dB_FS_floating_value = target_dB_FS_floating_value

        # Get all audio files
        self.clean_files = list(self.clean_path.rglob("*.wav"))
        self.noise_files = list(self.noisy_path.rglob("*.wav"))

        if not self.clean_files:
            raise ValueError(f"No WAV files found in clean directory: {clean_path}")
        if not self.noise_files:
            raise ValueError(f"No WAV files found in noise directory: {noisy_path}")

    def __len__(self) -> int:
        return len(self.clean_files)

    def _load_and_process_audio(self, file_path: Path) -> Union[torch.Tensor, None]:
        """Load and preprocess audio file to target sample rate and channels."""
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

        # Check for zero-length audio
        if waveform.numel() == 0:
            print(f"Warning: {file_path} is empty.")
            return None

        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        return waveform

    def _normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio to a target RMS level in dBFS, with optional variability."""
        if self.target_dB_FS_floating_value > 0.0:
            # Randomly select a normalization level within the specified range
            variable_dB_FS = random.uniform(
                self.target_dB_FS - self.target_dB_FS_floating_value,
                self.target_dB_FS + self.target_dB_FS_floating_value
            )
        else:
            # Use the fixed target level
            variable_dB_FS = self.target_dB_FS

        rms = waveform.pow(2).mean().sqrt()
        rms_db = 20 * torch.log10(rms + 1e-8)
        gain_db = variable_dB_FS - rms_db
        gain = 10 ** (gain_db / 20)
        waveform = waveform * gain
        return waveform

    def _get_noise_segment(self, length: int) -> torch.Tensor:
        """Generate a noise segment with silence padding."""
        noise = torch.empty(1, 0)
        while noise.shape[1] < length:
            # Randomly select noise file
            noise_file = random.choice(self.noise_files)
            noise_segment = self._load_and_process_audio(noise_file)
            if noise_segment is None:
                continue  # Skip if loading failed or file is empty

            # Normalize noise
            noise_segment = self._normalize_audio(noise_segment)

            # Add silence padding
            silence = torch.zeros(1, self.silence_length)
            noise_segment = torch.cat([noise_segment, silence], dim=1)

            noise = torch.cat([noise, noise_segment], dim=1)

        # Trim or pad to the exact required length
        noise = noise[:, :length]
        return noise

    def _mix_with_snr(self, clean: torch.Tensor, noise: torch.Tensor, snr: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mix clean and noise signals at the specified SNR."""
        # Normalize clean audio
        clean = self._normalize_audio(clean)

        # Compute power
        clean_power = clean.pow(2).mean()
        noise_power = noise.pow(2).mean()

        # Calculate scaling factor for noise
        snr_linear = 10 ** (snr / 10)
        scale = torch.sqrt(clean_power / (snr_linear * noise_power + 1e-8))
        scaled_noise = noise * scale

        # Mix signals
        noisy = clean + scaled_noise

        # Clipping prevention
        max_amp = torch.max(torch.abs(noisy))
        if max_amp > 0.99:
            scaling_factor = 0.99 / max_amp
            noisy = noisy * scaling_factor
            clean = clean * scaling_factor

        return noisy.squeeze(0), clean.squeeze(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a pair of clean and noisy audio samples."""
        # Load clean audio
        clean_file = self.clean_files[idx]
        clean = self._load_and_process_audio(clean_file)
        while clean is None:
            # If loading failed, pick another sample
            idx = random.randint(0, len(self.clean_files) - 1)
            clean_file = self.clean_files[idx]
            clean = self._load_and_process_audio(clean_file)

        # Ensure correct length
        if clean.shape[1] > self.sub_sample_length:
            start = random.randint(0, clean.shape[1] - self.sub_sample_length)
            clean = clean[:, start:start + self.sub_sample_length]
        else:
            # Pad if too short
            padding = self.sub_sample_length - clean.shape[1]
            clean = torch.nn.functional.pad(clean, (0, padding))

        # Generate noise of matching length
        noise = self._get_noise_segment(length=self.sub_sample_length)

        # Randomly select an SNR value
        snr = random.uniform(self.snr_range[0], self.snr_range[1])
        # Mix clean and noise with the selected SNR
        noisy, clean = self._mix_with_snr(clean, noise, snr)

        return noisy, clean
