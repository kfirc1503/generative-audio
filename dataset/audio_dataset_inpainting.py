import random
from pathlib import Path
from typing import Union, Tuple, Optional
import torch
import torchaudio
import pydantic
from torch.utils.data import Dataset


class AudioInpaintingConfig(pydantic.BaseModel):
    """Configuration for audio inpainting dataset"""
    clean_path: Union[str, Path]
    sample_rate: int = 16000
    missing_length_seconds: float = 1.0
    missing_start_seconds: Optional[float] = None  # If None, will be random
    missing_end_seconds: Optional[float] = None  # If None, will be computed from start + length
    sub_sample_length_seconds: float = 3.0
    target_dB_FS: float = -25.0
    target_dB_FS_floating_value: float = 0.0

    # Computed fields
    sub_sample_length: int = pydantic.Field(None)
    missing_length: int = pydantic.Field(None)

    @pydantic.model_validator(mode='after')
    def compute_lengths(self) -> 'AudioInpaintingConfig':
        """Compute sample lengths after initialization"""
        self.sub_sample_length = int(self.sub_sample_length_seconds * self.sample_rate)
        self.missing_length = int(self.missing_length_seconds * self.sample_rate)
        return self


class AudioInpaintingDataset(Dataset):
    def __init__(self, config: AudioInpaintingConfig):
        """
        Dataset for audio inpainting that creates masked segments in clean audio.

        Args:
            config (AudioInpaintingConfig): Configuration object containing dataset parameters
        """
        self.config = config
        self.clean_path = Path(config.clean_path).resolve()
        self.clean_files = list(self.clean_path.rglob("*.wav"))

        if not self.clean_files:
            raise ValueError(f"No WAV files found in clean directory: {self.clean_path}")

    def __len__(self) -> int:
        return len(self.clean_files)

    def _load_and_process_audio(self, file_path: Path) -> Union[torch.Tensor, None]:
        """Load and preprocess audio file"""
        try:
            waveform, sr = torchaudio.load(file_path)

            # Convert to mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if necessary
            if sr != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.config.sample_rate
                )
                waveform = resampler(waveform)

            return waveform

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def _normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio to target dB FS"""
        if self.config.target_dB_FS_floating_value > 0.0:
            target_db = random.uniform(
                self.config.target_dB_FS - self.config.target_dB_FS_floating_value,
                self.config.target_dB_FS + self.config.target_dB_FS_floating_value
            )
        else:
            target_db = self.config.target_dB_FS

        rms = waveform.pow(2).mean().sqrt()
        rms_db = 20 * torch.log10(rms + 1e-8)
        gain_db = target_db - rms_db
        gain = 10 ** (gain_db / 20)
        return waveform * gain

    def _create_mask(self, audio_length: int) -> torch.Tensor:
        """Create binary mask for inpainting (1 for kept samples, 0 for missing)"""
        mask = torch.ones(1, audio_length)

        if self.config.missing_start_seconds is None:
            # Random start position that ensures the gap fits within the audio
            max_start = audio_length - self.config.missing_length
            start_idx = random.randint(0, max_start)
        else:
            start_idx = int(self.config.missing_start_seconds * self.config.sample_rate)

        end_idx = start_idx + self.config.missing_length
        mask[:, start_idx:end_idx] = 0

        return mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple: (masked_audio, mask, clean_audio)
                - masked_audio: Audio with masked segment set to 0
                - mask: Binary mask (1 for kept samples, 0 for missing)
                - clean_audio: Original clean audio
        """
        clean_file = self.clean_files[idx]
        clean_audio = self._load_and_process_audio(clean_file)

        if clean_audio is None:
            # Handle error case by returning a different file
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Normalize audio
        clean_audio = self._normalize_audio(clean_audio)

        # Create random subsegment if needed
        if clean_audio.shape[1] > self.config.sub_sample_length:
            max_start = clean_audio.shape[1] - self.config.sub_sample_length
            start_idx = random.randint(0, max_start)
            clean_audio = clean_audio[:, start_idx:start_idx + self.config.sub_sample_length]

        # Create mask and masked audio
        mask = self._create_mask(clean_audio.shape[1])
        masked_audio = clean_audio * mask

        return masked_audio, mask, clean_audio