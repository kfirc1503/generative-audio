import random
from pathlib import Path
from typing import Union, Tuple, Optional
import torch
import torchaudio
import pydantic
from torch.utils.data import Dataset
from utils import audio_to_stft, StftConfig


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
    stft_configuration: StftConfig

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
        # convert the clean and the masked audio into a stft form:
        device = torch.device("cpu")
        stft_clean = audio_to_stft(clean_audio, self.config.stft_configuration, device)
        stft_masked = audio_to_stft(masked_audio, self.config.stft_configuration, device)
        # convert the mask into a spec mask:
        mask_frames = self.time_to_spec_mask(mask, stft_clean.shape[3], masked_audio.shape[1])

        return masked_audio, mask, clean_audio, stft_masked, mask_frames, stft_clean

    def time_to_spec_mask(self, mask_time, T_frames, waveform_length, center=True):
        """
        Convert a time-domain mask to a spectrogram time-frame mask.

        Args:
            mask_time (torch.Tensor): Time-domain mask of shape [1, T], where T is the number of samples.
                                      Values are expected to be binary (0 or 1).
            T_frames (int): Number of time frames in the STFT domain.
            waveform_length (int): Number of samples in the original waveform (T).
            win_length (int): Window length used in STFT.
            hop_length (int): Hop length used in STFT.
            center (bool): Whether the STFT was computed with `center=True`.
                           If True, frames are centered around t_frame*hop_length.
                           If False, frames start at t_frame*hop_length.
            device (torch.device): Optional device to place the returned mask on.

        Returns:
            torch.Tensor: A mask of shape [T_frames], where each entry is either 0 or 1.
                          1 means the entire frame is unmasked, 0 means at least one sample in that frame was masked.
        """
        win_length = self.config.stft_configuration.win_length
        hop_length = self.config.stft_configuration.hop_length

        # Check shape of mask_time
        assert mask_time.dim() == 2 and mask_time.shape[0] == 1, "mask_time should be [1, T] shape."

        mask_frames = []
        half_window = win_length // 2

        for t_frame in range(T_frames):
            if center:
                # Frame is centered
                start = t_frame * hop_length - half_window
            else:
                # Frame starts exactly at t_frame * hop_length
                start = t_frame * hop_length
            end = start + win_length

            # Clip boundaries
            start = max(start, 0)
            end = min(end, waveform_length)

            # If the window partially goes out of waveform range, that frame is shorter
            # Check if there are any samples in this frame at all
            if end <= start:
                # No samples in this frame due to clipping, consider it masked (0)
                frame_mask_value = 0.0
            else:
                frame_values = mask_time[0, start:end]
                # Frame mask is 1 only if all samples in that window are 1
                frame_mask_value = float((frame_values.min() == 1))

            mask_frames.append(frame_mask_value)

        mask_frames = torch.tensor(mask_frames)
        return mask_frames
