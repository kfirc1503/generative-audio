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
    missing_length_seconds: float = 0.128  # 128ms default
    missing_start_seconds: Optional[float] = None  # If None, will be random
    missing_end_seconds: Optional[float] = None  # If None, will be computed from start + length
    sub_sample_length_seconds: float = 3.0
    target_dB_FS: float = -25.0
    target_dB_FS_floating_value: float = 0.0
    stft_configuration: StftConfig
    use_vad: bool = False  # Whether to use VAD for masking

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

        if config.use_vad:
            # Load VAD model
            self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                               model='silero_vad')
            (self.get_speech_timestamps, _, self.read_audio, _, _) = utils

    def __len__(self) -> int:
        return len(self.clean_files)

    def get_speech_segments(self, audio_path: Path , audio: torch.Tensor) -> list:
        """Get speech segments from audio file using VAD"""
        # wav = self.read_audio(str(audio_path))
        speech_timestamps = self.get_speech_timestamps(
            audio,
            self.model,
            threshold=0.5,
            sampling_rate=self.config.sample_rate,
            min_speech_duration_ms=int(self.config.missing_length_seconds * 1000),
            return_seconds=False
        )
        return speech_timestamps

    def _create_random_mask(self, audio_length: int) -> torch.Tensor:
        """Create a random mask"""
        mask = torch.ones(1, audio_length)
        if self.config.missing_start_seconds is None:
            max_start = audio_length - self.config.missing_length
            start_idx = random.randint(0, max_start)
        else:
            start_idx = int(self.config.missing_start_seconds * self.config.sample_rate)

        end_idx = start_idx + self.config.missing_length
        mask[:, start_idx:end_idx] = 0
        return mask

    def _create_mask(self, audio_length: int, file_path: Path , audio: torch.Tensor) -> torch.Tensor:
        """Create binary mask for inpainting (1 for kept samples, 0 for missing)"""
        if not self.config.use_vad:
            return self._create_random_mask(audio_length)

        # Get speech segments
        valid_segments = self.get_speech_segments(file_path , audio)

        if not valid_segments:
            return self._create_random_mask(audio_length)

        # Randomly choose one of the valid speech segments
        segment = random.choice(valid_segments)
        segment_start = segment['start']
        segment_end = segment['end']

        # Create mask
        mask = torch.ones(1, audio_length)

        # Calculate valid start positions within the segment
        max_start = segment_end - segment_start - self.config.missing_length
        relative_start = random.randint(0, max_start)
        mask_start = segment_start + relative_start
        mask_end = mask_start + self.config.missing_length

        # Apply the mask
        mask[:, mask_start:mask_end] = 0
        return mask

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

        # Normalize audio
        clean_audio = self._normalize_audio(clean_audio)

        # Create random subsegment if needed
        if clean_audio.shape[1] > self.config.sub_sample_length:
            max_start = clean_audio.shape[1] - self.config.sub_sample_length
            start_idx = random.randint(0, max_start)
            clean_audio = clean_audio[:, start_idx:start_idx + self.config.sub_sample_length]

        # Create mask and masked audio
        mask = self._create_mask(clean_audio.shape[1], clean_file, clean_audio)
        masked_audio = clean_audio * mask

        # Convert to STFT
        device = torch.device("cpu")
        stft_clean = audio_to_stft(clean_audio, self.config.stft_configuration, device)

        # Convert the mask into a spec mask
        mask_frames = self.time_to_spec_mask(mask, stft_clean.shape[3], masked_audio.shape[1])
        mask_expand = mask_frames.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        mask_expand = mask_expand.expand(-1, stft_clean.shape[1], stft_clean.shape[2], -1)
        stft_masked = stft_clean * mask_expand
        stft_masked = stft_masked.squeeze(0)
        stft_clean = stft_clean.squeeze(0)

        return stft_masked, mask_frames, stft_clean , masked_audio

    def time_to_spec_mask(self, mask_time, T_frames, waveform_length, center=True):
        """Convert time-domain mask to spectrogram mask"""
        win_length = self.config.stft_configuration.win_length
        hop_length = self.config.stft_configuration.hop_length

        assert mask_time.dim() == 2 and mask_time.shape[0] == 1, "mask_time should be [1, T] shape."

        mask_frames = []
        half_window = win_length // 2

        for t_frame in range(T_frames):
            if center:
                start = t_frame * hop_length - half_window
            else:
                start = t_frame * hop_length
            end = start + win_length

            start = max(start, 0)
            end = min(end, waveform_length)

            if end <= start:
                frame_mask_value = 0.0
            else:
                frame_values = mask_time[0, start:end]
                frame_mask_value = float((frame_values.min() == 1))

            mask_frames.append(frame_mask_value)

        mask_frames = torch.tensor(mask_frames)
        return mask_frames