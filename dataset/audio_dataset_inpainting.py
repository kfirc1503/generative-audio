import random
from pathlib import Path
from typing import Union, Tuple, Optional
import torch
import torchaudio
import pydantic
from torch.utils.data import Dataset
from utils import audio_to_stft, StftConfig
import numpy as np
from dataclasses import dataclass


@dataclass
class AudioInpaintingSample:
    """Class to hold all information about an audio sample"""
    # STFT and mask data for training
    stft_masked: torch.Tensor
    mask_frames: torch.Tensor
    stft_clean: torch.Tensor
    masked_audio: torch.Tensor

    # File information
    clean_audio_path: Path
    subsample_start_idx: int  # Where the subsample starts in the original audio

    # Mask information
    mask_start_idx: int  # Start index of the masked region in time domain
    mask_end_idx: int  # End index of the masked region in time domain
    mask_start_frame_idx: int  # Start index in spectrogram frames
    mask_end_frame_idx: int  # End index in spectrogram frames
    # LibriSpeech metadata
    transcription: str
    sample_rate: int = 16000

    def get_training_tuple(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the tuple format expected by the training loop"""
        return self.stft_masked, self.mask_frames, self.stft_clean, self.masked_audio

    @property
    def mask_start_time(self) -> float:
        """Returns the start time of the mask in seconds"""
        return self.mask_start_idx / self.sample_rate

    @property
    def mask_end_time(self) -> float:
        """Returns the end time of the mask in seconds"""
        return self.mask_end_idx / self.sample_rate

    @property
    def subsample_start_time(self) -> float:
        """Returns the start time of the subsample in seconds"""
        return self.subsample_start_idx / self.sample_rate

    @property
    def mask_duration(self) -> float:
        """Returns the duration of the mask in seconds"""
        return (self.mask_end_idx - self.mask_start_idx) / self.sample_rate


class AudioInpaintingConfig(pydantic.BaseModel):
    """Configuration for audio inpainting dataset"""
    clean_path: Union[str, Path]
    sample_rate: int = 16000
    missing_spec_frames: int = 17  # Number of spectrogram frames to mask
    sub_sample_length_seconds: float = 3.0
    target_dB_FS: float = -25.0
    target_dB_FS_floating_value: float = 0.0
    stft_configuration: StftConfig
    use_vad: bool = False
    seed: Optional[int] = None
    is_random_sub_sample: bool = True
    missing_start_seconds: Optional[float] = None

    # Computed fields
    sub_sample_length: int = pydantic.Field(None)
    missing_length: int = pydantic.Field(None)  # Time domain length
    missing_length_seconds: float = pydantic.Field(None)

    @pydantic.model_validator(mode='after')
    def compute_lengths(self) -> 'AudioInpaintingConfig':
        """Compute sample lengths after initialization"""
        self.sub_sample_length = int(self.sub_sample_length_seconds * self.sample_rate)
        # Calculate time-domain mask length from spectrogram frames
        self.missing_length = self.missing_spec_frames * self.stft_configuration.hop_length
        self.missing_length_seconds = self.missing_length / self.sample_rate
        return self


class AudioInpaintingDataset(Dataset):
    def __init__(self, config: AudioInpaintingConfig):
        """
        Dataset for audio inpainting that creates masked segments in clean audio.
        Works with LibriSpeech directory structure.
        """
        self.config = config
        self.clean_path = Path(config.clean_path).resolve()

        # Get all .flac files
        self.clean_files = list(self.clean_path.rglob("*.flac"))
        if not self.clean_files:
            raise ValueError(f"No FLAC files found in LibriSpeech directory: {self.clean_path}")

        # Load transcriptions
        self.transcriptions = {}
        for trans_file in self.clean_path.rglob("*.trans.txt"):
            with open(trans_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        file_id, transcription = parts
                        self.transcriptions[file_id] = transcription

        print(f"Found {len(self.clean_files)} files in {self.clean_path}")
        print("Sample files:")
        for file in self.clean_files[:5]:
            print(f"  {file.relative_to(self.clean_path)}")

        if config.use_vad:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad'
            )
            (self.get_speech_timestamps, _, self.read_audio, _, _) = utils

    def __len__(self) -> int:
        return len(self.clean_files)

    def _get_librispeech_info(self, file_path: Path) -> str:
        """Extract LibriSpeech transcription from file path"""
        file_id = file_path.stem
        return self.transcriptions.get(file_id, "")

    def _load_and_process_audio(self, file_path: Path) -> Union[torch.Tensor, None]:
        """Load and preprocess audio file"""
        try:
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
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

    def _create_random_mask(self, audio_length: int) -> Tuple[torch.Tensor, int, int]:
        """
        Create random binary mask aligned to hop_length grid

        Args:
            audio_length: Length of the audio in samples

        Returns:
            Tuple of (mask tensor, start_idx, end_idx)
        """
        hop_length = self.config.stft_configuration.hop_length
        time_mask_length = self.config.missing_length

        mask = torch.ones(1, audio_length)
        max_start = audio_length - time_mask_length - 1

        if self.config.missing_start_seconds is None:
            start_idx = random.randint(0, max_start)
            # Align start_idx to hop_length grid
            start_idx = (start_idx // hop_length) * hop_length
        else:
            start_idx = int(self.config.missing_start_seconds * self.config.sample_rate)
            start_idx = min(start_idx, max_start)
            start_idx = (start_idx // hop_length) * hop_length

        end_idx = start_idx + time_mask_length
        mask[:, start_idx:end_idx] = 0

        # Verify mask consistency
        num_zeros = (mask == 0).sum()
        assert num_zeros == time_mask_length, (
            f"Inconsistent mask size: got {num_zeros}, expected {time_mask_length}. "
            f"Start: {start_idx}, End: {end_idx}, Audio length: {audio_length}"
        )

        return mask, start_idx, end_idx

    def _create_mask(self, audio_length: int, file_path: Path, audio: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Create binary mask starting from desired number of spectrogram frames

        Args:
            audio_length: Length of the audio in samples
            file_path: Path to audio file (for VAD)
            audio: Audio tensor [1, T]

        Returns:
            Tuple of (mask tensor, start_idx, end_idx)
        """
        if not self.config.use_vad:
            return self._create_random_mask(audio_length)

        hop_length = self.config.stft_configuration.hop_length
        time_mask_length = self.config.missing_length

        # VAD-based masking
        valid_segments = self.get_speech_timestamps(
            audio[0],
            self.model,
            threshold=0.5,
            sampling_rate=self.config.sample_rate,
            min_speech_duration_ms=int(self.config.missing_length_seconds * 1000),
            return_seconds=False
        )

        if not valid_segments:
            # print(f"No valid speech segments found in {file_path}, falling back to random mask")
            return self._create_random_mask(audio_length)

        valid_segments = [
            seg for seg in valid_segments
            if (seg['end'] - seg['start']) >= time_mask_length
        ]

        if not valid_segments:
            # print(f"No segments long enough in {file_path}, falling back to random mask")
            return self._create_random_mask(audio_length)

        segment = random.choice(valid_segments)
        segment_start = (segment['start'] // hop_length) * hop_length
        segment_end = segment['end']

        max_start = segment_end - time_mask_length
        start_idx = random.randint(segment_start, max_start)
        start_idx = (start_idx // hop_length) * hop_length
        end_idx = start_idx + time_mask_length

        mask = torch.ones(1, audio_length)
        mask[:, start_idx:end_idx] = 0

        # Verify mask consistency
        num_zeros = (mask == 0).sum()
        assert num_zeros == time_mask_length, (
            f"Inconsistent mask size: got {num_zeros}, expected {time_mask_length}. "
            f"Start: {start_idx}, End: {end_idx}, Audio length: {audio_length}"
        )

        return mask, start_idx, end_idx

    def time_to_spec_mask(self, mask_time: torch.Tensor, T_frames: int, waveform_length: int) -> torch.Tensor:
        """Convert time-domain mask to spectrogram mask ensuring consistent size"""
        win_length = self.config.stft_configuration.win_length
        hop_length = self.config.stft_configuration.hop_length

        assert mask_time.dim() == 2 and mask_time.shape[0] == 1, "mask_time should be [1, T] shape."

        # Convert time indices to frame indices
        time_zeros = torch.where(mask_time[0] == 0)[0]
        frame_start = time_zeros[0] // hop_length
        frame_end = frame_start + self.config.missing_spec_frames

        # Create spectrogram mask
        spec_mask = torch.ones(T_frames)
        spec_mask[frame_start:frame_end] = 0

        # Verify mask consistency
        num_zeros = (spec_mask == 0).sum()
        assert num_zeros == self.config.missing_spec_frames, (
            f"Inconsistent spec mask size: got {num_zeros}, "
            f"expected {self.config.missing_spec_frames}"
        )

        return spec_mask

    def __getitem__(self, idx: int) -> AudioInpaintingSample:
        """Returns an AudioInpaintingSample object containing all information about the sample"""
        if self.config.seed is not None:
            rng_state = torch.get_rng_state()
            random_state = random.getstate()
            np_state = np.random.get_state()
            seed = self.config.seed + idx
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        clean_file = self.clean_files[idx]
        full_audio = self._load_and_process_audio(clean_file)
        if full_audio is None:
            return self.__getitem__((idx + 1) % len(self))

        transcription = self._get_librispeech_info(clean_file)
        full_audio = self._normalize_audio(full_audio)

        if full_audio.shape[1] < self.config.sub_sample_length:
            return self.__getitem__((idx + 1) % len(self))

        subsample_start_idx = 0
        if full_audio.shape[1] > self.config.sub_sample_length:
            max_start = full_audio.shape[1] - self.config.sub_sample_length
            if self.config.is_random_sub_sample:
                subsample_start_idx = random.randint(0, max_start)
            clean_audio = full_audio[:, subsample_start_idx:subsample_start_idx + self.config.sub_sample_length]
        else:
            clean_audio = full_audio

        mask, mask_start_idx, mask_end_idx = self._create_mask(clean_audio.shape[1], clean_file, clean_audio)
        masked_audio = clean_audio * mask

        if self.config.seed is not None:
            torch.set_rng_state(rng_state)
            random.setstate(random_state)
            np.random.set_state(np_state)

        device = torch.device("cpu")
        stft_clean = audio_to_stft(clean_audio, self.config.stft_configuration, device)

        mask_frames = self.time_to_spec_mask(mask, stft_clean.shape[3], masked_audio.shape[1])

        # Find the start and end frames
        zero_frames = torch.where(mask_frames == 0)[0]
        mask_start_frame = zero_frames[0].item()
        mask_end_frame = zero_frames[-1].item()

        mask_expand = mask_frames.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        mask_expand = mask_expand.expand(-1, stft_clean.shape[1], stft_clean.shape[2], -1)
        stft_masked = stft_clean * mask_expand
        stft_masked = stft_masked.squeeze(0)
        stft_clean = stft_clean.squeeze(0)

        return AudioInpaintingSample(
            stft_masked=stft_masked,
            mask_frames=mask_frames,
            stft_clean=stft_clean,
            masked_audio=masked_audio,
            clean_audio_path=clean_file,
            subsample_start_idx=subsample_start_idx,
            mask_start_idx=mask_start_idx,
            mask_end_idx=mask_end_idx,
            mask_start_frame_idx=mask_start_frame,
            mask_end_frame_idx=mask_end_frame,
            transcription=transcription,
            sample_rate=self.config.sample_rate
        )