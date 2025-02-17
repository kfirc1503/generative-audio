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
    mask_start_idx: int  # Start index of the masked region
    mask_end_idx: int  # End index of the masked region
    mask_start_frame_idx: int
    mask_end_frame_idx: int
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
    missing_length_seconds: float = 0.128  # 128ms default
    missing_start_seconds: Optional[float] = None  # If None, will be random
    missing_end_seconds: Optional[float] = None  # If None, will be computed from start + length
    sub_sample_length_seconds: float = 3.0
    target_dB_FS: float = -25.0
    target_dB_FS_floating_value: float = 0.0
    stft_configuration: StftConfig
    use_vad: bool = False  # Whether to use VAD for masking
    seed: Optional[int] = None  # Added seed parameter
    is_random_sub_sample: bool = True
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
                    # Format: {file-id} {transcription}
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
        file_id = file_path.stem  # Gets filename without extension
        return self.transcriptions.get(file_id, "")

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

    def _create_random_mask(self, audio_length: int) -> Tuple[torch.Tensor, int, int]:
        """Create a random mask for inpainting"""
        mask = torch.ones(1, audio_length)
        if self.config.missing_start_seconds is None:
            max_start = audio_length - self.config.missing_length
            start_idx = random.randint(0, max_start)
        else:
            start_idx = int(self.config.missing_start_seconds * self.config.sample_rate)

        end_idx = start_idx + self.config.missing_length
        mask[:, start_idx:end_idx] = 0
        return mask, start_idx, end_idx

    def _create_mask(self, audio_length: int, file_path: Path, audio: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Create binary mask for inpainting and return mask indices"""
        # If not using VAD, return random mask
        if not self.config.use_vad:
            return self._create_random_mask(audio_length)

        # Try to get speech segments using VAD
        valid_segments = self.get_speech_timestamps(
            audio,
            self.model,
            threshold=0.5,
            sampling_rate=self.config.sample_rate,
            min_speech_duration_ms=int(self.config.missing_length_seconds * 1000),
            return_seconds=False
        )

        # If no valid segments found, fall back to random masking
        if not valid_segments:
            return self._create_random_mask(audio_length)

        margin_in_samples = 2000
        valid_segments = [
            seg for seg in valid_segments
            if (seg['end'] - seg['start']) >= self.config.missing_length + margin_in_samples
        ]

        # Randomly choose one of the valid speech segments
        segment = random.choice(valid_segments)
        segment_start = segment['start']
        segment_end = segment['end']

        # Calculate valid start positions within the segment
        segment_length = segment_end - segment_start
        if segment_length <= self.config.missing_length:
            return self._create_random_mask(audio_length)  # Segment too short, fall back to random
        small_margin = 350
        max_start = segment_length - self.config.missing_length
        relative_start = random.randint(small_margin, max_start - small_margin)
        mask_start = segment_start + relative_start
        mask_end = mask_start + self.config.missing_length

        # Create and apply the mask
        mask = torch.ones(1, audio_length)
        mask[:, mask_start:mask_end] = 0
        return mask, mask_start, mask_end

    # def time_to_spec_mask(self, mask_time, T_frames, waveform_length, center=True):
    #     """Convert time-domain mask to spectrogram mask ensuring consistent size"""
    #     win_length = self.config.stft_configuration.win_length
    #     hop_length = self.config.stft_configuration.hop_length
    #
    #     assert mask_time.dim() == 2 and mask_time.shape[0] == 1, "mask_time should be [1, T] shape."
    #
    #     mask_frames = []
    #     half_window = win_length // 2
    #
    #     # Pre-calculate valid frame range
    #     if center:
    #         valid_start = -half_window
    #         valid_end = waveform_length + half_window
    #     else:
    #         valid_start = 0
    #         valid_end = waveform_length
    #
    #     for t_frame in range(T_frames):
    #         frame_start = t_frame * hop_length
    #         if center:
    #             frame_start -= half_window
    #         frame_end = frame_start + win_length
    #
    #         # Ensure frame boundaries are valid
    #         frame_start = max(valid_start, frame_start)
    #         frame_end = min(valid_end, frame_end)
    #
    #         if frame_end <= frame_start:
    #             frame_mask_value = 0.0
    #         else:
    #             # Map frame boundaries to valid indices in mask_time
    #             valid_start_idx = max(0, frame_start)
    #             valid_end_idx = min(waveform_length, frame_end)
    #             frame_values = mask_time[0, valid_start_idx:valid_end_idx]
    #             frame_mask_value = float((frame_values.min() == 1))
    #
    #         mask_frames.append(frame_mask_value)
    #
    #     spec_mask = torch.tensor(mask_frames)
    #
    #     # Verify spec mask consistency
    #     num_zeros = (spec_mask == 0).sum()
    #     expected_zeros = int(self.config.missing_length_seconds * self.config.sample_rate / hop_length)
    #     assert abs(
    #         num_zeros - expected_zeros) <= 1, f"Inconsistent spec mask size: got {num_zeros}, expected {expected_zeros}"
    #     return spec_mask


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

        return torch.tensor(mask_frames)

    def __getitem__(self, idx: int) -> AudioInpaintingSample:
        """Returns an AudioInpaintingSample object containing all information about the sample"""
        # Set seed based on index for consistent sampling
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

        # Get LibriSpeech transcription
        transcription = self._get_librispeech_info(clean_file)

        # Normalize audio
        full_audio = self._normalize_audio(full_audio)

        # Skip if audio is too short
        if full_audio.shape[1] < self.config.sub_sample_length:
            return self.__getitem__((idx + 1) % len(self))

        # Create random subsegment if needed
        subsample_start_idx = 0
        if full_audio.shape[1] > self.config.sub_sample_length:
            max_start = full_audio.shape[1] - self.config.sub_sample_length
            if self.config.is_random_sub_sample:
                subsample_start_idx = random.randint(0, max_start)
            clean_audio = full_audio[:, subsample_start_idx:subsample_start_idx + self.config.sub_sample_length]
        else:
            clean_audio = full_audio

        # Create mask and masked audio
        mask, mask_start_idx, mask_end_idx = self._create_mask(clean_audio.shape[1], clean_file, clean_audio)
        masked_audio = clean_audio * mask

        # Restore random states if seed was set
        if self.config.seed is not None:
            torch.set_rng_state(rng_state)
            random.setstate(random_state)
            np.random.set_state(np_state)

        # Convert to STFT
        device = torch.device("cpu")
        stft_clean = audio_to_stft(clean_audio, self.config.stft_configuration, device)

        # Convert the mask into a spec mask
        mask_frames = self.time_to_spec_mask(mask, stft_clean.shape[3], masked_audio.shape[1])

        # Find the start and end frames (where mask_frames is 0)
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