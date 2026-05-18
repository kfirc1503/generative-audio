"""GRID audio-visual inpainting Dataset.

Expects per-speaker layout:

  <root>/s<spk>/s<spk>_16kHz/<utt>.wav         16 kHz audio
  <root>/s<spk>/s<spk>.landmarks/<utt>.npy     raw 68×2 landmarks @ 25 fps
  <root>/s<spk>/s<spk>.landmarks/video_feat_mean.npy
  <root>/s<spk>/s<spk>.landmarks/video_feat_std.npy
  <root>/s<spk>/align/<utt>.align              GRID alignment (optional, for CTC/PER)

Audio mean/std (global, training-set) is shared across speakers and lives at <root>/audio_feat_mean.npy and audio_feat_std.npy.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from . import audio as A
from . import landmarks as LM
from .masking import sample_gap_mask


PAPER_TRAIN = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20, 22,23,24,25, 28]
PAPER_VAL   = [26, 27, 29, 31]
PAPER_TEST  = [30, 32, 33, 34]


class GridAVDataset(Dataset):
    def __init__(
        self,
        root: str,
        speakers: List[int],
        *,
        audio_len_samples: int = 48000,         # 3 s @ 16 kHz
        audio_mean_path: Optional[str] = None,
        audio_std_path: Optional[str] = None,
        load_video: bool = True,
        load_phones: bool = False,
        phone_dict: Optional[List[str]] = None,
        word_dict: Optional[Tuple[List[str], List[str]]] = None,
        mask_kwargs: Optional[dict] = None,
    ):
        self.root = Path(root)
        self.audio_len_samples = audio_len_samples
        self.load_video = load_video
        self.load_phones = load_phones
        self.phone_dict = phone_dict
        self.word_dict = word_dict
        self.mask_kwargs = mask_kwargs or {}

        self.utts: List[Tuple[int, Path]] = []
        for spk in speakers:
            wav_dir = self.root / f"s{spk}" / f"s{spk}_16kHz"
            self.utts.extend((spk, p) for p in sorted(wav_dir.glob("*.wav")))

        if audio_mean_path and audio_std_path:
            self.audio_mean = torch.from_numpy(np.load(audio_mean_path)).float()
            self.audio_std = torch.from_numpy(np.load(audio_std_path)).float()
        else:
            self.audio_mean = torch.zeros(A.N_FREQ_BINS)
            self.audio_std = torch.ones(A.N_FREQ_BINS)

    def __len__(self):
        return len(self.utts)

    def _load_audio(self, wav_path: Path) -> torch.Tensor:
        wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        assert sr == A.SAMPLE_RATE, f"expected 16 kHz, got {sr}"
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if len(wav) >= self.audio_len_samples:
            wav = wav[: self.audio_len_samples]
        else:
            wav = np.pad(wav, (0, self.audio_len_samples - len(wav)))
        return torch.from_numpy(wav)

    def _load_video(self, spk: int, utt: str, n_audio_frames: int) -> torch.Tensor:
        lm_dir = self.root / f"s{spk}" / f"s{spk}.landmarks"
        landmarks = np.load(lm_dir / f"{utt}.npy")             # (T_v, 68, 2)
        mean = np.load(lm_dir / "video_feat_mean.npy")
        std = np.load(lm_dir / "video_feat_std.npy")
        feat = LM.motion_vector(landmarks)                     # (T_v, 68, 2)
        feat = (feat - mean) / (std + 1e-8)
        feat = feat.reshape(feat.shape[0], -1)                 # (T_v, 136)
        feat = LM.upsample_to_audio_rate(feat, n_audio_frames) # (T_a, 136)
        return torch.from_numpy(feat).float()

    def _load_phones(self, spk: int, utt: str) -> torch.Tensor:
        from .phones import align_to_phone_labels
        align_path = self.root / f"s{spk}" / "align" / f"{utt}.align"
        words, dicts = self.word_dict
        labels = align_to_phone_labels(str(align_path), words, dicts, self.phone_dict)
        return torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        spk, wav_path = self.utts[idx]
        utt = wav_path.stem

        wav = self._load_audio(wav_path)
        complex_spec = A.stft(wav.unsqueeze(0)).squeeze(0)          # (T, F)
        target_log = A.log_magnitude(complex_spec)                  # (T, F)
        target_norm = (target_log - self.audio_mean) / self.audio_std

        T = target_norm.shape[0]
        mask_np, _, _ = sample_gap_mask(T, A.N_FREQ_BINS, **self.mask_kwargs)
        mask = torch.from_numpy(mask_np)
        audio_in = target_norm * mask

        item = {
            "audio_in": audio_in,
            "mask": mask,
            "target_norm": target_norm,
            "phase": torch.angle(complex_spec),
            "wav": wav,
            "utt_id": f"s{spk}_{utt}",
        }
        if self.load_video:
            item["video_in"] = self._load_video(spk, utt, T)
        if self.load_phones:
            item["phones"] = self._load_phones(spk, utt)
        return item
