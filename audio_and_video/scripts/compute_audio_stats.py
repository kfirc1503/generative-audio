"""Compute global mean/std of log-magnitude STFT over a speaker subset, save as .npy.

Streams over wavs and accumulates Welford-style sums; constant memory.
Usage:
    python compute_audio_stats.py                   # train split (paper PAPER_TRAIN)
    python compute_audio_stats.py --speakers 1 2 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audio-visual-speech-inpainting-refactor"))
from av_si.data import audio as A
from av_si.data.grid_dataset import PAPER_TRAIN


def stream_speaker(spk_dir: Path):
    for wav_path in sorted(spk_dir.glob("*.wav")):
        wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        assert sr == A.SAMPLE_RATE
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if len(wav) < A.WIN_LENGTH:
            continue
        spec = A.stft(torch.from_numpy(wav).unsqueeze(0))
        yield A.log_magnitude(spec).squeeze(0).numpy()  # (T, F)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/storage/kfir/data/audio_and_video/grid_corpus/extracted"))
    p.add_argument("--speakers", type=int, nargs="*", default=None)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="where to write audio_feat_mean.npy / audio_feat_std.npy (default: <root>/)")
    args = p.parse_args()

    speakers = args.speakers or PAPER_TRAIN
    out_dir = args.out_dir or args.root
    out_dir.mkdir(parents=True, exist_ok=True)

    F = A.N_FREQ_BINS
    n_frames = 0
    sum_ = np.zeros(F, dtype=np.float64)
    sum_sq = np.zeros(F, dtype=np.float64)

    print(f"speakers ({len(speakers)}): {speakers}")
    for spk in speakers:
        spk_dir = args.root / f"s{spk}" / f"s{spk}_16kHz"
        if not spk_dir.is_dir():
            print(f"  [skip] s{spk}: missing {spk_dir}")
            continue
        wavs = list(spk_dir.glob("*.wav"))
        for log_mag in stream_speaker(spk_dir):
            n_frames += log_mag.shape[0]
            sum_ += log_mag.sum(axis=0)
            sum_sq += (log_mag.astype(np.float64) ** 2).sum(axis=0)
        print(f"  s{spk:<2}  {len(wavs)} wavs, total frames so far: {n_frames}")

    mean = (sum_ / n_frames).astype(np.float32)
    var = (sum_sq / n_frames - mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    np.save(out_dir / "audio_feat_mean.npy", mean)
    np.save(out_dir / "audio_feat_std.npy", std)
    print(f"\nsaved → {out_dir}/audio_feat_mean.npy, audio_feat_std.npy")
    print(f"  mean: shape={mean.shape}  range=[{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  std:  shape={std.shape}  range=[{std.min():.3f}, {std.max():.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
