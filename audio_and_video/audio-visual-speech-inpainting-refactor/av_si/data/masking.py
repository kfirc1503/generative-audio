"""Random gap mask generation (Morrone §3.1, port of dataset_generator.get_intrusions_mask).

Mask convention: 1 = reliable, 0 = lost (matches upstream code; complement of paper M(k,l)).
"""
import math
import random
from typing import Optional

import numpy as np


def sample_gap_mask(
    spec_len: int,
    freq_bins: int = 257,
    *,
    cov_mean: float = 0.30,        # 900 ms / 3000 ms (paper default for variable masks)
    cov_std: float = 0.10,         # 300 ms / 3000 ms
    n_max_intr: int = 8,
    min_intr_frames: int = 3,      # ≥ 36 ms at 12 ms hop
    max_total_cov: float = 0.80,
    rng: Optional[random.Random] = None,
):
    rng = rng or random
    n_intr = rng.randint(1, n_max_intr)

    floor = min_intr_frames * n_intr / spec_len
    ratio = max(floor, min(rng.gauss(cov_mean, cov_std), max_total_cov))
    mask_frames = int(round(spec_len * ratio))
    true_coverage = mask_frames / spec_len

    intr_lens = []
    for i in range(n_intr):
        if i == n_intr - 1:
            intr_lens.append(mask_frames - sum(intr_lens))
        elif i == 0:
            cap = max(min_intr_frames, int((mask_frames - min_intr_frames * (n_intr - 1)) * math.exp(-(n_intr - 1) / 6)))
            intr_lens.append(rng.randint(min_intr_frames, cap))
        else:
            cap = max(min_intr_frames, int((mask_frames - sum(intr_lens) - min_intr_frames * (n_intr - i - 1)) * math.exp(-(n_intr - 1) / 6)))
            intr_lens.append(rng.randint(min_intr_frames, cap))
    rng.shuffle(intr_lens)

    onsets = []
    for i, length in enumerate(intr_lens):
        if i == 0 and i == n_intr - 1:
            onsets.append(rng.randint(0, max(0, spec_len - mask_frames)))
        elif i == 0:
            onsets.append(rng.randint(0, max(0, (spec_len - mask_frames - (n_intr - 1)) // 2)))
        elif i == n_intr - 1:
            onsets.append(rng.randint(onsets[-1], onsets[-1] + intr_lens[i - 1] + 1 + spec_len - intr_lens[i]))
        else:
            lo = onsets[-1] + intr_lens[i - 1] + 1
            hi = max(lo, (onsets[-1] + intr_lens[i - 1] + 1 + spec_len - sum(intr_lens[i:]) - (n_intr - i - 1)) // 2)
            onsets.append(rng.randint(lo, hi))

    mask = np.ones((spec_len, freq_bins), dtype=np.float32)
    for o, length in zip(onsets, intr_lens):
        mask[o : o + length] = 0.0
    return mask, true_coverage, n_intr


def fixed_gap_mask(spec_len: int, freq_bins: int, gap_frames: int, rng: Optional[random.Random] = None) -> np.ndarray:
    """Single fixed-length gap, randomly placed. Matches the per-gap-size test sets in Fig. 2."""
    rng = rng or random
    onset = rng.randint(0, spec_len - gap_frames)
    mask = np.ones((spec_len, freq_bins), dtype=np.float32)
    mask[onset : onset + gap_frames] = 0.0
    return mask
