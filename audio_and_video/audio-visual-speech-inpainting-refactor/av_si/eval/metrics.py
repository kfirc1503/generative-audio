"""Evaluation metrics: L1 (log-mag), STOI, PESQ, PER."""
from typing import List

import numpy as np
import torch


def l1_log_mag(target_log: torch.Tensor, estimate_log: torch.Tensor, mask: torch.Tensor = None) -> float:
    """Mean |target - estimate| in log-magnitude domain. If mask given (1 reliable / 0 lost),
    score only the lost tiles (matches paper Table 1 'L1 N')."""
    diff = (target_log - estimate_log).abs()
    if mask is not None:
        lost = 1.0 - mask
        return (diff * lost).sum().item() / lost.sum().clamp(min=1).item()
    return diff.mean().item()


def stoi(reference: np.ndarray, estimate: np.ndarray, sample_rate: int = 16000, extended: bool = False) -> float:
    from pystoi import stoi as _stoi
    return float(_stoi(reference, estimate, sample_rate, extended=extended))


def pesq(reference: np.ndarray, estimate: np.ndarray, sample_rate: int = 16000, mode: str = "nb") -> float:
    """ITU-T P.862. mode: 'nb' (8 kHz) or 'wb' (16 kHz). Returns NaN on PESQ failure."""
    from pesq import pesq as _pesq, PesqError
    try:
        return float(_pesq(sample_rate, reference, estimate, mode))
    except PesqError:
        return float("nan")


def per(ref: List[int], hyp: List[int]) -> float:
    """Levenshtein phone error rate."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / n


def ctc_greedy_decode(log_probs: torch.Tensor, blank: int = 0) -> List[int]:
    """Collapse CTC log-probs (T, C) → phone label sequence."""
    ids = log_probs.argmax(dim=-1).tolist()
    out, prev = [], None
    for x in ids:
        if x != prev and x != blank:
            out.append(x)
        prev = x
    return out
