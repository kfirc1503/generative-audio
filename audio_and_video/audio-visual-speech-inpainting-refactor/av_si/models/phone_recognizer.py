"""Standalone phone recognizer for PER (paper footnote 1: 2-layer BLSTM × 250 + FC + softmax)."""
import torch
import torch.nn as nn


class PhoneRecognizer(nn.Module):
    def __init__(self, n_phones: int, audio_feat_dim: int = 257, hidden_size: int = 250, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.blstm = nn.LSTM(
            input_size=audio_feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(2 * hidden_size, n_phones + 1)  # +1 = CTC blank

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        h, _ = self.blstm(spec)
        return self.head(h)
