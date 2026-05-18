"""Stacked BLSTM speech inpainter — PyTorch port of Morrone et al. 2021 §2.1."""
from typing import Optional

import torch
import torch.nn as nn


class StackedBLSTM(nn.Module):
    def __init__(
        self,
        input_mode: str = "av",
        audio_feat_dim: int = 257,
        video_feat_dim: int = 136,
        hidden_size: int = 250,
        num_layers: int = 3,
        dropout: float = 0.0,
        n_phones: Optional[int] = None,
    ):
        super().__init__()
        if input_mode not in {"a", "v", "av"}:
            raise ValueError(f"input_mode must be 'a' | 'v' | 'av', got {input_mode!r}")
        self.input_mode = input_mode
        self.audio_feat_dim = audio_feat_dim
        self.video_feat_dim = video_feat_dim

        in_dim = {
            "a": audio_feat_dim,
            "v": video_feat_dim,
            "av": audio_feat_dim + video_feat_dim,
        }[input_mode]

        self.blstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.spec_head = nn.Linear(2 * hidden_size, audio_feat_dim)
        self.ctc_head = nn.Linear(2 * hidden_size, n_phones) if n_phones is not None else None

    def forward(
        self,
        audio_in: torch.Tensor,
        mask: torch.Tensor,
        video_in: Optional[torch.Tensor] = None,
    ) -> dict:
        # audio_in: (B, T, audio_feat_dim) — normalized log-magnitude, lost tiles zeroed.
        # mask:     (B, T, audio_feat_dim) — 1 = reliable, 0 = lost (upstream convention).
        # video_in: (B, T, video_feat_dim) when input_mode in {"v", "av"}.
        if self.input_mode == "a":
            x = audio_in
        elif self.input_mode == "v":
            x = video_in
        else:
            x = torch.cat([audio_in, video_in], dim=-1)

        h, _ = self.blstm(x)
        spec_logits = self.spec_head(h)
        # Paper Eq. (1): Y_hat = O ⊙ M_paper + X. Upstream uses the inverted
        # mask (1 = reliable), so the equivalent is O * (1 - mask) + audio_in.
        y_hat = spec_logits * (1.0 - mask) + audio_in

        out = {"y_hat": y_hat, "spec_logits": spec_logits}
        if self.ctc_head is not None:
            out["phone_logits"] = self.ctc_head(h)
        return out
