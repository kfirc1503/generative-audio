import torch
import torch.nn as nn
import pydantic
from typing import Tuple
import torch.nn.functional as F


class UNetConfig(pydantic.BaseModel):
    in_channels: int = 2
    out_channels: int = 2
    channels_list: Tuple[int, ...] = (32, 64, 128, 256)
    bottleneck_channels: int = 512
    min_channels_decoder: int = 64
    n_groups: int = 8


class UNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        ch = config.in_channels

        # Encoder
        self.encoder_blocks = nn.ModuleList([])
        ch_hidden_list = []

        # Initial block
        layers = []
        # Using padding='same' for PyTorch >= 2.0
        layers.append(nn.Conv2d(ch, config.channels_list[0], kernel_size=3, padding='same'))
        ch = config.channels_list[0]
        self.encoder_blocks.append(nn.Sequential(*layers))
        ch_hidden_list.append(ch)

        for i_level in range(len(config.channels_list)):
            ch_ = config.channels_list[i_level]
            downsample = i_level != 0

            layers = []
            if downsample:
                layers.append(nn.MaxPool2d(2))
            layers.append(nn.Conv2d(ch, ch_, kernel_size=3, padding='same'))
            ch = ch_
            layers.append(nn.GroupNorm(config.n_groups, ch))
            layers.append(nn.LeakyReLU(0.1))
            self.encoder_blocks.append(nn.Sequential(*layers))
            ch_hidden_list.append(ch)

        # Bottleneck
        ch_ = config.bottleneck_channels
        layers = []
        layers.append(nn.Conv2d(ch, ch_, kernel_size=3, padding='same'))
        ch = ch_
        layers.append(nn.GroupNorm(config.n_groups, ch))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Conv2d(ch, ch, kernel_size=3, padding='same'))
        layers.append(nn.GroupNorm(config.n_groups, ch))
        layers.append(nn.LeakyReLU(0.1))
        self.bottleneck = nn.Sequential(*layers)

        # Decoder
        self.decoder_blocks = nn.ModuleList([])
        for i_level in reversed(range(len(config.channels_list))):
            ch_ = max(config.channels_list[i_level], config.min_channels_decoder)
            downsample = i_level != 0
            ch = ch + ch_hidden_list.pop()
            layers = []

            layers.append(nn.Conv2d(ch, ch_, kernel_size=3, padding='same'))
            ch = ch_
            layers.append(nn.GroupNorm(config.n_groups, ch))
            layers.append(nn.LeakyReLU(0.1))
            if downsample:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.decoder_blocks.append(nn.Sequential(*layers))

        ch = ch + ch_hidden_list.pop()
        layers = []
        layers.append(nn.Conv2d(ch, config.out_channels, kernel_size=1, padding='same'))
        self.decoder_blocks.append(nn.Sequential(*layers))

    def forward(self, x_in):
        # Store original dimensions
        orig_freq_dim = x_in.size(2)
        orig_time_dim = x_in.size(3)

        x = x_in
        h = []
        for block in self.encoder_blocks:
            x = block(x)
            h.append(x)

        x = self.bottleneck(x)
        for block in self.decoder_blocks:
            enc_feat = h.pop()
            # Crop enc_feat if necessary to match x's dimensions
            # x: [B, Cx, Hx, Wx], enc_feat: [B, Ce, He, We]
            diffH = enc_feat.size(2) - x.size(2)
            diffW = enc_feat.size(3) - x.size(3)

            # Crop only if diff is positive
            if diffH > 0 or diffW > 0:
                enc_feat = enc_feat[:, :,
                                    diffH // 2:enc_feat.size(2) - (diffH - diffH // 2),
                                    diffW // 2:enc_feat.size(3) - (diffW - diffW // 2)]

            x = torch.cat((x, enc_feat), dim=1)
            x = block(x)
        # Now x may have dimensions slightly off from (orig_F, orig_T)
        # Use interpolate to match exactly
        x = F.interpolate(x, size=(orig_freq_dim, orig_time_dim), mode='nearest')
        return x





class RestorationWrapper(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config
        self.net = UNet(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x shape: (B, 2, F, T) [real, imag]
        x_in = x
        # Normalize both real and imag
        x = self.net(x)

        # Broadcast mask if needed
        if mask.shape[1] == 1:
            mask = mask.expand(-1, x.shape[1], -1, -1)
        else:
            mask = mask
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = mask.expand(-1, x.shape[1], x.shape[2], -1)

        # Apply inpainting
        x = x_in * (1 - mask) + x * mask
        return x
