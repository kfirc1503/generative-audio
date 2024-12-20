import torch
import torch.nn as nn
import pydantic
from typing import Tuple

from nppc.auxil import NetWrapper


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
        layers.append(nn.ZeroPad2d(2))
        ch_ = config.channels_list[0]
        layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
        ch = ch_
        self.encoder_blocks.append(nn.Sequential(*layers))
        ch_hidden_list.append(ch)

        for i_level in range(len(config.channels_list)):
            ch_ = config.channels_list[i_level]
            downsample = i_level != 0

            layers = []
            if downsample:
                layers.append(nn.MaxPool2d(2))
            layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
            ch = ch_
            layers.append(nn.GroupNorm(config.n_groups, ch))
            layers.append(nn.LeakyReLU(0.1))
            self.encoder_blocks.append(nn.Sequential(*layers))
            ch_hidden_list.append(ch)

        # Bottleneck
        ch_ = config.bottleneck_channels
        layers = []
        layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
        ch = ch_
        layers.append(nn.GroupNorm(config.n_groups, ch))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Conv2d(ch, ch, 3, padding=1))
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

            layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
            ch = ch_
            layers.append(nn.GroupNorm(config.n_groups, ch))
            layers.append(nn.LeakyReLU(0.1))
            if downsample:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.decoder_blocks.append(nn.Sequential(*layers))

        ch = ch + ch_hidden_list.pop()
        layers = []
        layers.append(nn.Conv2d(ch, config.out_channels, 1))
        layers.append(nn.ZeroPad2d(-2))
        self.decoder_blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        h = []
        for block in self.encoder_blocks:
            x = block(x)
            h.append(x)

        x = self.bottleneck(x)
        for block in self.decoder_blocks:
            x = torch.cat((x, h.pop()), dim=1)
            x = block(x)
        return x


# class RestorationWrapper(nn.Module):
#     def __init__(self, net: nn.Module, mask: torch.Tensor):
#         super().__init__()
#         self.net = net
#         self.mask = mask
#
#     def forward(self, x: torch.Tensor):
#         x_in = x
#         x = self.net(x)
#
#         if self.mask.shape[1] == 1:
#             mask = self.mask.expand(-1, x.shape[1], -1, -1)
#         else:
#             mask = self.mask
#         x = x_in * (1 - mask) + x * mask
#         return x


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
        # Apply inpainting
        x = x_in * (1 - mask) + x * mask
        return x
