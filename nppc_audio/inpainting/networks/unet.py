import torch
import torch.nn as nn
import torch.nn.functional as F
import pydantic
from typing import Tuple
from utils import normalize_spectrograms, denormalize_spectrograms
from nppc_audio.inpainting.networks.tmp_utils import *


# class UNetConfig(pydantic.BaseModel):
#     in_channels: int = 2
#     out_channels: int = 2
#     channels_list: Tuple[int, ...] = (32, 64, 128, 256)
#     bottleneck_channels: int = 512
#     min_channels_decoder: int = 64
#     n_groups: int = 8
#
#
# class UNet(nn.Module):
#     def __init__(self, config: UNetConfig):
#         super().__init__()
#         ch = config.in_channels
#
#         # Encoder
#         self.encoder_blocks = nn.ModuleList([])
#         ch_hidden_list = []
#
#         # Initial block
#         layers = []
#         # Using padding='same' for PyTorch >= 2.0
#         layers.append(nn.Conv2d(ch, config.channels_list[0], kernel_size=3, padding='same'))
#         ch = config.channels_list[0]
#         self.encoder_blocks.append(nn.Sequential(*layers))
#         ch_hidden_list.append(ch)
#
#         for i_level in range(len(config.channels_list)):
#             ch_ = config.channels_list[i_level]
#             downsample = i_level != 0
#
#             layers = []
#             if downsample:
#                 layers.append(nn.MaxPool2d(2))
#             layers.append(nn.Conv2d(ch, ch_, kernel_size=3, padding='same'))
#             ch = ch_
#             layers.append(nn.GroupNorm(config.n_groups, ch))
#             layers.append(nn.LeakyReLU(0.1))
#             self.encoder_blocks.append(nn.Sequential(*layers))
#             ch_hidden_list.append(ch)
#
#         # Bottleneck
#         ch_ = config.bottleneck_channels
#         layers = []
#         layers.append(nn.Conv2d(ch, ch_, kernel_size=3, padding='same'))
#         ch = ch_
#         layers.append(nn.GroupNorm(config.n_groups, ch))
#         layers.append(nn.LeakyReLU(0.1))
#         layers.append(nn.Conv2d(ch, ch, kernel_size=3, padding='same'))
#         layers.append(nn.GroupNorm(config.n_groups, ch))
#         layers.append(nn.LeakyReLU(0.1))
#         self.bottleneck = nn.Sequential(*layers)
#
#         # Decoder
#         self.decoder_blocks = nn.ModuleList([])
#         for i_level in reversed(range(len(config.channels_list))):
#             ch_ = max(config.channels_list[i_level], config.min_channels_decoder)
#             downsample = i_level != 0
#             ch = ch + ch_hidden_list.pop()
#             layers = []
#
#             layers.append(nn.Conv2d(ch, ch_, kernel_size=3, padding='same'))
#             ch = ch_
#             layers.append(nn.GroupNorm(config.n_groups, ch))
#             layers.append(nn.LeakyReLU(0.1))
#             if downsample:
#                 layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
#             self.decoder_blocks.append(nn.Sequential(*layers))
#
#         ch = ch + ch_hidden_list.pop()
#         layers = []
#         layers.append(nn.Conv2d(ch, config.out_channels, kernel_size=1, padding='same'))
#         self.decoder_blocks.append(nn.Sequential(*layers))
#
#     def forward(self, x_in):
#         # Store original dimensions
#         orig_freq_dim = x_in.size(2)
#         orig_time_dim = x_in.size(3)
#
#         x = x_in
#         h = []
#         for block in self.encoder_blocks:
#             x = block(x)
#             h.append(x)
#
#         x = self.bottleneck(x)
#         for block in self.decoder_blocks:
#             enc_feat = h.pop()
#             # Crop enc_feat if necessary to match x's dimensions
#             # x: [B, Cx, Hx, Wx], enc_feat: [B, Ce, He, We]
#             diffH = enc_feat.size(2) - x.size(2)
#             diffW = enc_feat.size(3) - x.size(3)
#
#             # Crop only if diff is positive
#             if diffH > 0 or diffW > 0:
#                 enc_feat = enc_feat[:, :,
#                                     diffH // 2:enc_feat.size(2) - (diffH - diffH // 2),
#                                     diffW // 2:enc_feat.size(3) - (diffW - diffW // 2)]
#
#             x = torch.cat((x, enc_feat), dim=1)
#             x = block(x)
#         # Now x may have dimensions slightly off from (orig_F, orig_T)
#         # Use interpolate to match exactly
#         x = F.interpolate(x, size=(orig_freq_dim, orig_time_dim), mode='nearest')
#         return x
#


# Configuration
##############################################################################
class UNetConfig(pydantic.BaseModel):
    """
    A minimal config that only lets you pick how many channels
    come in (in_channels) and go out (out_channels).
    The internal channel sizes for each encoder/decoder block
    are still fixed as per Table 1 of the paper.
    """
    in_channels: int = 1
    out_channels: int = 1


##############################################################################
# Encoder Block
##############################################################################
class EncoderBlock(nn.Module):
    """
    One "blue" encoder block:
    - 2D convolution (stride=2, 'same' padding)
    - BatchNorm2d
    - ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        pad = kernel_size // 2  # 'same' padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=pad)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


##############################################################################
# Decoder Block
##############################################################################
class DecoderBlock(nn.Module):
    """
    One "green" decoder block:
    - Upsampling (scale factor=2)
    - Concatenate skip connection
    - 2D convolution (stride=1, 'same' padding)
    - BatchNorm2d
    - LeakyReLU(0.2)
    """

    def __init__(self, in_channels, out_channels, kernel_size, final=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        pad = kernel_size // 2  # 'same' padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=pad)
        self.bn = nn.BatchNorm2d(out_channels)
        self.final = final
        if not self.final:
            self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, skip):
        x = self.upsample(x)  # 1) Upsample
        x = torch.cat([x, skip], dim=1)  # 2) Concatenate skip connection
        x = self.conv(x)  # 3) 2D convolution
        x = self.bn(x)
        if not self.final:
            x = self.act(x)
        return x


##############################################################################
# U-Net
##############################################################################
class UNet2(nn.Module):
    """
    6 encoder blocks, 6 decoder blocks.
    The final output is a 2D spectrogram [B, 1, F, T].
    """

    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config
        # Encoder (blue blocks)
        self.enc1 = EncoderBlock(config.in_channels, 16, 7)  # Block 1: (7, 16)
        self.enc2 = EncoderBlock(16, 32, 5)  # Block 2: (5, 32)
        self.enc3 = EncoderBlock(32, 64, 5)  # Block 3: (5, 64)
        self.enc4 = EncoderBlock(64, 128, 3)  # Block 4: (3, 128)
        self.enc5 = EncoderBlock(128, 128, 3)  # Block 5: (3, 128)
        self.enc6 = EncoderBlock(128, 128, 3)  # Block 6: (3, 128)
        # self.enc7 = EncoderBlock(128, 256, 3)
        #
        # # Decoder (green blocks)
        # self.dec7 = DecoderBlock(256, 256, 3)
        self.dec6 = DecoderBlock(128 + 128, 128, 3)  # Block 6
        self.dec5 = DecoderBlock(128 + 128, 128, 3)  # Block 5
        self.dec4 = DecoderBlock(128 + 64, 64, 3)  # Block 4
        self.dec3 = DecoderBlock(64 + 32, 32, 3)  # Block 3
        self.dec2 = DecoderBlock(32 + 16, 16, 3)  # Block 2
        self.dec1 = DecoderBlock(16 + 1, config.out_channels, 3, final=True)  # Block 1

    def forward(self, x):
        """
        Forward pass of the U-Net.
        Input:  x [B, 1, F, T] (masked spectrogram)
        Output: x [B, 1, F, T] (reconstructed spectrogram)
        """
        # -------------------
        # Encode (downsample)
        # -------------------
        e1 = self.enc1(x)  # [B, 16, F/2,  T/2]
        e2 = self.enc2(e1)  # [B, 32, F/4,  T/4]
        e3 = self.enc3(e2)  # [B, 64, F/8,  T/8]
        e4 = self.enc4(e3)  # [B, 128, F/16, T/16]
        e5 = self.enc5(e4)  # [B, 128, F/32, T/32]
        e6 = self.enc6(e5)  # [B, 128, F/64, T/64]

        # -------------------
        # Decode (upsample)
        # -------------------
        d6 = self.dec6(e6, e5)  # [B, 128, F/32, T/32]
        d5 = self.dec5(d6, e4)  # [B, 128, F/16, T/16]
        d4 = self.dec4(d5, e3)  # [B,  64, F/8,  T/8]
        d3 = self.dec3(d4, e2)  # [B,  32, F/4,  T/4]
        d2 = self.dec2(d3, e1)  # [B,  16, F/2,  T/2]
        out = self.dec1(d2, x)  # [B,   1, F,    T]

        return out


class UNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super(UNet, self).__init__()
        self.config = config
        self.inc = inconv(self.config.in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512 , dropout=0)
        self.down4 = down(512, 512 , dropout=0)
        self.up1 = up(1024, 256, dropout=0)
        self.up2 = up(512, 128, dropout=0)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, self.config.out_channels)


        # x = 4
        # self.inc = inconv(self.config.in_channels, x)
        # self.down1 = down(x, 2*x)
        # self.down2 = down(2*x, 4*x)
        # self.down3 = down(4*x, 8*x , dropout=0)
        # self.down4 = down(8*x, 8*x , dropout=0)
        # self.up1 = up(16*x, 4*x, dropout=0)
        # self.up2 = up(8*x, 2*x, dropout=0)
        # self.up3 = up(4*x, x)
        # self.up4 = up(2*x, x)
        # self.outc = outconv(x, self.config.out_channels)





    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class RestorationWrapper(nn.Module):
    def __init__(self, base_net: UNet):
        super().__init__()
        self.net = base_net

    def forward(self, x_in: torch.Tensor, mask: torch.Tensor):
        # input dims of the mask are [B,1,F,T]
        # the dims of x change according to the in_channels config
        x = self.net(x_in)
        # Ensure mask is broadcastable to match x_in's shape [B, K, F, T]
        mask_broadcasted = mask
        if x.shape[1] > 1:  # If x_in has more than 1 channel (K > 1)
            mask_broadcasted = mask_broadcasted.expand(-1, x.shape[1], -1,-1)  # Broadcast along the channel dimension
        # Apply inpainting
        if x_in.shape[1] > 1:
            masked_spec = x_in[:,0,:,:]
            masked_spec = masked_spec.unsqueeze(1).expand(-1,mask_broadcasted.shape[1],-1,-1)
            x = masked_spec * mask_broadcasted + x * (1 - mask_broadcasted)
        else:
            x = x_in * mask_broadcasted + x * (1 - mask_broadcasted)
        return x
