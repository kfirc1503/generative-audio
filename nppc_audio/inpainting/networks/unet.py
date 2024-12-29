# import torch
# import torch.nn as nn
# import pydantic
# from typing import Tuple
# import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import pydantic
from typing import Tuple
from utils import normalize_spectrograms,denormalize_spectrograms
from nppc_audio.inpainting.networks.tmp_utils import *

# class UNetConfig(pydantic.BaseModel):
#     """
#     Matches your style of config.
#     Note: We won't use `min_channels_decoder` here,
#     since we're using a 'classic' approach with
#     symmetrical up/down channels. If you like,
#     you can adapt the logic to incorporate that.
#     """
#     in_channels: int = 1
#     out_channels: int = 1
#     channels_list: Tuple[int, ...] = (32, 64, 128, 256)
#     bottleneck_channels: int = 512
#     n_groups: int = 8  # For GroupNorm
# #
# #
# def double_conv(
#     in_ch: int,
#     out_ch: int,
#     n_groups: int,
#     negative_slope: float = 0.1
# ) -> nn.Sequential:
#     """
#     A standard double-conv block:
#       Conv2d -> GroupNorm -> LeakyReLU -> Conv2d -> GroupNorm -> LeakyReLU
#     """
#     return nn.Sequential(
#         nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
#         nn.GroupNorm(n_groups, out_ch),
#         nn.LeakyReLU(negative_slope, inplace=True),
#
#         nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#         nn.GroupNorm(n_groups, out_ch),
#         nn.LeakyReLU(negative_slope, inplace=True),
#     )
#
#
# class UNet(nn.Module):
#     """
#     A "classic" 2D UNet with double-conv blocks at each stage,
#     using transposed convolution for upsampling.
#
#     1) Encoder:
#        - For each entry in `config.channels_list`, do double_conv,
#          then downsample via MaxPool2d(2).
#     2) Bottleneck:
#        - Another double_conv from the last channel to `config.bottleneck_channels`.
#     3) Decoder:
#        - Upsample via ConvTranspose2d, concatenate skip, double_conv.
#     4) Final 1x1 conv to config.out_channels.
#     """
#
#     def __init__(self, config: UNetConfig):
#         super().__init__()
#         self.config = config
#
#         # ---------------------
#         # 1) Encoder
#         # ---------------------
#         self.down_blocks = nn.ModuleList()
#         self.pool = nn.MaxPool2d(2)
#
#         in_ch = config.in_channels
#         # We will store the output channels of each down block in skip_channels
#         self.skip_channels = []
#
#         for out_ch in config.channels_list:
#             block = double_conv(in_ch, out_ch, config.n_groups)
#             self.down_blocks.append(block)
#             self.skip_channels.append(out_ch)
#             in_ch = out_ch  # next block's input
#         # Note: after the last down_block, we'll do a pool before the bottleneck.
#
#         # ---------------------
#         # 2) Bottleneck
#         # ---------------------
#         self.bottleneck = double_conv(
#             in_ch, config.bottleneck_channels, config.n_groups
#         )
#         # The output of the bottleneck has config.bottleneck_channels channels
#
#         # ---------------------
#         # 3) Decoder
#         # ---------------------
#         self.up_blocks = nn.ModuleList()
#
#         # reversed_channels: reverse the encoder channels_list
#         # so we can upsample from bottleneck_channels -> channels_list[-1],
#         # then channels_list[-2], etc.
#         in_ch = config.bottleneck_channels
#         reversed_channels = list(reversed(config.channels_list))
#
#         for out_ch in reversed_channels:
#             # 3a) Upsample
#             self.up_blocks.append(
#                 nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
#             )
#
#             # 3b) Double Conv (we'll concat skip connection, so input = out_ch + skip.size(1))
#             # So final input to double_conv = out_ch + skip_channels_from_encoder
#             # We won't define this up-block as a single nn.Sequential because we need
#             # to do the concat in forward(). We'll store each double_conv separately.
#             # The next line's out_ch must match the final channels after double_conv
#             self.up_blocks.append(
#                 double_conv(out_ch + out_ch, out_ch, config.n_groups)
#             )
#
#             in_ch = out_ch
#
#         # ---------------------
#         # 4) Final Output Conv
#         # ---------------------
#         self.final_conv = nn.Conv2d(in_ch, config.out_channels, kernel_size=1)
#
#     def forward(self, x):
#         # x shape: [B, in_channels, H, W]
#
#         # ---------- Encoder ----------
#         skip_feats = []
#         out = x
#         for block in self.down_blocks:
#             out = block(out)     # double conv
#             skip_feats.append(out)
#             out = self.pool(out) # downsample
#
#         # ---------- Bottleneck ----------
#         out = self.bottleneck(out)  # [B, bottleneck_channels, H/2^depth, W/2^depth]
#
#         # ---------- Decoder ----------
#         # skip_feats in forward order: skip_feats[0] -> channels_list[0], ...
#         # but we need them in reverse for upsampling, so skip_feats[-1] is the deepest.
#         # However, notice we used pool *after* each block, so the skip_feats align with
#         # the dimension before pooling.
#         # We'll pop from skip_feats in reverse.
#         for i in range(0, len(self.up_blocks), 2):
#             # i: even -> upsample, i+1: double_conv
#             upsample_layer = self.up_blocks[i]
#             double_conv_layer = self.up_blocks[i + 1]
#
#             # Upsample
#             out = upsample_layer(out)
#
#             # Skip connection
#             skip = skip_feats.pop()  # last element
#             # Concat
#             # out.shape might differ from skip if input dims weren't multiples of 2^N
#             if out.size(2) != skip.size(2) or out.size(3) != skip.size(3):
#                 # We can do a simple center-crop of skip or an interpolate of out
#                 diffH = skip.size(2) - out.size(2)
#                 diffW = skip.size(3) - out.size(3)
#                 if diffH > 0 or diffW > 0:
#                     skip = skip[:, :, diffH // 2: skip.size(2) - (diffH - diffH // 2),
#                                 diffW // 2: skip.size(3) - (diffW - diffW // 2)]
#
#                 elif diffH < 0 or diffW < 0:
#                     # out is bigger - let's just interpolate out to skip's size
#                     out = F.interpolate(out, size=(skip.size(2), skip.size(3)), mode='nearest')
#
#             out = torch.cat([skip, out], dim=1)
#
#             # Double conv
#             out = double_conv_layer(out)
#
#         # ---------- Final Conv ----------
#         out = self.final_conv(out)
#         return out



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


class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()
        self.inc = inconv(2, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, 2)

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

#
# class UNetConfig(pydantic.BaseModel):
#     """
#     Configuration for the U-Net architecture.
#
#     Attributes:
#         input_channels (int): Number of input channels.
#         output_channels (int): Number of output channels.
#         base_channels (int): Number of channels in the first convolutional layer.
#     """
#     input_channels: int = 1
#     output_channels: int = 1
#     base_channels: int = 64
#
#
# class UNet(nn.Module):
#     def __init__(self, config: UNetConfig):
#         """
#         Implements the U-Net architecture for audio spectrogram inpainting.
#
#         Args:
#             config (UNetConfig): Configuration object for the U-Net.
#         """
#         super(UNet, self).__init__()
#
#         # Encoder layers
#         self.enc1 = self.conv_block(config.input_channels, config.base_channels)
#         self.enc2 = self.conv_block(config.base_channels, config.base_channels * 2)
#         self.enc3 = self.conv_block(config.base_channels * 2, config.base_channels * 4)
#         self.enc4 = self.conv_block(config.base_channels * 4, config.base_channels * 8)
#
#         # Bottleneck
#         self.bottleneck = self.conv_block(config.base_channels * 8, config.base_channels * 16)
#
#         # Decoder layers
#         self.up4 = self.upsample(config.base_channels * 16, config.base_channels * 8)
#         self.dec4 = self.conv_block(config.base_channels * 16, config.base_channels * 8)
#
#         self.up3 = self.upsample(config.base_channels * 8, config.base_channels * 4)
#         self.dec3 = self.conv_block(config.base_channels * 8, config.base_channels * 4)
#
#         self.up2 = self.upsample(config.base_channels * 4, config.base_channels * 2)
#         self.dec2 = self.conv_block(config.base_channels * 4, config.base_channels * 2)
#
#         self.up1 = self.upsample(config.base_channels * 2, config.base_channels)
#         self.dec1 = self.conv_block(config.base_channels * 2, config.base_channels)
#
#         # Final output layer
#         self.final = nn.Conv2d(config.base_channels, config.output_channels, kernel_size=1)
#
#     def conv_block(self, in_channels, out_channels):
#         """
#         Convolutional block with two Conv2D layers followed by ReLU activations.
#
#         Args:
#             in_channels (int): Number of input channels.
#             out_channels (int): Number of output channels.
#
#         Returns:
#             nn.Sequential: The convolutional block.
#         """
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#
#     def upsample(self, in_channels, out_channels):
#         """
#         Upsampling block using transposed convolution.
#
#         Args:
#             in_channels (int): Number of input channels.
#             out_channels (int): Number of output channels.
#
#         Returns:
#             nn.ConvTranspose2d: The upsampling layer.
#         """
#         return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#
#     def forward(self, x):
#         """
#         Forward pass through the U-Net.
#
#         Args:
#             x (torch.Tensor): Input tensor of shape (batch, channels, height, width).
#
#         Returns:
#             torch.Tensor: Output tensor of shape (batch, output_channels, height, width).
#         """
#         # Encoder path
#         e1 = self.enc1(x)
#         e2 = self.enc2(F.max_pool2d(e1, kernel_size=2))
#         e3 = self.enc3(F.max_pool2d(e2, kernel_size=2))
#         e4 = self.enc4(F.max_pool2d(e3, kernel_size=2))
#
#         # Bottleneck
#         b = self.bottleneck(F.max_pool2d(e4, kernel_size=2))
#
#         # Decoder path
#         d4 = self.up4(b)
#         d4 = self.dec4(torch.cat([d4, e4], dim=1))
#
#         d3 = self.up3(d4)
#         d3 = self.dec3(torch.cat([d3, e3], dim=1))
#
#         d2 = self.up2(d3)
#         d2 = self.dec2(torch.cat([d2, e2], dim=1))
#
#         d1 = self.up1(d2)
#         d1 = self.dec1(torch.cat([d1, e1], dim=1))
#
#         # Final output
#         return self.final(d1)
#


    ###############################################################################
    # Configuration
    ###############################################################################
#
# class UNetConfig(pydantic.BaseModel):
#     in_channels: int = 1  # e.g. single-channel magnitude input
#     out_channels: int = 1  # e.g. single-channel magnitude output
#     base_filters: int = 64  # number of filters in the first downsampling block
#     num_layers: int = 4  # how many down/up blocks (excluding final output conv)
#     leaky_slope: float = 0.2  # slope for LeakyReLU
#     dropout: float = 0.5  # dropout probability applied at deeper layers
#
# ###############################################################################
# # Downsampling and Upsampling blocks
# ###############################################################################
#
# def down_block(in_ch, out_ch, slope, dropout=0.0):
#     """Downsampling block: Conv -> InstanceNorm -> LeakyReLU -> (optional) Dropout."""
#     layers = [
#         nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
#         nn.InstanceNorm2d(out_ch, affine=False),
#         nn.LeakyReLU(slope, inplace=True)
#     ]
#     if dropout > 0:
#         layers.append(nn.Dropout(dropout))
#     return nn.Sequential(*layers)
#
# def up_block(in_ch, out_ch, dropout=0.0):
#     """Upsampling block: TransposeConv -> InstanceNorm -> ReLU -> (optional) Dropout."""
#     layers = [
#         nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
#         nn.InstanceNorm2d(out_ch, affine=False),
#         nn.ReLU(inplace=True)
#     ]
#     if dropout > 0:
#         layers.append(nn.Dropout(dropout))
#     return nn.Sequential(*layers)
#
# ###############################################################################
# # U-Net Generator
# ###############################################################################
#
# class UNet(nn.Module):
#     """
#     A simplified U-Net for audio inpainting on magnitude spectrograms.
#     """
#
#     def __init__(self, config: UNetConfig):
#         super().__init__()
#         self.config = config
#
#         # Build downsampling layers
#         self.downs = nn.ModuleList()
#         filters = config.base_filters
#
#         # First down block (might skip InstanceNorm in some references, but we keep it)
#         self.downs.append(down_block(config.in_channels, filters, slope=config.leaky_slope, dropout=0.0))
#
#         # Further down blocks
#         in_ch = filters
#         for i in range(1, config.num_layers):
#             out_ch = min(in_ch * 2, 512)  # typical pix2pix-capped at 512
#             # Apply dropout only in deeper layers
#             do = config.dropout if i >= config.num_layers - 1 else 0.0
#             self.downs.append(down_block(in_ch, out_ch, slope=config.leaky_slope, dropout=do))
#             in_ch = out_ch
#
#         # Build upsampling layers
#         self.ups = nn.ModuleList()
#         for i in range(config.num_layers - 1):
#             # Reverse logic: up block gets skip connection -> sum of filters
#             out_ch = in_ch // 2
#             do = config.dropout if i < 1 else 0.0  # optional logic for dropout in up layers
#             self.ups.append(up_block(in_ch * 2, out_ch, dropout=do))
#             in_ch = out_ch
#
#         # Final up block
#         # in_ch is the filters in the last up layer, but with skip connection => in_ch*2
#         self.last_up = nn.ConvTranspose2d(in_ch * 2, config.out_channels,
#                                           kernel_size=4, stride=2, padding=1)
#
#         # Output activation (Sigmoid or Tanh, depending on your use-case)
#         self.activation = nn.Sigmoid()
#
#     def forward(self, x):
#         # Down pass
#         skips = []
#         d_out = x
#         for down in self.downs:
#             d_out = down(d_out)
#             skips.append(d_out)
#
#         # Bottleneck is d_out
#         u_out = d_out
#
#         # Up pass
#         # use all but the last skip (the last skip is the deepest)
#         for up in self.ups:
#             skip = skips.pop()  # skip from the previous layer
#             u_out = torch.cat([u_out, skip], dim=1)
#             u_out = up(u_out)
#
#         # Last skip is the first down block
#         first_skip = skips.pop()
#         u_out = torch.cat([u_out, first_skip], dim=1)
#         u_out = self.last_up(u_out)
#         return self.activation(u_out)


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
class UNet(nn.Module):
    """
    6 encoder blocks, 6 decoder blocks.
    The final output is a 2D spectrogram [B, 1, F, T].
    """

    def __init__(self):
        super().__init__()

        # Encoder (blue blocks)
        self.enc1 = EncoderBlock(1, 16, 7)  # Block 1: (7, 16)
        self.enc2 = EncoderBlock(16, 32, 5)  # Block 2: (5, 32)
        self.enc3 = EncoderBlock(32, 64, 5)  # Block 3: (5, 64)
        self.enc4 = EncoderBlock(64, 128, 3)  # Block 4: (3, 128)
        self.enc5 = EncoderBlock(128, 128, 3)  # Block 5: (3, 128)
        self.enc6 = EncoderBlock(128, 128, 3)  # Block 6: (3, 128)

        # Decoder (green blocks)
        self.dec6 = DecoderBlock(128 + 128, 128, 3)  # Block 6
        self.dec5 = DecoderBlock(128 + 128, 128, 3)  # Block 5
        self.dec4 = DecoderBlock(128 + 64, 64, 3)  # Block 4
        self.dec3 = DecoderBlock(64 + 32, 32, 3)  # Block 3
        self.dec2 = DecoderBlock(32 + 16, 16, 3)  # Block 2
        self.dec1 = DecoderBlock(16 + 1, 1, 3, final=True)  # Block 1

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

class RestorationWrapper(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config
        self.net = UNet()

    def forward(self, x_in: torch.Tensor, mask: torch.Tensor):
        # x shape: (B, 2, F, T) [real, imag]
        # Normalize both real and imag
        #normalized the masked spectogram,
        # x_in_norm, mean, std = normalize_spectrograms(x_in)

        x_norm = self.net(x_in)
        x = x_norm
        #Denormalized
        # x = denormalize_spectrograms(x_norm, mean, std)
        # Apply inpainting
        # x = x_in * (1 - mask) + x * mask
        x = x_in * mask + x * (1 - mask)
#         x = x_in[:,:-1,:,:] * mask + x * (1 - mask)


        return x


