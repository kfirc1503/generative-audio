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
    dropout: float = 0.0


class LatentEncoderConfig(UNetConfig):
    n_dirs: int


class LatentEncoderMultiLevelConfig(UNetConfig):
    """Config for multi-level latent encoder that outputs W for all skip levels"""
    n_dirs: int
    # Which levels to output W for (True = output W, False = skip)
    # Order: [skip1, skip2, skip3, skip4, bottleneck]
    # Default: all levels
    output_skip1: bool = True   # 64 channels
    output_skip2: bool = True   # 128 channels  
    output_skip3: bool = True   # 256 channels
    output_skip4: bool = True   # 512 channels
    output_bottleneck: bool = True  # 512 channels


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


# class UNet(nn.Module):
#     def __init__(self, config: UNetConfig):
#         super(UNet, self).__init__()
#         self.config = config
#         self.inc = inconv(self.config.in_channels, 64)
#         self.down1 = down(64, 128)
#         self.down2 = down(128, 256)
#         self.down3 = down(256, 512 , dropout=self.config.dropout)
#         self.down4 = down(512, 512 , dropout=self.config.dropout)
#         self.up1 = up(1024, 256, dropout=self.config.dropout)
#         self.up2 = up(512, 128, dropout=self.config.dropout)
#         self.up3 = up(256, 64)
#         self.up4 = up(128, 64)
#         self.outc = outconv(64, self.config.out_channels)
#
#
#         # x = 4
#         # self.inc = inconv(self.config.in_channels, x)
#         # self.down1 = down(x, 2*x)
#         # self.down2 = down(2*x, 4*x)
#         # self.down3 = down(4*x, 8*x , dropout=0)
#         # self.down4 = down(8*x, 8*x , dropout=0)
#         # self.up1 = up(16*x, 4*x, dropout=0)
#         # self.up2 = up(8*x, 2*x, dropout=0)
#         # self.up3 = up(4*x, x)
#         # self.up4 = up(2*x, x)
#         # self.outc = outconv(x, self.config.out_channels)
#
#
#
#
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         x = self.outc(x)
#         return x


class Encoder(nn.Module):
    def __init__(self, in_channels=1, dropout=0):
        super(Encoder, self).__init__()
        self.inc = inconv(in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512, dropout=dropout)
        self.down4 = down(512, 512, dropout=dropout)

    def forward(self, x):
        # Store intermediate outputs for skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x5, [x4, x3, x2, x1]


class Decoder(nn.Module):
    def __init__(self, out_channels=1, dropout=0):
        super(Decoder, self).__init__()
        self.up1 = up(1024, 256, dropout=dropout)  # 512 + 512 = 1024
        self.up2 = up(512, 128, dropout=dropout)  # 256 + 256 = 512
        self.up3 = up(256, 64)  # 128 + 128 = 256
        self.up4 = up(128, 64)  # 64 + 64 = 128
        self.outc = outconv(64, out_channels)

    def forward(self, x, skip_connections):
        x = self.up1(x, skip_connections[0])
        x = self.up2(x, skip_connections[1])
        x = self.up3(x, skip_connections[2])
        x = self.up4(x, skip_connections[3])
        x = self.outc(x)
        return x


class UNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super(UNet, self).__init__()
        self.config = config
        self.encoder = Encoder(self.config.in_channels, dropout=self.config.dropout)
        self.decoder = Decoder(self.config.out_channels, dropout=self.config.dropout)

    def forward(self, x):
        # Get encoder output and skip connections
        encoded, skip_connections = self.encoder(x)
        # Pass to decoder
        decoded = self.decoder(encoded, skip_connections)
        return decoded


# class RestorationWrapper(nn.Module):
#     def __init__(self, base_net: UNet):
#         super().__init__()
#         self.net = base_net
#
#     def forward(self, x_in: torch.Tensor, mask: torch.Tensor):
#         # input dims of the mask are [B,1,F,T]
#         # the dims of x change according to the in_channels config
#         x = self.net(x_in)
#         # Ensure mask is broadcastable to match x_in's shape [B, K, F, T]
#         mask_broadcasted = mask
#         if x.shape[1] > 1:  # If x_in has more than 1 channel (K > 1)
#             mask_broadcasted = mask_broadcasted.expand(-1, x.shape[1], -1, -1)  # Broadcast along the channel dimension
#         # Apply inpainting
#         if x_in.shape[1] > 1:
#             masked_spec = x_in[:, 0, :, :]
#             masked_spec = masked_spec.unsqueeze(1).expand(-1, mask_broadcasted.shape[1], -1, -1)
#             x = masked_spec * mask_broadcasted + x * (1 - mask_broadcasted)
#         else:
#             x = x_in * mask_broadcasted + x * (1 - mask_broadcasted)
#         return x


class RestorationWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()

        self.net = net
        # self.mask = mask

    def forward(self, x, mask):
        x_in = x

        # x = (x - 0.5) / 0.2
        x = self.net(x)
        # x = (x * 0.2) + 0.5

        x = x_in + x * mask
        return x







class LatentEncoder(nn.Module):
    def __init__(self, config:LatentEncoderConfig):
        super(LatentEncoder, self).__init__()

        self.n_dirs = config.n_dirs
        n_dirs = config.n_dirs
        dropout = config.dropout
        in_channels = config.in_channels


        # Keep exact same architecture as original UNet
        self.inc = inconv(in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512, dropout=dropout)
        # Modified last down layer to have n_dirs * 512 channels
        self.down4 = down(512, 512 * n_dirs, dropout=dropout)

    def forward(self, x,mask):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        mask2 = F.max_pool2d(mask, kernel_size=2)
        x3 = self.down2(x2)
        mask3 = F.max_pool2d(mask2, kernel_size=2)
        x4 = self.down3(x3)
        mask4 = F.max_pool2d(mask3, kernel_size=2)
        x5 = self.down4(x4)
        mask5 = F.max_pool2d(mask4, kernel_size=2)  # Final mask in latent space
        # Reshape to separate n_dirs dimension using C // n_dirs
        B, C, H, W = x5.shape
        x5 = x5.view(B, self.n_dirs, C // self.n_dirs, H, W)  # Shape: [B, n_dirs, C//n_dirs, H, W]

        return x5, mask5
        # return x5

# Example usage:
# encoder = LatentEncoder(in_channels=1, n_dirs=5)
# x = torch.randn(128, 1, 128, 256)
# latent = encoder(x)
# print(latent.shape)  # Should be [128, 5, 512, 8, 16]


class LatentEncoderMultiLevel(nn.Module):
    """
    Multi-level latent encoder that outputs W directions for:
    - Bottleneck (512 channels)
    - Skip connection 4 (512 channels)
    - Skip connection 3 (256 channels)
    - Skip connection 2 (128 channels)
    - Skip connection 1 (64 channels)
    
    This allows modifying skip connections during inference,
    which the decoder actually uses (unlike bottleneck-only approach).
    """
    
    def __init__(self, config: LatentEncoderMultiLevelConfig):
        super(LatentEncoderMultiLevel, self).__init__()
        
        self.n_dirs = config.n_dirs
        self.config = config
        n_dirs = config.n_dirs
        dropout = config.dropout
        in_channels = config.in_channels
        
        # Shared encoder backbone (same as original LatentEncoder)
        self.inc = inconv(in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512, dropout=dropout)
        self.down4 = down(512, 512, dropout=dropout)
        
        # Output heads for each level
        # Each head takes the features at that level and outputs n_dirs directions
        
        # Bottleneck head: 512 -> 512 * n_dirs
        if config.output_bottleneck:
            self.head_bottleneck = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512 * n_dirs, kernel_size=1)
            )
        
        # Skip4 head: 512 -> 512 * n_dirs (same resolution as x4)
        if config.output_skip4:
            self.head_skip4 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512 * n_dirs, kernel_size=1)
            )
        
        # Skip3 head: 256 -> 256 * n_dirs
        if config.output_skip3:
            self.head_skip3 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256 * n_dirs, kernel_size=1)
            )
        
        # Skip2 head: 128 -> 128 * n_dirs
        if config.output_skip2:
            self.head_skip2 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128 * n_dirs, kernel_size=1)
            )
        
        # Skip1 head: 64 -> 64 * n_dirs
        if config.output_skip1:
            self.head_skip1 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64 * n_dirs, kernel_size=1)
            )
    
    def forward(self, x, mask):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, in_channels, F, T]
            mask: Mask tensor [B, 1, F, T] (1 = known, 0 = missing)
            
        Returns:
            w_bottleneck: [B, n_dirs, 512, H_b, W_b] or None
            w_skips: List of [w_skip4, w_skip3, w_skip2, w_skip1] or None for each
            masks: Dict of masks at each level
        """
        n_dirs = self.n_dirs
        
        # Encode through backbone
        # Note: inc (inconv) does NOT downsample, down1-4 DO downsample
        # Mask pooling must match the encoder's downsampling pattern
        x1 = self.inc(x)      # [B, 64, F, T] - same spatial size as input
        # mask1 should match x1's spatial size (no pooling yet)
        
        x2 = self.down1(x1)   # [B, 128, F/2, T/2]
        mask2 = F.max_pool2d(mask, kernel_size=2)  # Pool from original mask
        
        x3 = self.down2(x2)   # [B, 256, F/4, T/4]
        mask3 = F.max_pool2d(mask2, kernel_size=2)
        
        x4 = self.down3(x3)   # [B, 512, F/8, T/8]
        mask4 = F.max_pool2d(mask3, kernel_size=2)
        
        x5 = self.down4(x4)   # [B, 512, F/16, T/16] - bottleneck
        mask5 = F.max_pool2d(mask4, kernel_size=2)
        
        # Generate W directions at each level
        B = x.shape[0]
        
        # Bottleneck W
        w_bottleneck = None
        if self.config.output_bottleneck:
            w_b = self.head_bottleneck(x5)  # [B, 512*n_dirs, H, W]
            _, C_total, H, W = w_b.shape
            C = C_total // n_dirs
            w_bottleneck = w_b.view(B, n_dirs, C, H, W)
        
        # Skip4 W (uses x4 features)
        w_skip4 = None
        if self.config.output_skip4:
            w_s4 = self.head_skip4(x4)
            _, C_total, H, W = w_s4.shape
            C = C_total // n_dirs
            w_skip4 = w_s4.view(B, n_dirs, C, H, W)
        
        # Skip3 W (uses x3 features)
        w_skip3 = None
        if self.config.output_skip3:
            w_s3 = self.head_skip3(x3)
            _, C_total, H, W = w_s3.shape
            C = C_total // n_dirs
            w_skip3 = w_s3.view(B, n_dirs, C, H, W)
        
        # Skip2 W (uses x2 features)
        w_skip2 = None
        if self.config.output_skip2:
            w_s2 = self.head_skip2(x2)
            _, C_total, H, W = w_s2.shape
            C = C_total // n_dirs
            w_skip2 = w_s2.view(B, n_dirs, C, H, W)
        
        # Skip1 W (uses x1 features)
        w_skip1 = None
        if self.config.output_skip1:
            w_s1 = self.head_skip1(x1)
            _, C_total, H, W = w_s1.shape
            C = C_total // n_dirs
            w_skip1 = w_s1.view(B, n_dirs, C, H, W)
        
        # Pack outputs
        w_skips = [w_skip4, w_skip3, w_skip2, w_skip1]  # Order matches decoder's skip_connections
        masks = {
            'bottleneck': mask5,
            'skip4': mask4,
            'skip3': mask3,
            'skip2': mask2,
            'skip1': mask  # x1 has same spatial size as input, so use original mask
        }
        
        return w_bottleneck, w_skips, masks


# Example usage for LatentEncoderMultiLevel:
# config = LatentEncoderMultiLevelConfig(in_channels=2, n_dirs=5)
# encoder = LatentEncoderMultiLevel(config)
# x = torch.randn(4, 2, 128, 256)
# mask = torch.ones(4, 1, 128, 256)
# w_bottleneck, w_skips, masks = encoder(x, mask)
# print(f"w_bottleneck: {w_bottleneck.shape}")  # [4, 5, 512, 8, 16]
# print(f"w_skip4: {w_skips[0].shape}")  # [4, 5, 512, 16, 32]
# print(f"w_skip3: {w_skips[1].shape}")  # [4, 5, 256, 32, 64]
# print(f"w_skip2: {w_skips[2].shape}")  # [4, 5, 128, 64, 128]
# print(f"w_skip1: {w_skips[3].shape}")  # [4, 5, 64, 128, 256]
