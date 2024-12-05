import torch
from typing import Optional
from torch.nn import functional
from FullSubNet_plus.speech_enhance.fullsubnet_plus.model.fullsubnet_plus import FullSubNet_Plus, FullSubNetPlusConfig
import pydantic
import torch
from typing import Optional, List
from torch.nn import functional
from omegaconf import ListConfig
from FullSubNet_plus.speech_enhance.audio_zen.acoustics.feature import drop_band
from FullSubNet_plus.speech_enhance.audio_zen.model.base_model import BaseModel
from FullSubNet_plus.speech_enhance.audio_zen.model.module.sequence_model import SequenceModel
from FullSubNet_plus.speech_enhance.audio_zen.model.module.attention_model import ChannelSELayer, ChannelECAlayer, \
    ChannelCBAMLayer, \
    ChannelTimeSenseSELayer


class MultiDirectionConfig(FullSubNetPlusConfig):
    n_directions: int = 4  # Number of output CRMs for uncertainty


class MultiDirectionFullSubNet_Plus(FullSubNet_Plus):
    def __init__(self, config: Optional[MultiDirectionConfig] = None):
        if config is None:
            config = MultiDirectionConfig()

        # Modify config for multiple outputs
        config.output_size = 2 * config.n_directions  # 2 for real and imaginary parts

        # Initialize parent class
        super().__init__(config)

        # Store number of directions
        self.n_directions = config.n_directions

        # Modify fullband models for concatenated input
        input_size = self.num_freqs * 2  # Double for enhanced input
        self.fb_model = SequenceModel(
            input_size=input_size,
            output_size=self.num_freqs,
            hidden_size=self.fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model="TCN",
            output_activate_function=self.fb_output_activate_function
        )

        self.fb_model_real = SequenceModel(
            input_size=input_size,
            output_size=self.num_freqs,
            hidden_size=self.fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model="TCN",
            output_activate_function=self.fb_output_activate_function
        )

        self.fb_model_imag = SequenceModel(
            input_size=input_size,
            output_size=self.num_freqs,
            hidden_size=self.fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model="TCN",
            output_activate_function=self.fb_output_activate_function
        )

    def forward(self, noisy_mag, noisy_real, noisy_imag,
                enhanced_mag=None, enhanced_real=None, enhanced_imag=None):
        """
        Forward pass with enhanced input and multiple CRM outputs.

        Args:
            noisy_mag: noisy magnitude spectrogram [B, 1, F, T]
            noisy_real: noisy real part [B, 1, F, T]
            noisy_imag: noisy imaginary part [B, 1, F, T]
            enhanced_mag: enhanced magnitude from previous model [B, 1, F, T]
            enhanced_real: enhanced real part from previous model [B, 1, F, T]
            enhanced_imag: enhanced imaginary part from previous model [B, 1, F, T]

        Returns:
            [B, 2*n_directions, F, T] (multiple sets of real and imaginary masks)
        """
        # Apply padding
        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead])
        noisy_real = functional.pad(noisy_real, [0, self.look_ahead])
        noisy_imag = functional.pad(noisy_imag, [0, self.look_ahead])
        enhanced_mag = functional.pad(enhanced_mag, [0, self.look_ahead])
        enhanced_real = functional.pad(enhanced_real, [0, self.look_ahead])
        enhanced_imag = functional.pad(enhanced_imag, [0, self.look_ahead])

        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()

        # Process both inputs through attention
        fb_input_noisy = self.norm(noisy_mag).reshape(batch_size, num_channels * num_freqs, num_frames)
        fb_input_enhanced = self.norm(enhanced_mag).reshape(batch_size, num_channels * num_freqs, num_frames)

        fb_input_noisy = self.channel_attention(fb_input_noisy)
        fb_input_enhanced = self.channel_attention(fb_input_enhanced)

        # Same for real and imaginary parts
        fbr_input_noisy = self.norm(noisy_real).reshape(batch_size, num_channels * num_freqs, num_frames)
        fbr_input_enhanced = self.norm(enhanced_real).reshape(batch_size, num_channels * num_freqs, num_frames)

        fbr_input_noisy = self.channel_attention_real(fbr_input_noisy)
        fbr_input_enhanced = self.channel_attention_real(fbr_input_enhanced)

        fbi_input_noisy = self.norm(noisy_imag).reshape(batch_size, num_channels * num_freqs, num_frames)
        fbi_input_enhanced = self.norm(enhanced_imag).reshape(batch_size, num_channels * num_freqs, num_frames)

        fbi_input_noisy = self.channel_attention_imag(fbi_input_noisy)
        fbi_input_enhanced = self.channel_attention_imag(fbi_input_enhanced)

        # Concatenate noisy and enhanced inputs
        fb_input = torch.cat([fb_input_noisy, fb_input_enhanced], dim=1)
        fbr_input = torch.cat([fbr_input_noisy, fbr_input_enhanced], dim=1)
        fbi_input = torch.cat([fbi_input_noisy, fbi_input_enhanced], dim=1)

        # Process through fullband models
        fb_output = self.fb_model(fb_input).reshape(batch_size, 1, num_freqs, num_frames)
        fbr_output = self.fb_model_real(fbr_input).reshape(batch_size, 1, num_freqs, num_frames)
        fbi_output = self.fb_model_imag(fbi_input).reshape(batch_size, 1, num_freqs, num_frames)

        # Unfold outputs
        fb_output_unfolded = self.unfold(fb_output, num_neighbor=self.fb_num_neighbors)
        fb_output_unfolded = fb_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1,
                                                        num_frames)

        fbr_output_unfolded = self.unfold(fbr_output, num_neighbor=self.fb_num_neighbors)
        fbr_output_unfolded = fbr_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1,
                                                          num_frames)

        fbi_output_unfolded = self.unfold(fbi_output, num_neighbor=self.fb_num_neighbors)
        fbi_output_unfolded = fbi_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1,
                                                          num_frames)

        # Process subband
        noisy_mag_unfolded = self.unfold(noisy_mag, num_neighbor=self.sb_num_neighbors)
        noisy_mag_unfolded = noisy_mag_unfolded.reshape(batch_size, num_freqs, self.sb_num_neighbors * 2 + 1,
                                                        num_frames)

        # Concatenate all features
        sb_input = torch.cat([noisy_mag_unfolded, fb_output_unfolded, fbr_output_unfolded, fbi_output_unfolded], dim=2)
        sb_input = self.norm(sb_input)

        # Apply drop band if needed
        if batch_size > 1:
            sb_input = drop_band(sb_input.permute(0, 2, 1, 3), num_groups=self.num_groups_in_drop_band)
            num_freqs = sb_input.shape[2]
            sb_input = sb_input.permute(0, 2, 1, 3)

        # Process through subband model
        sb_input = sb_input.reshape(
            batch_size * num_freqs,
            (self.sb_num_neighbors * 2 + 1) + 3 * (self.fb_num_neighbors * 2 + 1),
            num_frames
        )

        # Get multiple CRM outputs
        sb_masks = self.sb_model(sb_input)
        sb_masks = sb_masks.reshape(batch_size, num_freqs, self.n_directions, 2, num_frames)
        sb_masks = sb_masks.permute(0, 2, 3, 1, 4)  # [B, n_directions, 2, F, T]

        # Remove look-ahead padding and reshape
        output = sb_masks[..., self.look_ahead:]
        output = output.reshape(batch_size, 2 * self.n_directions, num_freqs, -1)

        return output
