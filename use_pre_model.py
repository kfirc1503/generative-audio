from pydantic import BaseModel
import FullSubNet_plus.speech_enhance.fullsubnet_plus.model.fullsubnet_plus as fullsubnet_plus
from pathlib import Path
import torch
import tarfile
import io
import utils
from FullSubNet_plus.speech_enhance.fullsubnet_plus.model.fullsubnet_plus import FullSubNet_Plus


class FullSubNetPlusConfig(BaseModel):
    num_freqs = 257
    look_ahead = 2
    sequence_model = "LSTM"
    sb_num_neighbors = 15
    fb_num_neighbors = 0
    fb_output_activate_function = "ReLU"
    sb_output_activate_function = False
    fb_model_hidden_size = 512
    sb_model_hidden_size = 384
    channel_attention_model = "TSSE"
    norm_type = "offline_laplace_norm"
    num_groups_in_drop_band = 2
    output_size = 2
    subband_num = 1
    kersize = [3, 5, 10]
    weight_init = False


tmp_config = FullSubNetPlusConfig()


tar_path = Path("./FullSubNet_plus/best_model.tar")
pre_train_model = fullsubnet_plus.FullSubNet_Plus(**tmp_config.dict())
pre_train_model = utils.preload_model(tar_path , pre_train_model)


pre_train_model.eval()  # Set model to evaluation mode
with torch.no_grad():
    # Prepare input
    path_to_audio = "./FullSubNet_plus/data/noisy/_1eSheWjfJQ.wav"
    noisy_mag, noisy_real, noisy_imag = utils.prepare_input(path_to_audio)

    # Forward pass through model
    enhanced = pre_train_model(noisy_mag,noisy_real, noisy_imag)
    # Apply mask to get enhanced real and imaginary components
    enhanced_real = enhanced[:, 0] * noisy_real.squeeze(1)  # Remove channel dim from noisy_real
    enhanced_imag = enhanced[:, 1] * noisy_imag.squeeze(1)  # Remove channel dim from noisy_imag

    # Reconstruct complex STFT
    enhanced_complex = torch.complex(enhanced_real, enhanced_imag)

    enhanced_waveform = torch.istft(
        enhanced_complex,
        n_fft=512,  # Should match the FFT size used in prepare_input
        hop_length=256,  # Should match the hop length used in prepare_input
        win_length=512,  # Should match the window length used in prepare_input
        window=torch.hann_window(512).to(enhanced_complex.device),
        center=True,
        return_complex=False
    )


    kaka = 1
    # Convert back to audio if needed
    # ... additional processing to convert enhanced magnitude back to waveform





print(pre_train_model)
