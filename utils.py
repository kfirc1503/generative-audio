# Import necessary libraries for audio processing
import torchaudio
import torch
from typing import Union,Tuple
from pathlib import Path


from FullSubNet_plus.speech_enhance.fullsubnet_plus.model.fullsubnet_plus import FullSubNet_Plus, FullSubNetPlusConfig
import pydantic
from FullSubNet_plus.speech_enhance.audio_zen.acoustics.mask import decompress_cIRM


class StftConfig(pydantic.BaseModel):
    nfft: int = 512
    hop_length: int = 256
    win_length: int = 512


class AudioConfig(pydantic.BaseModel):
    sr: int = 16000
    stft_configuration: StftConfig


def model_outputs_to_waveforms(enhanced_masks, noisy_reals, noisy_imags, orig_length):
    """
    Convert model outputs back to waveforms.

    Args:
        enhanced_masks: Complex mask from model with shape [B, 2, F, T]
        noisy_reals: Real part of noisy STFT [B, 1, F, T]
        noisy_imags: Imaginary part of noisy STFT [B, 1, F, T]
    """
    # Permute mask to [B, F, T, 2]
    enhanced_masks = enhanced_masks.permute(0, 2, 3, 1)

    # Decompress the complex ideal ratio mask
    enhanced_masks = decompress_cIRM(enhanced_masks)
    noisy_reals = noisy_reals.squeeze(1)
    noisy_imags = noisy_imags.squeeze(1)
    # Remove channel dimension from noisy components
    enhanced_imags, enhanced_reals = noisy_to_enhanced(enhanced_masks, noisy_imags, noisy_reals)

    # Create complex tensor
    enhanced_complex = torch.complex(enhanced_reals, enhanced_imags)

    # Convert back to waveform using inverse STFT
    enhanced_waveforms = torch.istft(
        enhanced_complex,
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=torch.hann_window(512).to(enhanced_complex.device),
        center=True,
        return_complex=False,
        length=orig_length  # Specify the original length

    )

    return enhanced_waveforms


def noisy_to_enhanced(enhanced_masks, noisy_imags, noisy_reals):
    # Apply mask using the complex multiplication formula
    enhanced_reals = enhanced_masks[..., 0] * noisy_reals - enhanced_masks[..., 1] * noisy_imags
    enhanced_imags = enhanced_masks[..., 1] * noisy_reals + enhanced_masks[..., 0] * noisy_imags
    return enhanced_imags, enhanced_reals


def preload_model(model_path: Union[Path, str], model: FullSubNet_Plus) -> FullSubNet_Plus:
    """
    Preload model parameters (in "*.tar" format) at the start of experiment.

    Args:
        model_path (Path): The file path of the *.tar file
    """
    if type(model_path) == str:
        # turn him to path
        model_path = Path(model_path)
        # absolute_path = model_path.resolve()
    model_path = model_path.expanduser().absolute()
    assert model_path.exists(), f"The file {model_path.as_posix()} is not exist. please check path."

    model_checkpoint = torch.load(model_path.as_posix(), map_location="cpu")
    model.load_state_dict(model_checkpoint["model"], strict=False)
    return model


def load_pretrained_model(model_path: Union[Path, str], model_config: FullSubNetPlusConfig) -> FullSubNet_Plus:
    model = FullSubNet_Plus(model_config)
    model = preload_model(model_path, model)
    return model


def prepare_input_from_waveform(waveform, n_fft: int, hop_length: int, win_length: int, device: torch.device) -> tuple:
    """
    Prepare input for FullSubNet_Plus model from a waveform tensor.

    Args:
        waveform (torch.Tensor): Audio waveform tensor with shape [1, T] or [B, T]

    Returns:
        tuple: (noisy_mag, noisy_real, noisy_imag) tensors with shape [B, 1, F, T]
               where F=257 frequency bins and T=time steps
    """
    # Ensure input has batch dimension
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() == 2 and waveform.size(0) > 1:
        pass
        # waveform = waveform.unsqueeze(1)  # Add channel dimension
    window = torch.hann_window(win_length).to(device)

    # Calculate STFT (matching the parameters from the paper)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,  # Results in 257 frequency bins
        hop_length=hop_length,  # 50% overlap
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True
    )

    # Get components needed by the model
    noisy_real = stft.real
    noisy_imag = stft.imag
    noisy_mag = torch.sqrt(noisy_real ** 2 + noisy_imag ** 2)

    # Add channel dimension if needed [B, 1, F, T]
    noisy_mag = noisy_mag.unsqueeze(1)
    noisy_real = noisy_real.unsqueeze(1)
    noisy_imag = noisy_imag.unsqueeze(1)

    return noisy_mag, noisy_real, noisy_imag


def prepare_input(audio_path: str | Path):
    """
    Prepare audio input for FullSubNet_Plus model according to the official implementation

    Args:
        audio_path (str | Path): Relative or absolute path to audio file
    Returns:
        tuple: (noisy_mag, noisy_real, noisy_imag) tensors with shape [B, 1, F, T]
               where F=257 frequency bins and T=time steps
    """
    audio_path = Path(audio_path).resolve()

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load and convert to mono if needed
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Calculate STFT (matching the parameters from the paper)
    stft = torch.stft(
        waveform,
        n_fft=512,  # Results in 257 frequency bins
        hop_length=256,  # 50% overlap
        win_length=512,
        window=torch.hann_window(512).to(waveform.device),
        return_complex=True
    )

    # Get components needed by the model
    noisy_real = stft.real
    noisy_imag = stft.imag
    noisy_mag = torch.sqrt(noisy_real ** 2 + noisy_imag ** 2)

    # Add batch and channel dimensions [B, 1, F, T]
    noisy_mag = noisy_mag.unsqueeze(0)
    noisy_real = noisy_real.unsqueeze(0)
    noisy_imag = noisy_imag.unsqueeze(0)

    return noisy_mag, noisy_real, noisy_imag


# Example usage:
# pre_train_model.eval()  # Set model to evaluation mode
# with torch.no_grad():
#     # Prepare input
#     magnitude, phase = prepare_input("path_to_your_audio.wav")
#
#     # Forward pass through model
#     enhanced_magnitude = pre_train_model(magnitude)
#
#     # Convert back to audio if needed
#     # ... additional processing to convert enhanced magnitude back to waveform


def get_device(device_preference='cuda'):
    """Helper function to get the device to use"""
    if device_preference == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def crm_to_stft_components(crm: torch.Tensor, noisy_real: torch.Tensor, noisy_imag: torch.Tensor) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    # Remove channel dimension from noisy components
    noisy_real = noisy_real.squeeze(1)
    noisy_imag = noisy_imag.squeeze(1)
    enhanced_real, enhanced_imag = noisy_to_enhanced(crm,noisy_real, noisy_imag)

    enhanced_mag = torch.sqrt(enhanced_real ** 2 + enhanced_imag ** 2)
    return enhanced_mag, enhanced_real, enhanced_imag


def crm_to_spectogram(curr_pc_crm, noisy_complex):
    enhanced_real = curr_pc_crm[..., 0] * noisy_complex.real - curr_pc_crm[..., 1] * noisy_complex.imag
    enhanced_imag = curr_pc_crm[..., 1] * noisy_complex.real + curr_pc_crm[..., 0] * noisy_complex.imag
    enhanced_complex = torch.complex(enhanced_real, enhanced_imag)
    return enhanced_complex

