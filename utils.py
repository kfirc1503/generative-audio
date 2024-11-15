# Import necessary libraries for audio processing
import torchaudio
import torch
import torch.nn.functional as F
from pathlib import Path
from FullSubNet_plus.speech_enhance.fullsubnet_plus.model.fullsubnet_plus import FullSubNet_Plus, FullSubNetPlusConfig

from FullSubNet_plus.speech_enhance.audio_zen.acoustics.mask import decompress_cIRM

from FullSubNet_plus.speech_enhance.audio_zen.acoustics.mask import decompress_cIRM


def model_outputs_to_waveforms(enhanced_masks, noisy_reals, noisy_imags , orig_length):
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

    # Remove channel dimension from noisy components
    noisy_reals = noisy_reals.squeeze(1)
    noisy_imags = noisy_imags.squeeze(1)

    # Apply mask using the complex multiplication formula
    enhanced_reals = enhanced_masks[..., 0] * noisy_reals - enhanced_masks[..., 1] * noisy_imags
    enhanced_imags = enhanced_masks[..., 1] * noisy_reals + enhanced_masks[..., 0] * noisy_imags

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

def preload_model(model_path: Path, model: FullSubNet_Plus) -> FullSubNet_Plus:
    """
    Preload model parameters (in "*.tar" format) at the start of experiment.

    Args:
        model_path (Path): The file path of the *.tar file
    """
    model_path = model_path.expanduser().absolute()
    assert model_path.exists(), f"The file {model_path.as_posix()} is not exist. please check path."

    model_checkpoint = torch.load(model_path.as_posix(), map_location="cpu")
    model.load_state_dict(model_checkpoint["model"], strict=False)
    return model


def load_pretrained_model(model_path: Path, model_config: FullSubNetPlusConfig) -> FullSubNet_Plus:
    model = FullSubNet_Plus(**model_config.dict())
    model = preload_model(model_path, model)
    return model


def prepare_input_from_waveform(waveform: torch.Tensor) -> tuple:
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
