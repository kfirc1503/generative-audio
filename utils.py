# Import necessary libraries for audio processing
import torchaudio
import torch
import torch.nn.functional as F
from pathlib import Path


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