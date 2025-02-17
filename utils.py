# Import necessary libraries for audio processing
import torchaudio
import torch
from typing import Union, Tuple, List, Any
from pathlib import Path
import torch.nn as nn
from sklearn.decomposition import PCA
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


class OptimizerConfig(pydantic.BaseModel):
    type: str
    args: dict


class DataLoaderConfig(pydantic.BaseModel):
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = False


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


def audio_to_stft(waveform: torch.Tensor, stft_configuration: StftConfig, device: torch.device) -> torch.Tensor:
    # Ensure input has batch dimension
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() == 2 and waveform.size(0) > 1:
        pass
        # waveform = waveform.unsqueeze(1)  # Add channel dimension
    window = torch.hann_window(stft_configuration.win_length).to(device)

    # Calculate STFT (matching the parameters from the paper)
    stft = torch.stft(
        waveform,
        n_fft=stft_configuration.nfft,  # Results in 257 frequency bins
        hop_length=stft_configuration.hop_length,  # 50% overlap
        win_length=stft_configuration.win_length,
        window=window,
        center=True,
        return_complex=True
    )
    # `stft` is a complex tensor of shape [B, F, T]
    real_part = stft.real  # shape [B, F, T]
    imag_part = stft.imag  # shape [B, F, T]

    # Stack along a new channel dimension to get [B, 2, F, T]
    stft_real_imag = torch.stack([real_part, imag_part], dim=1)  # shape [B, 2, F, T]
    return stft_real_imag


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
    enhanced_real, enhanced_imag = noisy_to_enhanced(crm, noisy_real, noisy_imag)

    enhanced_mag = torch.sqrt(enhanced_real ** 2 + enhanced_imag ** 2)
    return enhanced_mag, enhanced_real, enhanced_imag


def crm_to_spectogram(curr_pc_crm, noisy_complex):
    enhanced_real = curr_pc_crm[..., 0] * noisy_complex.real - curr_pc_crm[..., 1] * noisy_complex.imag
    enhanced_imag = curr_pc_crm[..., 1] * noisy_complex.real + curr_pc_crm[..., 0] * noisy_complex.imag
    enhanced_complex = torch.complex(enhanced_real, enhanced_imag)
    return enhanced_complex


def normalize_spectrograms(spec):
    """Standardize to zero mean and unit variance"""
    B, C, F, T = spec.shape
    spec_flat = spec.view(B, C, -1)
    spec_mean = spec_flat.mean(dim=2, keepdim=True).unsqueeze(-1)
    spec_std = spec_flat.std(dim=2, keepdim=True).unsqueeze(-1)
    return (spec - spec_mean) / (spec_std + 1e-6), spec_mean, spec_std


def denormalize_spectrograms(spec_norm, spec_mean, spec_std):
    """Denormalize back to original scale"""
    return spec_norm * (spec_std + 1e-6) + spec_mean


def preprocess_log_magnitude(magnitude, eps=1e-6):
    """
    Convert magnitude spectrogram to normalized log-magnitude spectrogram.

    Args:
        magnitude (torch.Tensor): Input magnitude spectrogram.
        eps (float): Small constant to avoid log(0).

    Returns:
        torch.Tensor: Normalized log-magnitude spectrogram.
        torch.Tensor: Mean of the log-magnitude spectrogram.
        torch.Tensor: Standard deviation of the log-magnitude spectrogram.
    """
    log_mag = torch.log(magnitude + eps)
    mean = log_mag.mean()
    std = log_mag.std()
    # normalized_log_mag = log_mag
    normalized_log_mag = (log_mag - mean) / std
    return normalized_log_mag, mean, std


def preprocess_data(clean_spec, masked_spec, mask, plot_mean_std=False):
    mask = mask.unsqueeze(1).unsqueeze(2)
    mask = mask.expand(-1, 1, clean_spec.shape[2], -1)
    clean_spec_mag = torch.sqrt(clean_spec[:, 0, :, :] ** 2 + clean_spec[:, 1, :, :] ** 2)
    clean_spec_mag = clean_spec_mag.unsqueeze(1)
    masked_spec_mag = torch.sqrt(masked_spec[:, 0, :, :] ** 2 + masked_spec[:, 1, :, :] ** 2)
    masked_spec_mag = masked_spec_mag.unsqueeze(1)
    clean_spec_mag_norm_log, mean, std = preprocess_log_magnitude(clean_spec_mag)
    masked_spec_mag_log = torch.log(masked_spec_mag + 1e-6)
    masked_spec_mag_norm_log = (masked_spec_mag_log - mean) / std
    if plot_mean_std:
        return clean_spec_mag_norm_log, mask, masked_spec_mag_norm_log, mean, std
    return clean_spec_mag_norm_log, mask, masked_spec_mag_norm_log


def collate_fn(batch: List[Any]):
    """Custom collate function to handle AudioInpaintingSample batching."""

    # Stack tensors for training
    stft_masked = torch.stack([b.stft_masked for b in batch])
    mask_frames = torch.stack([b.mask_frames for b in batch])
    stft_clean = torch.stack([b.stft_clean for b in batch])
    masked_audio = torch.stack([b.masked_audio for b in batch])

    # Collect metadata in a dictionary
    metadata = {
        "clean_audio_paths": [str(b.clean_audio_path) for b in batch],
        "subsample_start_idx": [b.subsample_start_idx for b in batch],
        "mask_start_idx": [b.mask_start_idx for b in batch],
        "mask_end_idx": [b.mask_end_idx for b in batch],
        "mask_start_frame_idx": [b.mask_start_frame_idx for b in batch],
        "mask_end_frame_idx": [b.mask_end_frame_idx for b in batch],
        "transcriptions": [b.transcription for b in batch],
        "sample_rates": [b.sample_rate for b in batch],
    }

    return stft_masked, mask_frames ,stft_clean, masked_audio, metadata



def enable_dropout(model):
    """ Enable Dropout layers during inference for MC-Dropout """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()  # Keep Dropout active at inference time


def mc_dropout_inference(model, input_tensor, K=50):
    """
    Perform Monte Carlo Dropout inference to generate K outputs.

    model: The neural network (U-Net, Transformer, etc.)
    input_tensor: Input speech spectrogram (batch_size, input_dim)
    K: Number of stochastic forward passes
    """
    enable_dropout(model)  # Activate dropout during inference

    outputs = torch.stack([model(input_tensor) for _ in range(K)], dim=0)  # Shape: (K, B, D)
    return outputs


def compute_pca_and_importance_weights(outputs):
    """
    Apply PCA (via SVD) on multiple MC-Dropout outputs and compute importance weights.

    outputs: Tensor of shape (K, B, D) - Multiple inpainted spectrograms
    Returns:
        - principal_components: Top principal components
        - importance_weights: Contribution of each component
        - mean_prediction: Mean of the outputs
    """
    K, B, D = outputs.shape  # (num_samples, batch_size, feature_dim)

    # Compute mean prediction (E[x|y])
    mean_prediction = outputs.mean(dim=0)  # (B, D)

    # Compute residuals (centered data)
    centered_outputs = outputs - mean_prediction.unsqueeze(0)  # (K, B, D)

    # Flatten over batch dimension
    centered_outputs_flat = centered_outputs.view(K, -1).cpu().numpy()  # Shape: (K, B*D)

    # Perform PCA using SVD
    pca = PCA(n_components=min(K, 5))  # Extract top 5 PCs (or K if smaller)
    pca.fit(centered_outputs_flat)

    # Extract principal components
    principal_components = pca.components_  # Shape: (num_PCs, B*D)

    # Compute importance weights (normalized eigenvalues)
    eigenvalues = pca.explained_variance_
    importance_weights = eigenvalues / eigenvalues.sum()  # Normalize to sum to 1

    return principal_components, importance_weights, mean_prediction



def calculate_unet_baseline(model, masked_spec, mask, n_mc_samples=50, n_components=5):
    """
    Calculate U-Net baseline with MC Dropout and PCA analysis

    Args:
        model: The U-Net model
        masked_spec: Masked spectrogram input [B, 1, F, T]
        mask: Binary mask [B, 1, F, T] (1 for known regions, 0 for inpainting area)
        n_mc_samples: Number of MC dropout samples
        n_components: Number of principal components to extract
    Returns:
        dict containing:
        - mean_prediction: [1, 1, F, T]
        - principal_components: [1, n_components, F, T]  # Exactly this shape
        - importance_weights: [n_components]
    """
    # Enable dropout
    enable_dropout(model)

    # Collect MC samples
    mc_predictions = []
    B = masked_spec.shape[0]  # Batch size

    for _ in range(n_mc_samples):
        with torch.no_grad():
            pred = model(masked_spec, mask)  # Shape: [B, 1, F, T]

            # Extract inpainting area using each sample's own mask
            pred_flat = pred.reshape(B, 1, -1)  # [B, 1, F*T]
            mask_flat = mask.reshape(B, 1, -1)  # [B, 1, F*T]
            pred_inpaint = pred_flat[mask_flat == 0].reshape(B, -1)  # [B, N_masked]
            mc_predictions.append(pred_inpaint)

    # Stack predictions: [n_mc, N_masked_elements]
    mc_predictions = torch.stack(mc_predictions)

    # Reshape for PCA: [n_mc, B, N_masked_per_batch]
    N_masked_per_batch = (~mask[0,0].bool()).sum()   # Number of masked elements per batch item
    predictions_flat = mc_predictions.reshape(n_mc_samples, B, N_masked_per_batch)

    # Apply PCA analysis
    principal_components, importance_weights, mean_prediction = compute_pca_and_importance_weights(predictions_flat)
    # turn to torch, move back to the device:
    device = masked_spec.device
    principal_components = torch.from_numpy(principal_components).to(device)
    importance_weights = torch.from_numpy(importance_weights).to(device)
    mean_prediction = mean_prediction.to(device)

    # Reconstruct full spectrograms with zeros in known regions
    _, F, T = masked_spec.shape[1:]

    # Helper function to reconstruct full spectrogram
    def reconstruct_full_spec(inpaint_values):
        full_spec = torch.zeros((F, T), device=masked_spec.device)
        full_spec.reshape(-1)[mask[0, 0].reshape(-1) == 0] = inpaint_values
        return full_spec

    # Reshape principal components back to full spectrograms
    principal_components = principal_components.reshape(n_components, B, N_masked_per_batch)

    # Helper function to reconstruct full spectrogram (vectorized for batch)
    def reconstruct_full_spec_batch(inpaint_values):
        """
        Args:
            inpaint_values: [B, N_masked] tensor
        Returns:
            [B, F, T] tensor
        """
        full_spec = torch.zeros((B, F, T), device=masked_spec.device)
        full_spec.reshape(B, -1)[:, mask[0,0].reshape(-1) == 0] = inpaint_values
        return full_spec

    # Reconstruct PCs for all batches at once
    full_pcs = reconstruct_full_spec_batch(principal_components.transpose(0, 1).reshape(B, n_components * N_masked_per_batch))
    full_pcs = full_pcs.reshape(B, n_components, F, T)
    full_pcs = full_pcs.unsqueeze(1)  # [B, 1, n_components, F, T]

    # Reshape mean prediction to full spectrogram [B, 1, F, T]
    mean_prediction = mean_prediction.reshape(B, N_masked_per_batch)
    full_mean = reconstruct_full_spec_batch(mean_prediction).unsqueeze(1)  # [B, 1, F, T]

    # Add shape assertions to verify
    assert full_pcs.shape == (
        B, 1, n_components, F, T), f"PC shape is {full_pcs.shape}, expected ({B}, 1, {n_components}, {F}, {T})"
    assert full_mean.shape == (B, 1, F, T), f"Mean shape is {full_mean.shape}, expected ({B}, 1, {F}, {T})"

    return {
        'mean_prediction': full_mean,
        'principal_components': full_pcs,
        'importance_weights': importance_weights
    }