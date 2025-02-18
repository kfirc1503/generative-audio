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


# def compute_pca_and_importance_weights(outputs):
#     """
#     Apply PCA (via SVD) on multiple MC-Dropout outputs and compute importance weights.
#
#     outputs: Tensor of shape (K, B, D) - Multiple inpainted spectrograms
#     Returns:
#         - principal_components: Top principal components
#         - importance_weights: Contribution of each component
#         - mean_prediction: Mean of the outputs
#     """
#     K, B, D = outputs.shape  # (num_samples, batch_size, feature_dim)
#
#     # Compute mean prediction (E[x|y])
#     mean_prediction = outputs.mean(dim=0)  # (B, D)
#
#     # Compute residuals (centered data)
#     centered_outputs = outputs - mean_prediction.unsqueeze(0)  # (K, B, D)
#
#     # Flatten over batch dimension
#     centered_outputs_flat = centered_outputs.view(K, -1).cpu().numpy()  # Shape: (K, B*D)
#
#     # Perform PCA using SVD
#     pca = PCA(n_components=min(K, 5))  # Extract top 5 PCs (or K if smaller)
#     pca.fit(centered_outputs_flat)
#
#     # Extract principal components
#     principal_components = pca.components_  # Shape: (num_PCs, B*D)
#
#     # Compute importance weights (normalized eigenvalues)
#     eigenvalues = pca.explained_variance_
#     importance_weights = eigenvalues / eigenvalues.sum()  # Normalize to sum to 1
#
#     return principal_components, importance_weights, mean_prediction

import torch
import numpy as np
from sklearn.decomposition import PCA

def compute_pca_sklearn_batch(outputs, n_components=5):
    """
    Perform PCA independently on each item in a batch using scikit-learn's PCA.
    (Code version that expects outputs of shape (K, B, D).)

    Args:
        outputs (torch.Tensor): shape (K, B, D)
            - K: number of vectors per batch item
            - B: batch size
            - D: dimensionality of each vector
        n_components (int): number of principal components to keep

    Returns:
        principal_components        (torch.Tensor): shape (B, n_components, D)
            - The principal component directions (unit vectors).
        scaled_principal_components (torch.Tensor): shape (B, n_components, D)
            - Each principal component multiplied by its singular value
              (i.e., capturing the scale of variance).
        importance_weights          (torch.Tensor): shape (B, n_components)
            - Normalized singular values (fraction of total variance).
        mean_prediction             (torch.Tensor): shape (B, D)
            - Mean of the data for each batch item.
    """
    # According to your code, 'outputs' is shape (K, B, D)
    K, B, D = outputs.shape
    n_components = min(n_components, K)

    pcs_list = []            # Will store the unit principal components
    scaled_pcs_list = []     # Will store PCs scaled by their singular values
    weights_list = []        # Will store importance weights
    mean_list = []           # Will store means
    singular_vals_list = []

    # Loop over each batch item in dimension 1 (since shape is (K, B, D))
    for b in range(B):
        # Extract data for item b: shape (K, D)
        item_data = outputs[:, b, :]  # (K, D)

        # Convert to NumPy (for scikit-learn)
        item_np = item_data.cpu().numpy()

        # Initialize PCA
        pca = PCA(n_components=n_components)
        # Fit PCA on current item's data
        pca.fit(item_np)

        # Principal components (unit vectors), shape (n_components, D)
        pcs_np = pca.components_
        # Singular values, shape (n_components,)
        singular_vals = pca.singular_values_

        # Fraction of total variance for each PC
        importance = singular_vals / singular_vals.sum()

        # Mean vector, shape (D,)
        mean_vec_np = pca.mean_

        # --- SCALED PCS: multiply each PC by its singular value ---
        scaled_pcs_np = pcs_np * singular_vals[:, None]  # shape (n_components, D)

        # Convert back to torch
        pcs_torch = torch.from_numpy(pcs_np).float()
        scaled_pcs_torch = torch.from_numpy(scaled_pcs_np).float()
        importance_torch = torch.from_numpy(importance).float()
        mean_vec_torch = torch.from_numpy(mean_vec_np).float()
        singular_vals_torch = torch.from_numpy(singular_vals).float()

        # Accumulate in lists
        pcs_list.append(pcs_torch)
        scaled_pcs_list.append(scaled_pcs_torch)
        weights_list.append(importance_torch)
        mean_list.append(mean_vec_torch)
        singular_vals_list.append(singular_vals_torch)

    # Stack to get final shapes
    # principal_components: (B, n_components, D)
    principal_components = torch.stack(pcs_list, dim=0)
    # scaled_principal_components: (B, n_components, D)
    scaled_principal_components = torch.stack(scaled_pcs_list, dim=0)
    # importance_weights: (B, n_components)
    importance_weights = torch.stack(weights_list, dim=0)
    # mean_prediction: (B, D)
    mean_prediction = torch.stack(mean_list, dim=0)
    singular_vals = torch.stack(singular_vals_list , dim=0)

    return principal_components, scaled_principal_components, importance_weights, mean_prediction , singular_vals





def compute_pca_and_importance_weights(outputs):
    """
    Apply PCA (via SVD) on multiple MC-Dropout outputs and compute importance weights.
    Performs PCA separately for each batch item.

    Args:
        outputs: Tensor of shape (K, B, D) - Multiple inpainted spectrograms
            K: number of MC samples
            B: batch size
            D: number of masked elements
    Returns:
        - principal_components: (B, num_PCs, D)
        - importance_weights: (B, num_PCs)
        - mean_prediction: (B, D)
    """
    K, B, D = outputs.shape
    n_components = min(K, 5)  # Extract top 5 PCs (or K if smaller)

    # Initialize arrays for results
    principal_components = []
    importance_weights = []
    mean_predictions = []

    # Process each batch item separately
    for b in range(B):
        # Get predictions for current batch item
        batch_outputs = outputs[:, b, :]  # Shape: (K, D)

        # Compute mean prediction for this batch item
        mean_pred = batch_outputs.mean(dim=0)  # Shape: (D,)
        mean_predictions.append(mean_pred)

        # Center the data
        centered_outputs = batch_outputs - mean_pred.unsqueeze(0)  # Shape: (K, D)

        # Perform SVD
        U, S, Vh = torch.linalg.svd(centered_outputs, full_matrices=False)

        # Get top components and their weights
        top_components = Vh[:n_components]  # Shape: (n_components, D)
        weights = S[:n_components]  # Shape: (n_components,)
        weights = weights / weights.sum()  # Normalize weights

        principal_components.append(top_components)
        importance_weights.append(weights)

    # Stack results
    principal_components = torch.stack(principal_components)  # Shape: (B, n_components, D)
    importance_weights = torch.stack(importance_weights)  # Shape: (B, n_components)
    mean_predictions = torch.stack(mean_predictions)  # Shape: (B, D)

    return principal_components, importance_weights, mean_predictions


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
    principal_components2, importance_weights2, mean_prediction2 = compute_pca_and_importance_weights(predictions_flat)
    principal_components , scaled_principal_components, importance_weights, mean_prediction, singular_vals = compute_pca_sklearn_batch(predictions_flat)
    # turn to torch, move back to the device:
    device = masked_spec.device
    principal_components = principal_components.to(device)
    scaled_principal_components = scaled_principal_components.to(device)
    importance_weights = importance_weights.to(device)
    mean_prediction = mean_prediction.to(device)
    singular_vals = singular_vals.to(device)

    # Reconstruct full spectrograms with zeros in known regions
    _, F, T = masked_spec.shape[1:]


    # Helper function to reconstruct full spectrogram (vectorized for batch)
    def reconstruct_full_spec_batch(inpaint_values, mask):
        """
        Args:
            inpaint_values: [B, N_masked] or [B, n_components, N_masked]
            mask: [B, 1, F, T] original mask
        Returns:
            [B, F, T] or [B, n_components, F, T] tensor
        """
        B = inpaint_values.shape[0]
        F, T = mask.shape[2:]
        has_components = inpaint_values.dim() == 3

        if has_components:
            n_comp = inpaint_values.shape[1]
            full_spec = torch.zeros((B, n_comp, F, T), device=masked_spec.device)
            mask_flat = mask.reshape(B, 1, F * T) == 0  # [B, 1, F*T]
            for b in range(B):
                full_spec[b, :, :, :].reshape(n_comp, -1)[:, mask_flat[b, 0]] = inpaint_values[b]
        else:
            full_spec = torch.zeros((B, F, T), device=masked_spec.device)
            mask_flat = mask.reshape(B, 1, F * T) == 0  # [B, 1, F*T]
            for b in range(B):
                full_spec[b].reshape(-1)[mask_flat[b, 0]] = inpaint_values[b]

        return full_spec

    # Reconstruct PCs for all batches at once
    full_pcs = reconstruct_full_spec_batch(principal_components, mask)
    full_pcs_scaled = reconstruct_full_spec_batch(scaled_principal_components, mask)
    full_pcs = full_pcs.reshape(B, n_components, F, T)
    # full_pcs = full_pcs.unsqueeze(1)  # [B, 1, n_components, F, T]

    # Reshape mean prediction to full spectrogram [B, 1, F, T]
    mean_prediction = mean_prediction.reshape(B, N_masked_per_batch)
    full_mean = reconstruct_full_spec_batch(mean_prediction, mask).unsqueeze(1)  # [B, 1, F, T]

    # Add shape assertions to verify
    assert full_pcs.shape == (
        B, n_components, F, T), f"PC shape is {full_pcs.shape}, expected ({B}, 1, {n_components}, {F}, {T})"
    assert full_mean.shape == (B, 1, F, T), f"Mean shape is {full_mean.shape}, expected ({B}, 1, {F}, {T})"

    return {
        'mean_prediction': full_mean,
        'principal_components': full_pcs,
        'scaled_principal_components': full_pcs_scaled,
        'importance_weights': importance_weights,
        'singular_vals': singular_vals
    }