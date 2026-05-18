import torch
import matplotlib.pyplot as plt
import pydantic
from pathlib import Path
from nppc_audio.inpainting.nppc.nppc_model import NPPCModel, NPPCModelConfig
from nppc_audio.inpainting.networks.unet import LatentEncoder, LatentEncoderConfig, LatentEncoderMultiLevel, LatentEncoderMultiLevelConfig
from nppc_audio.inpainting.nppc.pc_wrapper import gram_schmidt_to_spec_mag
from nppc_audio.inpainting.validator.validator_nppc_latent_model import plot_pc_spectrograms_latent
from nppc_audio.inpainting.validator.validator_nppc_model import save_pc_audio_variations
import utils
import torch.nn.functional as F
import torchaudio
import whisper


def save_precomputed_audio_variations(pc_variations_spec, clean_spec, masked_audio, metadata,
                                      mean, std, save_dir, sample_idx,
                                      n_fft=255, hop_length=128, sample_rate=16000):
    """
    Convert pre-computed spectrogram variations to audio.
    Used for latent NPPC where variations are computed in latent space then decoded.
    """
    audio_dir = Path(save_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Get phase from clean spectrogram
    clean_spec_complex = torch.complex(clean_spec[0, 0], clean_spec[0, 1])
    clean_phase = torch.angle(clean_spec_complex)
    window = torch.hann_window(n_fft).to(clean_phase.device)
    
    # Load full clean audio
    clean_audio_path_ref = metadata['clean_audio_paths'][0]
    clean_audio_full = torchaudio.load(clean_audio_path_ref)[0].squeeze(0)
    
    whisper_model = whisper.load_model("base")
    transcriptions = {}
    
    def get_with_full_audio(clean_audio_full, pred_subsample_audio, metadata):
        subsample_start_idx = metadata['subsample_start_idx'][0]
        mask_start_idx = metadata['mask_start_idx'][0]
        mask_end_idx = metadata['mask_end_idx'][0]
        pred_audio_full = clean_audio_full.clone()
        pred_audio_full[subsample_start_idx + mask_start_idx: subsample_start_idx + mask_end_idx] = \
            pred_subsample_audio[mask_start_idx: mask_end_idx]
        return pred_audio_full
    
    n_dirs, n_alphas = pc_variations_spec.shape[0], pc_variations_spec.shape[1]
    alphas = torch.arange(-3, 3.5, 0.5)
    
    for dir_idx in range(n_dirs):
        pc_dir = audio_dir / f"pc_{dir_idx + 1}"
        pc_dir.mkdir(exist_ok=True)
        
        for alpha_idx, alpha in enumerate(alphas):
            spec_var = pc_variations_spec[dir_idx, alpha_idx]
            spec_mag_log = spec_var[0, 0].cpu() * std + mean
            spec_mag_linear = torch.exp(spec_mag_log) - 1e-6
            
            real_part = spec_mag_linear * torch.cos(clean_phase.cpu())
            imag_part = spec_mag_linear * torch.sin(clean_phase.cpu())
            complex_spec = torch.complex(real_part, imag_part)
            
            audio = torch.istft(complex_spec, n_fft=n_fft, hop_length=hop_length,
                               win_length=n_fft, window=window.cpu())
            audio_full = get_with_full_audio(clean_audio_full, audio, metadata)
            
            audio_path = pc_dir / f"alpha_{alpha:.1f}.wav"
            audio_path_full = pc_dir / f"alpha_{alpha:.1f}_full.wav"
            torchaudio.save(audio_path, audio.unsqueeze(0), sample_rate=sample_rate)
            torchaudio.save(audio_path_full, audio_full.unsqueeze(0), sample_rate=sample_rate)
            
            variation_name = f'pc{dir_idx + 1}_alpha{alpha:.1f}'
            transcriptions[variation_name] = whisper_model.transcribe(
                audio_path_full.as_posix(), language="en"
            )['text']
    
    with open(audio_dir / "transcriptions.txt", "w") as f:
        f.write(f"Ground Truth: {metadata['transcriptions'][0]}\n\n")
        for name, trans in transcriptions.items():
            f.write(f"{name}: {trans}\n")
    
    print(f"  Saved {n_dirs * n_alphas} audio files to {audio_dir}")


class ComparisonValidatorConfig(pydantic.BaseModel):
    # NPPC Regular Model Configuration
    nppc_checkpoint_path: str
    nppc_model_configuration: NPPCModelConfig
    
    # NPPC Latent Model Configuration
    nppc_latent_checkpoint_path: str
    nppc_latent_model_configuration: LatentEncoderConfig
    
    # NPPC Multi-Level Latent Model Configuration (optional)
    nppc_multi_level_checkpoint_path: str = None
    nppc_multi_level_model_configuration: LatentEncoderMultiLevelConfig = None
    
    # Multi-level validation mode: 'v1' (global GS) or 'v2' (per-layer GS)
    multi_level_validation_mode: str = "v1"
    
    # Shared settings
    device: str = "cuda"
    save_dir: str = "validation_comparison_results"
    max_dirs_to_plot: int = None


class ComparisonValidator:
    def __init__(self, config: ComparisonValidatorConfig):
        self.config = config
        self.device = config.device
        if config.device == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading NPPC Regular Model...")
        # Load NPPC Regular Model
        nppc_checkpoint_path = Path(config.nppc_checkpoint_path).absolute()
        nppc_checkpoint = torch.load(nppc_checkpoint_path, map_location="cpu")
        
        self.nppc_model = NPPCModel(config.nppc_model_configuration)
        self.nppc_model.load_state_dict(nppc_checkpoint["model_state_dict"])
        self.nppc_model.to(self.device)
        self.nppc_model.eval()
        print("NPPC Regular Model loaded successfully!")
        
        print("Loading NPPC Latent Model...")
        # Load NPPC Latent Model
        nppc_latent_checkpoint_path = Path(config.nppc_latent_checkpoint_path).absolute()
        nppc_latent_checkpoint = torch.load(nppc_latent_checkpoint_path, map_location="cpu")
        
        self.nppc_latent_model = LatentEncoder(config.nppc_latent_model_configuration)
        self.nppc_latent_model.load_state_dict(nppc_latent_checkpoint["model_state_dict"])
        self.nppc_latent_model.to(self.device)
        self.nppc_latent_model.eval()
        
        # Get restoration model from NPPC model (shared between both)
        self.restoration_model = self.nppc_model.pretrained_restoration_model
        self.restoration_model.eval()
        print("NPPC Latent Model loaded successfully!")
        
        # Load NPPC Multi-Level Latent Model (optional)
        self.nppc_multi_level_model = None
        if config.nppc_multi_level_checkpoint_path and config.nppc_multi_level_model_configuration:
            print("Loading NPPC Multi-Level Latent Model...")
            nppc_multi_level_checkpoint_path = Path(config.nppc_multi_level_checkpoint_path).absolute()
            nppc_multi_level_checkpoint = torch.load(nppc_multi_level_checkpoint_path, map_location="cpu", weights_only=False)
            
            self.nppc_multi_level_model = LatentEncoderMultiLevel(config.nppc_multi_level_model_configuration)
            self.nppc_multi_level_model.load_state_dict(nppc_multi_level_checkpoint["model_state_dict"])
            self.nppc_multi_level_model.to(self.device)
            self.nppc_multi_level_model.eval()
            print("NPPC Multi-Level Latent Model loaded successfully!")

    def validate_sample(self, masked_spec, mask, clean_spec, masked_audio, metadata, sample_len_seconds, sample_idx):
        """
        Validate both models on a single sample and save results separately
        """
        with torch.no_grad():
            # Move inputs to device
            masked_spec = masked_spec.to(self.device)
            mask = mask.to(self.device)
            clean_spec = clean_spec.to(self.device)
            
            # Preprocess data (shared for both models)
            clean_spec_mag_norm_log, mask, masked_spec_mag_log, mean, std = utils.preprocess_data(
                clean_spec, masked_spec, mask, plot_mean_std=True
            )
            
            print(f"\n=== Processing Sample {sample_idx} ===")
            
            # ============================================================
            # NPPC Regular Model
            # ============================================================
            print("Running NPPC Regular Model...")
            nppc_regular_results = self._validate_nppc_regular(
                masked_spec_mag_log, 
                mask, 
                clean_spec_mag_norm_log,
                sample_len_seconds,
                metadata,
                sample_idx
            )
            
            # Generate audio for NPPC Regular
            print("Generating audio for NPPC Regular...")
            alphas = torch.arange(-3, 3.5, 0.5)
            audio_save_path = Path(self.config.save_dir) / f"sample_{sample_idx}" / "nppc_regular" / "audio"
            pitch_save_path = Path(self.config.save_dir) / f"sample_{sample_idx}" / "nppc_regular" / "pitch"
            save_pc_audio_variations(
                clean_spec_mag_norm_log.cpu(),
                nppc_regular_results['pred_spec'],
                nppc_regular_results['pc_directions'],
                clean_spec.cpu(),
                mask.cpu(),
                masked_audio.cpu(),
                metadata,
                alphas,
                audio_save_path,
                pitch_save_path,
                mean.cpu(),
                std.cpu(),
                sample_idx
            )
            
            # ============================================================
            # NPPC Latent Model
            # ============================================================
            print("Running NPPC Latent Model...")
            nppc_latent_results = self._validate_nppc_latent(
                masked_spec_mag_log,
                mask,
                clean_spec_mag_norm_log,
                sample_len_seconds,
                metadata,
                sample_idx
            )
            
            # Generate audio for NPPC Latent (from pre-computed spectrograms)
            print("Generating audio for NPPC Latent...")
            latent_audio_save_path = Path(self.config.save_dir) / f"sample_{sample_idx}" / "nppc_latent" / "audio"
            save_precomputed_audio_variations(
                nppc_latent_results['pc_variations'],
                clean_spec.cpu(),
                masked_audio.cpu(),
                metadata,
                mean.cpu(),
                std.cpu(),
                latent_audio_save_path,
                sample_idx
            )
            
            results = {
                'nppc_regular': nppc_regular_results,
                'nppc_latent': nppc_latent_results
            }
            
            # ============================================================
            # NPPC Multi-Level Latent Model (optional)
            # ============================================================
            if self.nppc_multi_level_model is not None:
                # Choose validation mode: v1 (global GS) or v2 (per-layer GS)
                if self.config.multi_level_validation_mode == "v2":
                    print("Running NPPC Multi-Level Latent Model V2 (Per-Layer GS)...")
                    nppc_multi_level_results = self._validate_nppc_latent_multi_level_v2(
                        masked_spec_mag_log,
                        mask,
                        clean_spec_mag_norm_log,
                        sample_len_seconds,
                        metadata,
                        sample_idx
                    )
                    audio_folder = "nppc_multi_level_v2"
                else:
                    print("Running NPPC Multi-Level Latent Model V1 (Global GS)...")
                    nppc_multi_level_results = self._validate_nppc_latent_multi_level(
                        masked_spec_mag_log,
                        mask,
                        clean_spec_mag_norm_log,
                        sample_len_seconds,
                        metadata,
                        sample_idx
                    )
                    audio_folder = "nppc_multi_level"
                
                # Generate audio for NPPC Multi-Level (from pre-computed spectrograms)
                print("Generating audio for NPPC Multi-Level...")
                multi_level_audio_save_path = Path(self.config.save_dir) / f"sample_{sample_idx}" / audio_folder / "audio"
                save_precomputed_audio_variations(
                    nppc_multi_level_results['pc_variations'],
                    clean_spec.cpu(),
                    masked_audio.cpu(),
                    metadata,
                    mean.cpu(),
                    std.cpu(),
                    multi_level_audio_save_path,
                    sample_idx
                )
                
                results['nppc_multi_level'] = nppc_multi_level_results
            
            print(f"Sample {sample_idx} completed!")
            
            return results
    
    def _validate_nppc_regular(self, masked_spec_mag_log, mask, clean_spec_mag_norm_log, 
                               sample_len_seconds, metadata, sample_idx):
        """
        Run NPPC regular model and generate spectrogram variations
        """
        # Get PC directions from regular model
        pc_directions = self.nppc_model(masked_spec_mag_log, mask)
        
        # Get prediction
        pred_spec_mag_log = self.nppc_model.get_pred_spec_mag_norm(masked_spec_mag_log, mask)
        
        # Generate PC variations in spectrogram space
        alphas = torch.arange(-3, 3.5, 0.5).to(self.device)
        n_dirs = pc_directions.shape[1]
        
        # Limit number of directions if specified
        if self.config.max_dirs_to_plot is not None:
            n_dirs = min(n_dirs, self.config.max_dirs_to_plot)
            pc_directions = pc_directions[:, :n_dirs]
        
        # Store all variations: [n_dirs, n_alphas, B, 1, F, T]
        pc_variations_spec = []
        
        # DEBUG: Print regular NPPC direction info
        print(f"\n🔍 DEBUG - NPPC Regular W Directions:")
        print(f"   Shape: {pc_directions.shape}")
        # Flatten all dims except batch and n_dirs, then compute norm
        pc_flat = pc_directions.flatten(start_dim=2)  # [B, n_dirs, ...]
        pc_norms = pc_flat.norm(dim=2)  # [B, n_dirs]
        print(f"   W norms: min={pc_norms.min().item():.4e}, max={pc_norms.max().item():.4e}, mean={pc_norms.mean().item():.4e}")
        print(f"   W values: min={pc_directions.min().item():.4e}, max={pc_directions.max().item():.4e}")
        
        # DEBUG: Store for comparison
        debug_outputs_regular = {}
        
        for dir_idx in range(n_dirs):
            direction_variations = []
            
            # DEBUG for direction 0
            if dir_idx == 0:
                w_dir = pc_directions[:, dir_idx:dir_idx+1]
                print(f"\n🔍 DEBUG - Regular Direction {dir_idx}:")
                print(f"   Direction norm: {w_dir.norm().item():.4e}")
                print(f"   Direction range: [{w_dir.min().item():.4e}, {w_dir.max().item():.4e}]")
            
            for alpha in alphas:
                # Add alpha * pc_direction to MODEL PREDICTION (not masked input!)
                modified_spec = pred_spec_mag_log + alpha * pc_directions[:, dir_idx:dir_idx+1]
                direction_variations.append(modified_spec)
                
                # DEBUG: Store for comparison
                if dir_idx == 0:
                    debug_outputs_regular[alpha.item()] = modified_spec.clone()
            
            pc_variations_spec.append(torch.stack(direction_variations))
        
        # DEBUG: Compare alpha=-3 vs alpha=+3 for regular NPPC
        if -3.0 in debug_outputs_regular and 3.0 in debug_outputs_regular:
            print(f"\n🔍 DEBUG - NPPC Regular: Comparing alpha=-3 vs alpha=+3 (Direction 0):")
            spec_diff = (debug_outputs_regular[3.0] - debug_outputs_regular[-3.0]).abs()
            print(f"   Spec diff: max={spec_diff.max().item():.4e}, mean={spec_diff.mean().item():.4e}")
            
            # Check masked region
            mask_region = (1 - mask).bool()
            spec_diff_masked = spec_diff[mask_region.expand_as(spec_diff)]
            if spec_diff_masked.numel() > 0:
                print(f"   Spec diff in MASKED region: max={spec_diff_masked.max().item():.4e}, mean={spec_diff_masked.mean().item():.4e}")
        
        # Stack all directions: [n_dirs, n_alphas, B, 1, F, T]
        pc_variations_spec = torch.stack(pc_variations_spec)
        
        # Plot results
        spec_save_path = Path(self.config.save_dir) / f"sample_{sample_idx}" / "nppc_regular"
        fig = plot_pc_spectrograms_latent(
            masked_spec_mag_log.cpu(),
            clean_spec_mag_norm_log.cpu(),
            pred_spec_mag_log.cpu(),
            pc_variations_spec.cpu(),
            mask.cpu(),
            sample_len_seconds,
            metadata,
            spec_save_path,
            sample_idx,
            max_dirs=self.config.max_dirs_to_plot
        )
        
        return {
            'figure': fig,
            'pc_variations': pc_variations_spec.cpu(),
            'pc_directions': pc_directions.cpu(),
            'pred_spec': pred_spec_mag_log.cpu()
        }
    
    def _validate_nppc_latent(self, masked_spec_mag_log, mask, clean_spec_mag_norm_log,
                              sample_len_seconds, metadata, sample_idx):
        """
        Run NPPC latent model and generate spectrogram variations
        """
        # Get restoration model prediction
        pred_spec_mag_log = self.restoration_model(masked_spec_mag_log, 1-mask)
        
        # Encode to latent space using the restoration model's encoder
        latent_restored, skip_connections_restored = self.restoration_model.net.encoder(pred_spec_mag_log)
        latent_distorted, skip_connections_distorted = self.restoration_model.net.encoder(masked_spec_mag_log)
        
        # Prepare input for latent NPPC model (concatenate distorted and restored)
        masked_with_pred = torch.cat((masked_spec_mag_log, pred_spec_mag_log), dim=1)
        broadcasted_mask = mask.view(1, 1, mask.shape[-2], mask.shape[-1]).expand(
            masked_spec_mag_log.shape[0], 1, mask.shape[-2], mask.shape[-1]
        )
        
        # Get PC directions in latent space
        # Note: The latent model expects inverted mask (1 - mask)
        w_mat_latent, latent_mask = self.nppc_latent_model(masked_with_pred, 1 - broadcasted_mask)
        # w_mat_latent shape: [B, n_dirs, C, H, W] where C is latent channels (e.g., 512)
        
        # Apply latent masking - same as training!
        B, n_dirs, C, H, W = w_mat_latent.shape
        
        # Flatten latent mask and broadcast to match w_mat_latent shape
        latent_mask_flat = latent_mask.flatten(start_dim=1)  # [B, H*W]
        latent_mask_flat_expanded = latent_mask_flat.unsqueeze(1)  # [B, 1, H*W]
        latent_mask_flat_broadcasted = latent_mask_flat_expanded.expand(-1, n_dirs, -1)  # [B, n_dirs, H*W]
        # Repeat each mask value C times to match the channel dimension
        latent_mask_flat_broadcasted = latent_mask_flat_broadcasted.repeat_interleave(C, dim=2)  # [B, n_dirs, C*H*W]
        
        # Flatten w_mat_latent
        w_latent_flat = w_mat_latent.flatten(start_dim=2)  # [B, n_dirs, C*H*W]
        
        # Apply latent mask (same as training)
        w_latent_flat = w_latent_flat * latent_mask_flat_broadcasted
        
        # Apply Gram-Schmidt normalization after masking
        w_mat_latent_flat = gram_schmidt_to_spec_mag(w_latent_flat)
        w_mat_latent = w_mat_latent_flat.view(B, n_dirs, C, H, W)
        
        # DEBUG: Check W direction magnitudes
        w_norms = w_mat_latent_flat.norm(dim=2)
        print(f"\n🔍 DEBUG - NPPC Latent W Directions:")
        print(f"   W norms: min={w_norms.min().item():.4e}, max={w_norms.max().item():.4e}, mean={w_norms.mean().item():.4e}")
        print(f"   W values: min={w_mat_latent.min().item():.4e}, max={w_mat_latent.max().item():.4e}")
        print(f"   Latent mask sum: {latent_mask.sum().item()} (should be > 0)")
        
        # Limit number of directions if specified
        if self.config.max_dirs_to_plot is not None:
            n_dirs = min(n_dirs, self.config.max_dirs_to_plot)
            w_mat_latent = w_mat_latent[:, :n_dirs]
        else:
            n_dirs = w_mat_latent.shape[1]
        
        # Generate variations by adding PC directions in LATENT space, then decoding
        alphas = torch.arange(-3, 3.5, 0.5).to(self.device)
        
        # Store all variations: [n_dirs, n_alphas, B, 1, F, T]
        pc_variations_spec = []
        
        # DEBUG: Store outputs for comparison
        debug_outputs = {}
        
        for dir_idx in range(n_dirs):
            direction_variations = []
            w_direction = w_mat_latent[:, dir_idx, :, :]
            
            # DEBUG: Check direction magnitude
            if dir_idx == 0:
                print(f"\n🔍 DEBUG - Direction {dir_idx}:")
                print(f"   Direction norm: {w_direction.norm().item():.4e}")
                print(f"   Direction range: [{w_direction.min().item():.4e}, {w_direction.max().item():.4e}]")
            
            for alpha in alphas:
                # Add scaled direction to latent representation of MASKED input
                latent_var = latent_distorted + alpha * w_direction
                
                # Then decode the modified latent
                # CRITICAL: Use skip_connections_distorted (from masked input)
                decoded_output = self.restoration_model.net.decoder(latent_var, skip_connections_distorted)
                
                # IMPORTANT: Add residual connection (same as RestorationWrapper line 387)
                spec_var = masked_spec_mag_log + decoded_output * (1 - mask)
                
                # DEBUG: Store for comparison
                if dir_idx == 0:
                    debug_outputs[alpha.item()] = {
                        'latent_var': latent_var.clone(),
                        'decoded_output': decoded_output.clone(),
                        'spec_var': spec_var.clone()
                    }
                
                direction_variations.append(spec_var)
            
            pc_variations_spec.append(torch.stack(direction_variations))
        
        # DEBUG: Compare alpha=-3 vs alpha=+3 for direction 0
        if -3.0 in debug_outputs and 3.0 in debug_outputs:
            print(f"\n🔍 DEBUG - Comparing alpha=-3 vs alpha=+3 (Direction 0):")
            
            latent_diff = (debug_outputs[3.0]['latent_var'] - debug_outputs[-3.0]['latent_var']).abs()
            print(f"   Latent diff: max={latent_diff.max().item():.4e}, mean={latent_diff.mean().item():.4e}")
            
            decoded_diff = (debug_outputs[3.0]['decoded_output'] - debug_outputs[-3.0]['decoded_output']).abs()
            print(f"   Decoded output diff: max={decoded_diff.max().item():.4e}, mean={decoded_diff.mean().item():.4e}")
            
            spec_diff = (debug_outputs[3.0]['spec_var'] - debug_outputs[-3.0]['spec_var']).abs()
            print(f"   Spec var diff: max={spec_diff.max().item():.4e}, mean={spec_diff.mean().item():.4e}")
            
            # Check if decoder is ignoring latent (skip connections dominating)
            # Decode with ZERO latent to see baseline
            zero_latent = torch.zeros_like(latent_distorted)
            decoded_from_zero = self.restoration_model.net.decoder(zero_latent, skip_connections_distorted)
            decoded_from_distorted = self.restoration_model.net.decoder(latent_distorted, skip_connections_distorted)
            
            zero_vs_distorted = (decoded_from_zero - decoded_from_distorted).abs()
            print(f"\n🔍 DEBUG - Skip connection dominance test:")
            print(f"   Decoded(zero_latent) vs Decoded(distorted_latent):")
            print(f"   Diff: max={zero_vs_distorted.max().item():.4e}, mean={zero_vs_distorted.mean().item():.4e}")
            
            # Also check the masked region specifically
            mask_region = (1 - mask).bool()
            spec_diff_masked = spec_diff[mask_region.expand_as(spec_diff)]
            if spec_diff_masked.numel() > 0:
                print(f"\n🔍 DEBUG - Spec diff in MASKED region only:")
                print(f"   max={spec_diff_masked.max().item():.4e}, mean={spec_diff_masked.mean().item():.4e}")
            
            pc_variations_spec.append(torch.stack(direction_variations))
        
        # Stack all directions: [n_dirs, n_alphas, B, 1, F, T]
        pc_variations_spec = torch.stack(pc_variations_spec)
        
        # Plot results
        spec_save_path = Path(self.config.save_dir) / f"sample_{sample_idx}" / "nppc_latent"
        fig = plot_pc_spectrograms_latent(
            masked_spec_mag_log.cpu(),
            clean_spec_mag_norm_log.cpu(),
            pred_spec_mag_log.cpu(),
            pc_variations_spec.cpu(),
            mask.cpu(),
            sample_len_seconds,
            metadata,
            spec_save_path,
            sample_idx,
            max_dirs=self.config.max_dirs_to_plot
        )
        
        return {
            'figure': fig,
            'pc_variations': pc_variations_spec.cpu()
        }

    def _validate_nppc_latent_multi_level(self, masked_spec_mag_log, mask, clean_spec_mag_norm_log,
                                          sample_len_seconds, metadata, sample_idx):
        """
        Run NPPC multi-level latent model and generate spectrogram variations.
        This modifies BOTH bottleneck AND skip connections during decoding.
        """
        # Get restoration model prediction
        pred_spec_mag_log = self.restoration_model(masked_spec_mag_log, 1-mask)
        
        # Encode to latent space using the restoration model's encoder
        latent_restored, skip_connections_restored = self.restoration_model.net.encoder(pred_spec_mag_log)
        latent_distorted, skip_connections_distorted = self.restoration_model.net.encoder(masked_spec_mag_log)
        
        # Prepare input for multi-level NPPC model
        masked_with_pred = torch.cat((masked_spec_mag_log, pred_spec_mag_log), dim=1)
        broadcasted_mask = mask.view(1, 1, mask.shape[-2], mask.shape[-1]).expand(
            masked_spec_mag_log.shape[0], 1, mask.shape[-2], mask.shape[-1]
        )
        
        # Get PC directions for ALL levels (bottleneck + skip connections)
        w_bottleneck, w_skips, level_masks = self.nppc_multi_level_model(
            masked_with_pred, 1 - broadcasted_mask
        )
        # w_skips = [w_skip4, w_skip3, w_skip2, w_skip1]
        
        n_dirs = self.nppc_multi_level_model.n_dirs
        B = masked_spec_mag_log.shape[0]
        
        # Apply masking and Gram-Schmidt to each level (same as training)
        w_flat_list = []
        level_sizes = []
        
        # Process bottleneck
        if w_bottleneck is not None:
            mask_b = level_masks['bottleneck']
            B_m, _, H_m, W_m = mask_b.shape
            mask_b_full = mask_b.expand(B_m, w_bottleneck.shape[2], H_m, W_m)
            mask_b_full_w = mask_b_full.unsqueeze(1).expand(-1, n_dirs, -1, -1, -1)
            
            w_b_flat = w_bottleneck.flatten(start_dim=2)
            mask_b_flat_w = mask_b_full_w.flatten(start_dim=2)
            w_b_flat = w_b_flat * mask_b_flat_w
            w_flat_list.append(w_b_flat)
            level_sizes.append(('bottleneck', w_b_flat.shape[2], w_bottleneck.shape[2:]))
        
        # Process skip connections
        skip_names = ['skip4', 'skip3', 'skip2', 'skip1']
        for i, (w_skip, name) in enumerate(zip(w_skips, skip_names)):
            if w_skip is not None:
                mask_s = level_masks[name]
                B_m, _, H_m, W_m = mask_s.shape
                mask_s_full = mask_s.expand(B_m, w_skip.shape[2], H_m, W_m)
                mask_s_full_w = mask_s_full.unsqueeze(1).expand(-1, n_dirs, -1, -1, -1)
                
                w_s_flat = w_skip.flatten(start_dim=2)
                mask_s_flat_w = mask_s_full_w.flatten(start_dim=2)
                w_s_flat = w_s_flat * mask_s_flat_w
                w_flat_list.append(w_s_flat)
                level_sizes.append((name, w_s_flat.shape[2], w_skip.shape[2:]))
        
        # Concatenate all levels and apply Gram-Schmidt
        w_all_flat = torch.cat(w_flat_list, dim=2)
        
        # DEBUG: Print direction norms BEFORE Gram-Schmidt
        print(f"\n🔍 DEBUG - NPPC Multi-Level W Directions BEFORE Gram-Schmidt:")
        for dir_idx in range(w_all_flat.shape[1]):
            dir_norm = w_all_flat[:, dir_idx].norm().item()
            print(f"   Direction {dir_idx}: norm = {dir_norm:.4e}")
        
        w_all_flat = gram_schmidt_to_spec_mag(w_all_flat)
        
        # DEBUG: Print direction norms AFTER Gram-Schmidt
        print(f"\n🔍 DEBUG - NPPC Multi-Level W Directions AFTER Gram-Schmidt:")
        for dir_idx in range(w_all_flat.shape[1]):
            dir_norm = w_all_flat[:, dir_idx].norm().item()
            print(f"   Direction {dir_idx}: norm = {dir_norm:.4e}")
        
        # Split back into levels
        w_levels = {}
        offset = 0
        for name, size, shape in level_sizes:
            w_level_flat = w_all_flat[:, :, offset:offset+size]
            w_levels[name] = w_level_flat.view(B, n_dirs, *shape)
            offset += size
        
        # DEBUG: Print W norms for each level
        print(f"\n🔍 DEBUG - NPPC Multi-Level W Directions per level:")
        for name, w_level in w_levels.items():
            w_flat = w_level.flatten(start_dim=2)
            w_norms = w_flat.norm(dim=2)
            print(f"   {name}: W norms min={w_norms.min().item():.4e}, max={w_norms.max().item():.4e}, mean={w_norms.mean().item():.4e}")
        
        # Limit number of directions if specified
        if self.config.max_dirs_to_plot is not None:
            n_dirs = min(n_dirs, self.config.max_dirs_to_plot)
            w_levels = {k: v[:, :n_dirs] for k, v in w_levels.items()}
        
        # Generate variations by adding PC directions in latent space AND skip connections
        alphas = torch.arange(-3, 3.5, 0.5).to(self.device)
        
        pc_variations_spec = []
        debug_outputs = {}
        
        for dir_idx in range(n_dirs):
            direction_variations = []
            
            for alpha in alphas:
                # Apply W to bottleneck
                latent_var = latent_distorted.clone()
                if 'bottleneck' in w_levels:
                    w_bottleneck_dir = w_levels['bottleneck'][:, dir_idx]
                    latent_var = latent_distorted + alpha * w_bottleneck_dir
                
                # Apply W to skip connections
                skips_var = []
                skip_level_names = ['skip4', 'skip3', 'skip2', 'skip1']
                for j, (skip_conn, skip_name) in enumerate(zip(skip_connections_distorted, skip_level_names)):
                    if skip_name in w_levels:
                        w_skip_dir = w_levels[skip_name][:, dir_idx]
                        skips_var.append(skip_conn + alpha * w_skip_dir)
                    else:
                        skips_var.append(skip_conn)
                
                # Decode with modified latent AND modified skip connections
                decoded_output = self.restoration_model.net.decoder(latent_var, skips_var)
                
                # Add residual connection
                spec_var = masked_spec_mag_log + decoded_output * (1 - mask)
                
                # DEBUG: Store for comparison
                if dir_idx == 0:
                    debug_outputs[alpha.item()] = {
                        'latent_var': latent_var.clone(),
                        'decoded_output': decoded_output.clone(),
                        'spec_var': spec_var.clone()
                    }
                
                direction_variations.append(spec_var)
            
            pc_variations_spec.append(torch.stack(direction_variations))
        
        # DEBUG: Compare alpha=-3 vs alpha=+3 for direction 0
        if -3.0 in debug_outputs and 3.0 in debug_outputs:
            print(f"\n🔍 DEBUG - Multi-Level: Comparing alpha=-3 vs alpha=+3 (Direction 0):")
            
            latent_diff = (debug_outputs[3.0]['latent_var'] - debug_outputs[-3.0]['latent_var']).abs()
            print(f"   Latent diff: max={latent_diff.max().item():.4e}, mean={latent_diff.mean().item():.4e}")
            
            decoded_diff = (debug_outputs[3.0]['decoded_output'] - debug_outputs[-3.0]['decoded_output']).abs()
            print(f"   Decoded output diff: max={decoded_diff.max().item():.4e}, mean={decoded_diff.mean().item():.4e}")
            
            spec_diff = (debug_outputs[3.0]['spec_var'] - debug_outputs[-3.0]['spec_var']).abs()
            print(f"   Spec var diff: max={spec_diff.max().item():.4e}, mean={spec_diff.mean().item():.4e}")
            
            # Check spec diff in masked region
            mask_region = (1 - mask).bool()
            spec_diff_masked = spec_diff[mask_region.expand_as(spec_diff)]
            if spec_diff_masked.numel() > 0:
                print(f"   Spec diff in MASKED region: max={spec_diff_masked.max().item():.4e}, mean={spec_diff_masked.mean().item():.4e}")
            
            # Additional test: Compare with ONLY bottleneck modified (like old latent model)
            # This helps verify skip modifications are actually contributing
            latent_only_var = latent_distorted + 3.0 * w_levels['bottleneck'][:, 0]
            decoded_bottleneck_only = self.restoration_model.net.decoder(latent_only_var, skip_connections_distorted)
            decoded_with_skips = debug_outputs[3.0]['decoded_output']
            
            bottleneck_vs_full = (decoded_bottleneck_only - decoded_with_skips).abs()
            print(f"\n🔍 DEBUG - Skip contribution test (alpha=3):")
            print(f"   Decoded(bottleneck_only) vs Decoded(bottleneck+skips):")
            print(f"   Diff: max={bottleneck_vs_full.max().item():.4e}, mean={bottleneck_vs_full.mean().item():.4e}")
            
            # Also test: decoder with zero bottleneck vs original (skip dominance check)
            zero_latent = torch.zeros_like(latent_distorted)
            decoded_from_zero = self.restoration_model.net.decoder(zero_latent, skip_connections_distorted)
            decoded_from_distorted = self.restoration_model.net.decoder(latent_distorted, skip_connections_distorted)
            zero_vs_distorted = (decoded_from_zero - decoded_from_distorted).abs()
            print(f"\n🔍 DEBUG - Skip connection dominance test (baseline):")
            print(f"   Decoded(zero_latent) vs Decoded(distorted_latent):")
            print(f"   Diff: max={zero_vs_distorted.max().item():.4e}, mean={zero_vs_distorted.mean().item():.4e}")
        
        # Stack all directions
        pc_variations_spec = torch.stack(pc_variations_spec)
        
        # Plot results
        spec_save_path = Path(self.config.save_dir) / f"sample_{sample_idx}" / "nppc_multi_level"
        fig = plot_pc_spectrograms_latent(
            masked_spec_mag_log.cpu(),
            clean_spec_mag_norm_log.cpu(),
            pred_spec_mag_log.cpu(),
            pc_variations_spec.cpu(),
            mask.cpu(),
            sample_len_seconds,
            metadata,
            spec_save_path,
            sample_idx,
            max_dirs=self.config.max_dirs_to_plot
        )
        
        return {
            'figure': fig,
            'pc_variations': pc_variations_spec.cpu()
        }

    def _validate_nppc_latent_multi_level_v2(self, masked_spec_mag_log, mask, clean_spec_mag_norm_log,
                                              sample_len_seconds, metadata, sample_idx):
        """
        Run NPPC multi-level latent model V2 with PER-LAYER Gram-Schmidt.
        
        DIFFERENCE from v1:
        - V1: Concatenate all W, apply GLOBAL Gram-Schmidt, then split
        - V2: Apply Gram-Schmidt PER LAYER separately
        
        This matches the training of latent_multi_level_v2 mode.
        """
        # Get restoration model prediction
        pred_spec_mag_log = self.restoration_model(masked_spec_mag_log, 1-mask)
        
        # Encode to latent space
        latent_restored, skip_connections_restored = self.restoration_model.net.encoder(pred_spec_mag_log)
        latent_distorted, skip_connections_distorted = self.restoration_model.net.encoder(masked_spec_mag_log)
        
        # Prepare input for multi-level NPPC model
        masked_with_pred = torch.cat((masked_spec_mag_log, pred_spec_mag_log), dim=1)
        broadcasted_mask = mask.view(1, 1, mask.shape[-2], mask.shape[-1]).expand(
            masked_spec_mag_log.shape[0], 1, mask.shape[-2], mask.shape[-1]
        )
        
        # Get PC directions for ALL levels
        w_bottleneck, w_skips, level_masks = self.nppc_multi_level_model(
            masked_with_pred, 1 - broadcasted_mask
        )
        
        n_dirs = self.nppc_multi_level_model.n_dirs
        B = masked_spec_mag_log.shape[0]
        
        # Apply Gram-Schmidt PER LAYER (not global!)
        w_levels = {}
        
        # Process bottleneck
        if w_bottleneck is not None:
            mask_b = level_masks['bottleneck']
            B_m, _, H_m, W_m = mask_b.shape
            mask_b_full = mask_b.expand(B_m, w_bottleneck.shape[2], H_m, W_m)
            mask_b_full_w = mask_b_full.unsqueeze(1).expand(-1, n_dirs, -1, -1, -1)
            
            w_b_flat = w_bottleneck.flatten(start_dim=2)
            mask_b_flat_w = mask_b_full_w.flatten(start_dim=2)
            w_b_flat = w_b_flat * mask_b_flat_w
            
            # Per-layer Gram-Schmidt
            w_b_flat = gram_schmidt_to_spec_mag(w_b_flat)
            w_levels['bottleneck'] = w_b_flat.view(B, n_dirs, *w_bottleneck.shape[2:])
        
        # Process skip connections
        skip_names = ['skip4', 'skip3', 'skip2', 'skip1']
        for i, (w_skip, name) in enumerate(zip(w_skips, skip_names)):
            if w_skip is not None:
                mask_s = level_masks[name]
                B_m, _, H_m, W_m = mask_s.shape
                mask_s_full = mask_s.expand(B_m, w_skip.shape[2], H_m, W_m)
                mask_s_full_w = mask_s_full.unsqueeze(1).expand(-1, n_dirs, -1, -1, -1)
                
                w_s_flat = w_skip.flatten(start_dim=2)
                mask_s_flat_w = mask_s_full_w.flatten(start_dim=2)
                w_s_flat = w_s_flat * mask_s_flat_w
                
                # Per-layer Gram-Schmidt
                w_s_flat = gram_schmidt_to_spec_mag(w_s_flat)
                w_levels[name] = w_s_flat.view(B, n_dirs, *w_skip.shape[2:])
        
        # DEBUG: Print W norms for each level after per-layer GS
        print(f"\n🔍 DEBUG - NPPC Multi-Level V2 (Per-Layer GS):")
        for name, w_level in w_levels.items():
            for dir_idx in range(min(n_dirs, 5)):
                w_dir = w_level[:, dir_idx]
                print(f"   {name} Dir {dir_idx}: norm = {w_dir.norm().item():.4e}")
        
        # Limit number of directions if specified
        if self.config.max_dirs_to_plot is not None:
            n_dirs = min(n_dirs, self.config.max_dirs_to_plot)
            w_levels = {k: v[:, :n_dirs] for k, v in w_levels.items()}
        
        # Generate variations (same as v1 from here)
        alphas = torch.arange(-3, 3.5, 0.5).to(self.device)
        
        pc_variations_spec = []
        
        for dir_idx in range(n_dirs):
            direction_variations = []
            
            for alpha in alphas:
                # Apply W to bottleneck
                latent_var = latent_distorted.clone()
                if 'bottleneck' in w_levels:
                    w_bottleneck_dir = w_levels['bottleneck'][:, dir_idx]
                    latent_var = latent_distorted + alpha * w_bottleneck_dir
                
                # Apply W to skip connections
                skips_var = []
                skip_level_names = ['skip4', 'skip3', 'skip2', 'skip1']
                for j, (skip_conn, skip_name) in enumerate(zip(skip_connections_distorted, skip_level_names)):
                    if skip_name in w_levels:
                        w_skip_dir = w_levels[skip_name][:, dir_idx]
                        skips_var.append(skip_conn + alpha * w_skip_dir)
                    else:
                        skips_var.append(skip_conn)
                
                # Decode with modified latent AND modified skip connections
                decoded_output = self.restoration_model.net.decoder(latent_var, skips_var)
                
                # Add residual connection
                spec_var = masked_spec_mag_log + decoded_output * (1 - mask)
                
                direction_variations.append(spec_var)
            
            pc_variations_spec.append(torch.stack(direction_variations))
        
        # Stack all directions
        pc_variations_spec = torch.stack(pc_variations_spec)
        
        # Plot results
        spec_save_path = Path(self.config.save_dir) / f"sample_{sample_idx}" / "nppc_multi_level_v2"
        fig = plot_pc_spectrograms_latent(
            masked_spec_mag_log.cpu(),
            clean_spec_mag_norm_log.cpu(),
            pred_spec_mag_log.cpu(),
            pc_variations_spec.cpu(),
            mask.cpu(),
            sample_len_seconds,
            metadata,
            spec_save_path,
            sample_idx,
            max_dirs=self.config.max_dirs_to_plot
        )
        
        return {
            'figure': fig,
            'pc_variations': pc_variations_spec.cpu()
        }

