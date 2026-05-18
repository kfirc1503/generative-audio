import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from scipy.optimize import linear_sum_assignment

from typing import Literal, Optional, List
import pydantic
import torch.optim as optim
import json
import os
from datetime import datetime
from pathlib import Path
from tqdm.auto import tqdm
from nppc.auxil import LoopLoader
import matplotlib.pyplot as plt
from nppc_audio.inpainting.nppc.pc_wrapper import AudioInpaintingPCWrapperConfig, AudioInpaintingPCWrapper
from nppc_audio.inpainting.networks.unet import UNet, UNetConfig, RestorationWrapper
from nppc_audio.inpainting.nppc.nppc_model import NPPCModelConfig, NPPCModel
from dataset.audio_dataset_inpainting import AudioInpaintingDataset, AudioInpaintingConfig, AudioInpaintingSample
from nppc_audio.inpainting.networks.unet import LatentEncoder, UNetConfig, LatentEncoderConfig, LatentEncoderMultiLevel, LatentEncoderMultiLevelConfig
from nppc_audio.inpainting.nppc.pc_wrapper import gram_schmidt_to_spec_mag


import utils
from nppc_audio.trainer import NPPCAudioTrainer
from utils import OptimizerConfig, DataLoaderConfig, calculate_unet_baseline


class NPPCAudioInpaintingTrainerConfig(pydantic.BaseModel):
    nppc_model_configuration: NPPCModelConfig
    nppc_latent_model_configuration: LatentEncoderConfig
    nppc_latent_multi_level_configuration: Optional[LatentEncoderMultiLevelConfig] = None  # NEW
    data_configuration: AudioInpaintingConfig
    dataloader_configuration: DataLoaderConfig
    optimizer_configuration: OptimizerConfig
    # output_dir: str
    device: str = "cuda"
    save_interval: int = 10
    log_interval: int = 100
    second_moment_loss_lambda: float = 1.0
    second_moment_loss_grace: int = 500
    max_grad_norm: float = 1.0
    
    # Training mode: 'regular', 'latent', 'latent_multi_level', 'latent_multi_level_v2'
    training_mode: str = "latent"

    use_wandb: bool = False
    wandb_project_name: Optional[str] = "generative-audio"
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_artifact_name: str = "nppc_inpainting_model"  # Single artifact for all checkpoints


class NPPCAudioInpaintingTrainer(nn.Module):
    def __init__(self, config: NPPCAudioInpaintingTrainerConfig):
        super().__init__()
        self.config = config

        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project_name,
                name=config.wandb_run_name,
                config=config.model_dump(),
                tags=config.wandb_tags
            )

        ## this is suppose to be the same thing
        # self.nppc_model = self.config.nppc_model_configuration.make_instance()
        self.nppc_model = NPPCModel(self.config.nppc_model_configuration)
        self.nppc_latent_model = LatentEncoder(self.config.nppc_latent_model_configuration)
        
        # Initialize multi-level model if configured
        self.nppc_latent_multi_level_model = None
        if self.config.nppc_latent_multi_level_configuration is not None:
            self.nppc_latent_multi_level_model = LatentEncoderMultiLevel(
                self.config.nppc_latent_multi_level_configuration
            )
        
        self.device = self.config.device
        self.nppc_latent_model.to(self.device)
        self.nppc_latent_model.train()
        
        if self.nppc_latent_multi_level_model is not None:
            self.nppc_latent_multi_level_model.to(self.device)
            self.nppc_latent_multi_level_model.train()
        # create data loader:
        dataset = AudioInpaintingDataset(config.data_configuration)
        mask = torch.zeros((1, 28, 28)).to(self.device)
        mask[:, :20, :] = 1.
        # mask = 1 - mask
        self.mask = mask


        dataset = AudioInpaintingDataset(config.data_configuration)
        # train_set = torchvision.datasets.MNIST(root='./', download=True, train=True,
        #                                        transform=torchvision.transforms.ToTensor())

        # dataloader = torch.utils.data.DataLoader(
        #     train_set,
        #     batch_size=config.dataloader_configuration.batch_size,
        #     shuffle=True,
        # )

        print(f"Total sample pairs in dataset: {len(dataset)}")

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.dataloader_configuration.batch_size,  # Adjust based on your GPU memory
            shuffle=config.dataloader_configuration.shuffle,
            num_workers=config.dataloader_configuration.num_workers,
            pin_memory=config.dataloader_configuration.pin_memory,
            collate_fn=utils.collate_fn

        )
        self.dataloader = dataloader

        self.step = 0

        # ==========================================
        # Choose which model's parameters to optimize based on training_mode
        # ==========================================
        optimizer_class = getattr(optim, config.optimizer_configuration.type)
        
        if config.training_mode == "regular":
            self.optimizer = optimizer_class(
                self.nppc_model.parameters(),
                **config.optimizer_configuration.args,
            )
        elif config.training_mode == "latent":
            self.optimizer = optimizer_class(
                self.nppc_latent_model.parameters(),
                **config.optimizer_configuration.args,
            )
        elif config.training_mode == "latent_multi_level":
            if self.nppc_latent_multi_level_model is None:
                raise ValueError("training_mode='latent_multi_level' requires nppc_latent_multi_level_configuration")
            self.optimizer = optimizer_class(
                self.nppc_latent_multi_level_model.parameters(),
                **config.optimizer_configuration.args,
            )
        elif config.training_mode == "latent_multi_level_v2":
            if self.nppc_latent_multi_level_model is None:
                raise ValueError("training_mode='latent_multi_level_v2' requires nppc_latent_multi_level_configuration")
            self.optimizer = optimizer_class(
                self.nppc_latent_multi_level_model.parameters(),
                **config.optimizer_configuration.args,
            )
        else:
            raise ValueError(f"Unknown training_mode: {config.training_mode}")

    @staticmethod
    def _collate_fn(batch: List[AudioInpaintingSample]):
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
            "transcriptions": [b.transcription for b in batch],
            "sample_rates": [b.sample_rate for b in batch],
        }

        return stft_masked, mask_frames, stft_clean, masked_audio, metadata

    def train(self, n_steps=None, n_epochs=None, checkpoint_dir="checkpoints", save_flag=True, val_dataloader=None):
        """Main training loop using LoopLoader"""
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize loss history
        loss_history = []
        reconst_err_history = []
        val_loss_history = []
        val_reconst_err_history = []

        # Create loop loader
        loop_loader = LoopLoader(
            dataloader=self.dataloader,
            n_steps=n_steps,
            n_epochs=n_epochs
        )

        # Training loop with progress bar
        pbar = tqdm(loop_loader, total=len(loop_loader))
        for batch in pbar:
            # Move batch to device
            # Unpack batch including metadata
            masked_spec, mask_frames, clean_spec, masked_audio, metadata = batch

            # Move tensors to device
            masked_spec = masked_spec.to(self.device)
            mask_frames = mask_frames.to(self.device)
            clean_spec = clean_spec.to(self.device)

            batch = (masked_spec, mask_frames, clean_spec)
            
            # ==========================================
            # Training step based on training_mode
            # ==========================================
            if self.config.training_mode == "regular":
                reconst_err, objective, log_dict = self.base_step(batch)
            elif self.config.training_mode == "latent":
                reconst_err, objective, log_dict = self.latent_space_nppc_step(batch)
            elif self.config.training_mode == "latent_multi_level":
                reconst_err, objective, log_dict = self.latent_space_nppc_multi_level_step(batch)
            elif self.config.training_mode == "latent_multi_level_v2":
                reconst_err, objective, log_dict = self.latent_space_nppc_multi_level_step_v2(batch)
            else:
                raise ValueError(f"Unknown training_mode: {self.config.training_mode}")

            # Check for NaN/Inf in loss
            if torch.isnan(objective) or torch.isinf(objective):
                print(f"[Step {self.step}] NaN/Inf detected in loss! Skipping batch.")
                print(f"  reconst_err: {reconst_err.mean().item()}")
                print(f"  second_moment_mse: {log_dict['second_moment_mse'].mean().item()}")
                continue

            self.optimizer.zero_grad()
            objective.backward()

            # Apply gradient clipping and log gradient norm
            if self.config.training_mode == "regular":
                grad_norm = torch.nn.utils.clip_grad_norm_(self.nppc_model.parameters(), max_norm=self.config.max_grad_norm)
            elif self.config.training_mode == "latent":
                grad_norm = torch.nn.utils.clip_grad_norm_(self.nppc_latent_model.parameters(), max_norm=self.config.max_grad_norm)
            elif self.config.training_mode in ["latent_multi_level", "latent_multi_level_v2"]:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.nppc_latent_multi_level_model.parameters(), max_norm=self.config.max_grad_norm)

            self.optimizer.step()

            # Store loss
            loss_history.append(objective.item())
            reconst_err_history.append(reconst_err.mean().item())
            
            # Log gradient norm if it's high or at log intervals
            if grad_norm > self.config.max_grad_norm or self.step % self.config.log_interval == 0:
                if self.config.use_wandb:
                    wandb.log({"train/grad_norm": grad_norm.item()}, step=self.step)
                if grad_norm > self.config.max_grad_norm:
                    print(f"[Step {self.step}] High gradient norm detected: {grad_norm:.4f} (clipped to {self.config.max_grad_norm})")

            # Update progress bar
            pbar.set_description(
                f'Loss: {objective.item():.4f}'
            )
            pbar.set_description(
                f'Objective: {objective.item():.4f} | '
                f'Second Moment MSE: {log_dict["second_moment_mse"].mean().item():.4f} | '
                f'Reconstract Error: {reconst_err.mean().item():.4f}'
            )

            # Validation
            # if val_dataloader and self.step % self.config.log_interval == 0:
            #     val_loss, val_reconst_err = self.validate(val_dataloader)
            #     # val_loss, val_reconst_err = self.validate_latent(val_dataloader)
            #     val_loss_history.append(val_loss)
            #     val_reconst_err_history.append(val_reconst_err)
            #     print(f" | Validation objective at Step {self.step}: {val_loss:.4f}")
            #     print(f" | Validation Reconstract Error at Step {self.step}: {val_reconst_err:.4f}")
            #     # Update progress bar with validation metrics
            #     pbar.set_description(
            #         f'Train Obj: {objective.item():.4f} | Val Obj: {val_loss:.4f} | '
            #         f'Train Reconst: {reconst_err.mean().item():.4f} | Val Reconst: {val_reconst_err:.4f}'
            #     )

            self.step += 1

        # Plot loss curve
        fig = self.plot_loss_curve(loss_history, val_loss_history)

        if save_flag:
            # Save final checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_final_{timestamp}.pt"
            )

            # Save metrics including loss history
            self._get_and_save_metrics(checkpoint_dir, log_dict, n_epochs, n_steps, timestamp)
            self.save_checkpoint(final_checkpoint_path)

        # Log everything to wandb at the end
        if self.config.use_wandb:
            wandb.log({
                "train/final_loss": loss_history[-1],
                "train/avg_loss": sum(loss_history) / len(loss_history),
                "loss_curve": wandb.Image(fig)
            })

            # if val_dataloader:
            #     wandb.log({
            #         "val/final_loss": val_loss,
            #         "val/best_loss": min(val_loss_history) if val_loss_history else None
            #     })

            wandb.finish()
        plt.close(fig)

    def plot_loss_curve(self, loss_history, val_loss_history):
        """Plot the training and validation loss curves with both raw and smoothed versions"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot raw losses
        ax1.plot(loss_history, label='Training Loss', alpha=0.5)
        if val_loss_history:
            # Adjust validation points to match their actual steps
            val_steps = [i * self.config.log_interval for i in range(len(val_loss_history))]
            ax1.plot(val_steps, val_loss_history, label='Validation Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Raw Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot smoothed losses
        smoothed_loss = self._smooth_losses(loss_history)
        ax2.plot(smoothed_loss, label='Smoothed Training Loss')
        if val_loss_history:
            ax2.plot(val_steps, val_loss_history, label='Validation Loss')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Loss')
        ax2.set_title('Smoothed Training and Validation Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
        return fig

    def base_step2(self, batch):
        """
        Modified NPPC base step: Projects W_MC onto W_NPPC while preserving the original normalization structure.
        """
        masked_spec, mask, clean_spec = batch
        clean_spec_mag_norm_log, mask, masked_spec_mag_log = utils.preprocess_data(clean_spec, masked_spec, mask)

        # Step 1️⃣: Get NPPC Uncertainty Predictions
        w_mat = self.nppc_model(masked_spec_mag_log, mask)  # [B, n_dirs, F, T]

        # Reshape W_NPPC for PCA comparison
        w_mat_ = w_mat.flatten(2)  # Shape: [B, n_dirs, F*T]
        w_norms = w_mat_.norm(dim=2) + 1e-6  # Compute norms
        w_hat_mat = w_mat_ / w_norms[:, :, None]  # Normalized principal components

        pred_spec_mag_norm_log = self.nppc_model.get_pred_spec_mag_norm(masked_spec_mag_log, mask)

        # now we will get the w_mc -> mc dropout + pca:

        # Step 2️⃣: Compute MC-Dropout + PCA Results
        restoration_model = self.nppc_model.pretrained_restoration_model
        restoration_model.train()
        mc_dropout_after_pca_dict = calculate_unet_baseline(restoration_model, masked_spec_mag_log, mask)
        W_mc = mc_dropout_after_pca_dict['scaled_principal_components']
        # Ensure PCA components are scaled correctly
        # explained_variance = mc_dropout_after_pca_dict['importance_weights']
        # W_mc = W_mc * explained_variance[:, :, None, None] # Scale by sqrt of variance
        restoration_model.eval()
        singular_values = mc_dropout_after_pca_dict['singular_vals']

        # outputs_mc = mc_dropout_inference(self.nppc_model, masked_spec_mag_log, K=50)  # [K, B, F, T]
        # outputs_mc_flat = outputs_mc.flatten(2)  # [K, B, F*T]
        #
        # # Compute Principal Components via PCA
        # W_mc = compute_pca_on_mc_dropout(outputs_mc_flat)  # [B, F*T, K]

        # Compute Norms for W_MC
        w_mc_ = W_mc.flatten(2)
        w_mc_norms = w_mc_.norm(dim=2) + 1e-6
        W_mc_hat = w_mc_ / w_mc_norms[:, :, None]

        # Normalize W_MC
        # W_mc_hat = W_mc / w_mc_norms[:, None]

        # Step 3️⃣: Scale W_NPPC's norms to match W_MC
        # w_norms = w_norms / w_mc_norms  # Scale norms only, like in original code

        # Step 4️⃣: Compute Updated Loss Terms
        ## Projection of W_MC onto W_NPPC
        proj_coeffs = []
        reconst_err_list = []
        second_moment_list = []
        for i in range(w_hat_mat.shape[1]):  # for each direction
            w_i = w_hat_mat[:, i, :]  # [B, F*T] - i-th row of NPPC
            w_mc_i = W_mc_hat[:, i, :]  # [B, F*T] - i-th row of MC

            # Calculate projection coefficient
            proj_coeff = torch.sum(w_i * w_mc_i, dim=1)  # [B]
            proj_coeffs.append(proj_coeff)
            curr_reconst_err = 1 - proj_coeff.pow(2)
            curr_second_moment = (w_norms[:, i].pow(2) - singular_values[:, i].pow(2)).pow(2)
            # curr_second_moment = (w_norms[:,i].pow(2) - proj_coeff.detach().pow(2)).pow(2)

            reconst_err_list.append(curr_reconst_err)
            second_moment_list.append(curr_second_moment)

        # Stack all projections
        reconst_err = torch.stack(reconst_err_list, dim=1).mean(dim=1)
        second_moment_mse = torch.stack(second_moment_list, dim=1).mean(dim=1)
        proj_W_mc_on_W_nppc = torch.stack(proj_coeffs, dim=1)  # [B, 5]
        # # proj_W_mc_on_W_nppc = torch.einsum('bki,bkj->bk', w_hat_mat, W_mc_hat)  # Projection coefficients
        # reconst_err = 1 - proj_W_mc_on_W_nppc.pow(2).mean(dim=1)  # Reconstruction error from projection
        #
        # ## Second Moment MSE (Align Variances)
        # second_moment_mse = (w_norms.pow(2) - proj_W_mc_on_W_nppc.detach().pow(2)).pow(2)

        ## Final Loss
        objective = self._calculate_final_objective(reconst_err, second_moment_mse)

        # Step 5️⃣: Logging for Analysis
        log = {
            'w_mat': w_mat.detach(),
            'w_mc': W_mc.detach(),
            'proj_W_mc_on_W_nppc': proj_W_mc_on_W_nppc.detach(),
            'w_norms': w_norms.detach(),
            'reconst_err': reconst_err.detach(),
            'second_moment_mse': second_moment_mse.detach(),
            'objective': objective.detach()
        }

        return reconst_err, objective, log

    def base_step(self, batch):
        """
        base step function for training the nppc for the inpainting audio task
        Args:
            batch:
    
        Returns:
    
        """
        # firstly we should move the spec into a mag norm log specs:
        # masked_spec, mask, clean_spec = batch
        masked_spec, mask, clean_spec = batch  # ignore masked_spec
        clean_spec_mag_norm_log, mask, masked_spec_mag_log = utils.preprocess_data(clean_spec, masked_spec, mask)
    
        w_mat = self.nppc_model(masked_spec_mag_log, mask)  # [B,n_dirs,F,T]
    
        w_mat_ = w_mat.flatten(2)
        w_norms = w_mat_.norm(dim=2) + 1e-6
        w_hat_mat = w_mat_ / w_norms[:, :, None]
    
        pred_spec_mag_norm_log = self.nppc_model.get_pred_spec_mag_norm(masked_spec_mag_log, mask)
        err = (clean_spec_mag_norm_log - pred_spec_mag_norm_log).flatten(1)  # [B,F*T]
    
        ## Normalizing by the error's norm
        ## -------------------------------
        err_norm = err.norm(dim=1) + 1e-6
        err = err / err_norm[:, None]
        w_norms = w_norms / err_norm[:, None]
    
        ## W hat loss
        ## ----------
        err_proj = torch.einsum('bki,bi->bk', w_hat_mat, err)
        reconst_err = 1 - err_proj.pow(2).sum(dim=1)
        second_moment_mse = (w_norms.pow(2) - err_proj.detach().pow(2)).pow(2)
        # Compute final objective with adaptive weighting
        objective = self._calculate_final_objective(reconst_err, second_moment_mse)
        # Store logs
        log = {
            'w_mat': w_mat.detach(),
            'err_norm': err_norm.detach(),
            'err_proj': err_proj.detach(),  # Keeping the complex projection for logging if needed
            'w_norms': w_norms.detach(),
            'reconst_err': reconst_err.detach(),
            'second_moment_mse': second_moment_mse.detach(),
            'objective': objective.detach()
        }
    
        return reconst_err, objective, log

    # def base_step(self, batch):
    #     """
    #     base step function for training the nppc for the inpainting audio task
    #     Args:
    #         batch:

    #     Returns:

    #     """
    #     # firstly we should move the spec into a mag norm log specs:
    #     # masked_spec, mask, clean_spec = batch
    #     # masked_spec, mask, clean_spec = batch  # ignore masked_spec

    #     x_org = batch[0]
    #     x_distorted = x_org * (1 - self.mask)
    #     broadcasted_mask =  self.mask.view(1, 1, 28, 28).expand(x_org.shape[0], 1, 28, 28)
    #     with torch.no_grad():
    #         x_pred = self.nppc_model.pretrained_restoration_model(x_distorted, self.mask)
    #     # x_distorted_and_pred = torch.cat(
    #     #     (x_distorted, x_pred),
    #     #     dim=1
    #     # )

    #     # clean_spec_mag_norm_log, mask, masked_spec_mag_log = utils.preprocess_data(clean_spec, masked_spec, mask)

    #     # w_mat = self.nppc_model(x_distorted, 1 - broadcasted_mask)  # [B,n_dirs,F,T]
    #     w_mat = self.nppc_model(x_distorted, broadcasted_mask)  # [B,n_dirs,F,T]

    #     w_mat_ = w_mat.flatten(2)
    #     w_norms = w_mat_.norm(dim=2) + 1e-6
    #     w_hat_mat = w_mat_ / w_norms[:, :, None]

    #     err = (x_org - x_pred).flatten(1)  # [B,F*T]

    #     ## Normalizing by the error's norm
    #     ## -------------------------------
    #     err_norm = err.norm(dim=1) + 1e-6
    #     err = err / err_norm[:, None]
    #     w_norms = w_norms / err_norm[:, None]

    #     ## W hat loss
    #     ## ----------
    #     err_proj = torch.einsum('bki,bi->bk', w_hat_mat, err)
    #     reconst_err = 1 - err_proj.pow(2).sum(dim=1)
    #     second_moment_mse = (w_norms.pow(2) - err_proj.detach().pow(2)).pow(2)
    #     # Compute final objective with adaptive weighting
    #     objective = self._calculate_final_objective(reconst_err, second_moment_mse)
    #     # Store logs
    #     log = {
    #         'w_mat': w_mat.detach(),
    #         'err_norm': err_norm.detach(),
    #         'err_proj': err_proj.detach(),  # Keeping the complex projection for logging if needed
    #         'w_norms': w_norms.detach(),
    #         'reconst_err': reconst_err.detach(),
    #         'second_moment_mse': second_moment_mse.detach(),
    #         'objective': objective.detach()
    #     }

    #     return reconst_err, objective, log


    def latent_space_nppc_step(self, batch):
        """
        Perform NPPC in latent space instead of output space.
        Args:
            batch: input batch (masked_spec, mask, clean_spec)

        Returns:
            reconst_err_latent, objective_latent, log
        """
        # Step 1: Preprocessing data
        masked_spec, mask, clean_spec = batch
        clean_spec_norm_log, mask, masked_spec_norm_log = utils.preprocess_data(clean_spec, masked_spec, mask)
        pred_spec_mag_norm_log = self.nppc_model.get_pred_spec_mag_norm(masked_spec_norm_log, mask)


        # Step 2: Encoding into latent space
        with torch.no_grad():
            latent_clean , _ = self.nppc_model.pretrained_restoration_model.net.encoder(clean_spec_norm_log)
            latent_pred , _  = self.nppc_model.pretrained_restoration_model.net.encoder(pred_spec_mag_norm_log)
            latent_distored , _ = self.nppc_model.pretrained_restoration_model.net.encoder(masked_spec_norm_log)

        # Step 3: Latent Error computation
        #latent_err = latent_clean - latent_pred
        latent_err = latent_clean - latent_distored
        latent_err_flat = latent_err.flatten(start_dim=1)  # [B, latent_features]

        # Step 4: Predict latent directions
        # w_latent = self.nppc_latent_model(masked_spec_norm_log, mask)
        masked_with_pred_spec_mag_norm = torch.cat(
            (masked_spec_norm_log, pred_spec_mag_norm_log),
            dim=1
        )


        w_latent, latent_mask = self.nppc_latent_model(masked_with_pred_spec_mag_norm, 1 - mask)
        # Expand latent_mask from (32, 1, 8, 16) to (32, 512, 8, 16) to match latent_err dimensions
        latent_mask = latent_mask.expand(latent_mask.shape[0], latent_err.shape[1], latent_mask.shape[2], latent_mask.shape[3])

        latent_mask_flat = latent_mask.flatten(start_dim=1)  # [B, latent_features]
        # Don't mask the error - keep full error for stable norm computation (matches base_step logic)
        # latent_err_flat = latent_err_flat * latent_mask_flat

        latent_mask_flat_expanded = latent_mask_flat.unsqueeze(1)
        latent_mask_flat_broadcasted = latent_mask_flat_expanded.expand(-1, w_latent.shape[1], -1)

        w_latent_flat = w_latent.flatten(start_dim=2)  # [B, n_dirs, latent_features]

        w_latent_flat = w_latent_flat * latent_mask_flat_broadcasted

        # Step 5: Gram-Schmidt normalization (latent space)
        # still not implemented
        w_latent_flat = gram_schmidt_to_spec_mag(w_latent_flat)
        
        # DEBUG: Check for issues after Gram-Schmidt
        w_max = w_latent_flat.abs().max().item()
        if w_max > 1e3 or torch.isnan(w_latent_flat).any() or torch.isinf(w_latent_flat).any():
            print(f"\n⚠️  ALERT at step {self.step}: After Gram-Schmidt - w_max={w_max:.2e}, has_nan={torch.isnan(w_latent_flat).any()}, has_inf={torch.isinf(w_latent_flat).any()}\n", flush=True)

        w_norms_latent = w_latent_flat.norm(dim=2) + 1e-6
        w_hat_latent = w_latent_flat / w_norms_latent[:,:, None]

        # Step 6: Project latent error
        # Use MUCH larger epsilon to prevent division by small numbers
        raw_err_norm = latent_err_flat.norm(dim=1)
        latent_err_norm = torch.clamp(raw_err_norm, min=0.1) + 1e-3  # Clamp to at least 0.1
        
        # DEBUG: Check error norm
        err_norm_min = latent_err_norm.min().item()
        err_norm_max = latent_err_norm.max().item()
        raw_err_norm_min = raw_err_norm.min().item()
        if raw_err_norm_min < 0.1 or err_norm_max > 1e4:
            print(f"\n⚠️  ALERT at step {self.step}: raw_err_norm_min={raw_err_norm_min:.2e}, clamped to {err_norm_min:.2e}\n", flush=True)
        
        latent_err_normalized = latent_err_flat / latent_err_norm[:, None]
        w_norms_latent_normalized = w_norms_latent / latent_err_norm[:, None]
        
        # CRITICAL: Clamp normalized norms to prevent explosions
        # Reduced from 10.0 to 2.0 because: (2.0² - 0)² = 16, but (10.0² - 0)² = 10,000!
        w_norms_latent_normalized = torch.clamp(w_norms_latent_normalized, max=2.0)
        
        # DEBUG: Check normalized norms (CRITICAL - this is where explosion happens!)
        w_norm_max = w_norms_latent_normalized.abs().max().item()
        w_norm_mean = w_norms_latent_normalized.abs().mean().item()
        if w_norm_max > 2.0 or torch.isnan(w_norms_latent_normalized).any() or torch.isinf(w_norms_latent_normalized).any():
            print(f"\n🚨 CRITICAL ALERT at step {self.step}: w_norms_latent_normalized max={w_norm_max:.2e}, mean={w_norm_mean:.2e}, has_nan={torch.isnan(w_norms_latent_normalized).any()}, has_inf={torch.isinf(w_norms_latent_normalized).any()}\n", flush=True)
            print(f"   latent_err_norm values: min={err_norm_min:.2e}, max={err_norm_max:.2e}, mean={latent_err_norm.mean().item():.2e}\n", flush=True)
            print(f"   w_norms_latent values: min={w_norms_latent.min().item():.2e}, max={w_norms_latent.max().item():.2e}, mean={w_norms_latent.mean().item():.2e}\n", flush=True)

        latent_err_proj = torch.einsum('bki,bi->bk', w_hat_latent, latent_err_normalized)

        # Step 7: Latent reconstruction and variance losses
        reconst_err_latent = 1 - latent_err_proj.pow(2).sum(dim=1)
        second_moment_mse_latent = (w_norms_latent_normalized.pow(2) - latent_err_proj.detach().pow(2)).pow(2)
        
        # CRITICAL: Clamp second_moment_mse to prevent loss explosion
        # With w_norms_normalized clamped to 2.0, max is (2²-0)² = 16, but add safety clamp anyway
        second_moment_mse_latent = torch.clamp(second_moment_mse_latent, max=1.0)
        
        # DEBUG: Check for NaN/Inf in losses BEFORE using them
        if torch.isnan(reconst_err_latent).any() or torch.isinf(reconst_err_latent).any() or \
           torch.isnan(second_moment_mse_latent).any() or torch.isinf(second_moment_mse_latent).any():
            print(f"\n🚨 NaN/Inf DETECTED at step {self.step}! Skipping this batch.\n", flush=True)
            print(f"   reconst_err: has_nan={torch.isnan(reconst_err_latent).any()}, has_inf={torch.isinf(reconst_err_latent).any()}\n", flush=True)
            print(f"   second_moment_mse: has_nan={torch.isnan(second_moment_mse_latent).any()}, has_inf={torch.isinf(second_moment_mse_latent).any()}\n", flush=True)
            # Return a dummy zero loss to skip this batch
            objective_latent = torch.tensor(0.0, device=reconst_err_latent.device, requires_grad=True)
            log = {
                'latent_err_norm': latent_err_norm.detach(),
                'latent_err_proj': torch.zeros_like(latent_err_proj),
                'w_norms': torch.zeros_like(w_norms_latent),
                'reconst_err': torch.zeros_like(reconst_err_latent),
                'second_moment_mse': torch.zeros_like(second_moment_mse_latent),
                'objective': objective_latent.detach(),
                'w_latent': w_latent.detach()
            }
            return reconst_err_latent.detach(), objective_latent, log
        
        # DEBUG: Check final losses (now clamped, so should never fire)
        if reconst_err_latent.abs().max() > 2 or second_moment_mse_latent.abs().max() > 1.5:
            print(f"\n🚨 LOSS ALERT at step {self.step}: reconst_err max={reconst_err_latent.abs().max():.2e}, second_moment_mse max={second_moment_mse_latent.abs().max():.2e}\n", flush=True)
            print(f"   w_norms_latent_normalized: min={w_norms_latent_normalized.min().item():.2e}, max={w_norms_latent_normalized.max().item():.2e}, mean={w_norms_latent_normalized.mean().item():.2e}\n", flush=True)

        # Step 8: Final combined loss
        objective_latent = self._calculate_final_objective(
            reconst_err_latent,
            second_moment_mse_latent
        )

        # Logging dictionary
        log = {
            'latent_err_norm': latent_err_norm.detach(),
            'latent_err_proj': latent_err_proj.detach(),
            'w_norms': w_norms_latent.detach(),
            'reconst_err': reconst_err_latent.detach(),
            'second_moment_mse': second_moment_mse_latent.detach(),
            'objective': objective_latent.detach(),
            'w_latent': w_latent.detach()
        }

        return reconst_err_latent, objective_latent, log


    def latent_space_nppc_with_decoder_step(self, batch):
        """
        Perform NPPC in latent space but compute losses in spectrogram space.
        
        Key idea: 
        1. Predict W matrix in latent space (like latent_space_nppc_step)
        2. Decode W directions back to spectrogram space using decoder
        3. Compute reconstruction error in spectrogram space (like base_step)
        
        This aligns the training loss with the actual task (spectrogram reconstruction)
        while still leveraging the latent space structure.
        
        Args:
            batch: (masked_spec, mask, clean_spec)
            
        Returns:
            reconst_err, objective, log
        """
        # Step 1: Preprocessing data
        masked_spec, mask, clean_spec = batch
        clean_spec_norm_log, mask, masked_spec_norm_log = utils.preprocess_data(clean_spec, masked_spec, mask)
        pred_spec_mag_norm_log = self.nppc_model.get_pred_spec_mag_norm(masked_spec_norm_log, 1 - mask)
        
        # Step 2: Encoding into latent space (no grad - frozen encoder)
        with torch.no_grad():
            latent_clean, _ = self.nppc_model.pretrained_restoration_model.net.encoder(clean_spec_norm_log)
            latent_pred, _ = self.nppc_model.pretrained_restoration_model.net.encoder(pred_spec_mag_norm_log)
            _, skip_connections_distorted = self.nppc_model.pretrained_restoration_model.net.encoder(masked_spec_norm_log)
        
        # Step 3: Predict latent W directions using latent encoder
        masked_with_pred_spec_mag_norm = torch.cat(
            (masked_spec_norm_log, pred_spec_mag_norm_log),
            dim=1
        )
        
        w_latent, latent_mask = self.nppc_latent_model(masked_with_pred_spec_mag_norm, 1 - mask)
        # w_latent shape: [B, n_dirs, C, H, W]
        # latent_mask shape: [B, 1, H, W]
        
        # Apply latent masking and Gram-Schmidt IN LATENT SPACE (like validator line 312-316)
        B, n_dirs, C, latent_H, latent_W = w_latent.shape
        
        # Expand latent mask to match channels
        latent_mask = latent_mask.expand(latent_mask.shape[0], C, latent_mask.shape[2], latent_mask.shape[3])  # [B, C, H, W]
        latent_mask_flat = latent_mask.flatten(start_dim=1)  # [B, C*H*W]
        latent_mask_flat_expanded = latent_mask_flat.unsqueeze(1)  # [B, 1, C*H*W]
        latent_mask_flat_broadcasted = latent_mask_flat_expanded.expand(-1, n_dirs, -1)  # [B, n_dirs, C*H*W]
        
        # Flatten w_latent and apply mask
        w_latent_flat = w_latent.flatten(start_dim=2)  # [B, n_dirs, C*H*W]
        w_latent_flat = w_latent_flat * latent_mask_flat_broadcasted
        
        # Apply Gram-Schmidt in latent space BEFORE decoding (like validator line 315)
        w_latent_flat = gram_schmidt_to_spec_mag(w_latent_flat)
        w_latent = w_latent_flat.view(B, n_dirs, C, latent_H, latent_W)
        
        # Step 4: Decode each W direction back to spectrogram space
        # IMPORTANT: Decoder expects full latent representations, not direction vectors
        # So we decode: direction = decoder(latent + w_direction) - decoder(latent)
        
        decoder = self.nppc_model.pretrained_restoration_model.net.decoder
        
        # Compute baseline: masked_spec + decoder(latent_pred) * (1-mask)
        # This is what alpha=0 gives in the validator (line 346)
        pred_decoded_raw = decoder(latent_pred, skip_connections_distorted)  # [B, 1, F, T]
        pred_spec_baseline = masked_spec_norm_log + pred_decoded_raw * (1 - mask)  # [B, 1, F, T]
        
        # Now decode each direction by adding it to latent_pred
        w_spec_list = []
        for dir_idx in range(n_dirs):
            w_latent_i = w_latent[:, dir_idx, :, :, :]  # [B, C, latent_H, latent_W]
            
            # Old approach (decoding direction directly - doesn't work well):
            # w_spec_i = decoder(w_latent_i, skip_connections_distorted)  # [B, 1, F, T]
            
            # Decode latent_pred + w_direction WITH residual (like validator line 346!)
            latent_with_direction = latent_pred + w_latent_i
            decoded_with_direction = decoder(latent_with_direction, skip_connections_distorted)  # [B, 1, F, T]
            spec_with_direction = masked_spec_norm_log + decoded_with_direction * (1 - mask)  # Add residual!
            
            # The direction in spec space is the difference from baseline
            w_spec_i = spec_with_direction - pred_spec_baseline  # [B, 1, F, T]
            w_spec_list.append(w_spec_i)
        
        w_spec = torch.stack(w_spec_list, dim=1)  # [B, n_dirs, 1, F, T]
        
        # NOTE: w_spec is already masked from the residual connection above!
        # spec = masked_spec + decoder(...) * (1-mask) means w_spec inherently has (1-mask) applied
        # So we DON'T need to mask again here (would be double masking)
        
        # OLD: Mask again (caused double masking)
        # mask_broadcasted = (1 - mask).expand(-1, n_dirs, -1, -1)
        # w_spec = w_spec * mask_broadcasted.unsqueeze(2)
        
        # Apply Gram-Schmidt in spectrogram space to ensure orthogonality
        # Even though we did it in latent space, the decoder is non-linear and destroys orthogonality
        # This prevents negative reconstruction error (sum of squared projections > 1)
        w_spec = gram_schmidt_to_spec_mag(w_spec)
        
        # Step 5: Now compute loss in spectrogram space (like base_step)
        w_spec_flat = w_spec.flatten(2)  # [B, n_dirs, F*T]
        w_norms = w_spec_flat.norm(dim=2) + 1e-6
        w_hat_spec = w_spec_flat / w_norms[:, :, None]
        
        # Compute spectrogram error
        err = (clean_spec_norm_log - pred_spec_mag_norm_log).flatten(1)  # [B, F*T]
        
        # Normalize by error's norm
        err_norm = err.norm(dim=1) + 1e-6
        err = err / err_norm[:, None]
        w_norms = w_norms / err_norm[:, None]
        
        # Compute projection and losses
        err_proj = torch.einsum('bki,bi->bk', w_hat_spec, err)
        reconst_err = 1 - err_proj.pow(2).sum(dim=1)
        second_moment_mse = (w_norms.pow(2) - err_proj.detach().pow(2)).pow(2)
        
        # Final objective
        objective = self._calculate_final_objective(reconst_err, second_moment_mse)
        
        # Logging dictionary
        log = {
            'w_latent': w_latent.detach(),
            'w_spec': w_spec.detach(),
            'err_norm': err_norm.detach(),
            'err_proj': err_proj.detach(),
            'w_norms': w_norms.detach(),
            'reconst_err': reconst_err.detach(),
            'second_moment_mse': second_moment_mse.detach(),
            'objective': objective.detach()
        }
        
        return reconst_err, objective, log


    def latent_space_nppc_multi_level_step(self, batch):
        """
        Multi-level NPPC training: Predict W directions for ALL levels (bottleneck + skip connections).
        Loss is computed in LATENT SPACE (matching latent_space_nppc_step approach).
        
        Key idea:
        1. Predict W for bottleneck AND all skip connections
        2. Compute error at each level: err = clean - distorted (in latent space)
        3. Compute loss by projecting concatenated error onto concatenated W
        
        Args:
            batch: (masked_spec, mask, clean_spec)
            
        Returns:
            reconst_err, objective, log
        """
        # Step 1: Preprocessing data
        masked_spec, mask, clean_spec = batch
        clean_spec_norm_log, mask, masked_spec_norm_log = utils.preprocess_data(clean_spec, masked_spec, mask)
        pred_spec_mag_norm_log = self.nppc_model.get_pred_spec_mag_norm(masked_spec_norm_log, mask)
        
        # Step 2: Get encoder outputs for clean, pred, and distorted spectrograms
        with torch.no_grad():
            latent_clean, skip_connections_clean = self.nppc_model.pretrained_restoration_model.net.encoder(clean_spec_norm_log)
            latent_pred, skip_connections_pred = self.nppc_model.pretrained_restoration_model.net.encoder(pred_spec_mag_norm_log)
            latent_distorted, skip_connections_distorted = self.nppc_model.pretrained_restoration_model.net.encoder(masked_spec_norm_log)
        
        # Step 3: Predict multi-level W directions
        masked_with_pred = torch.cat((masked_spec_norm_log, pred_spec_mag_norm_log), dim=1)
        
        # Get W directions for all levels
        # Note: mask from preprocess_data already has shape [B, 1, F, T] or similar
        # Pass inverted mask (1 - mask) to the model
        w_bottleneck, w_skips, level_masks = self.nppc_latent_multi_level_model(
            masked_with_pred, 1 - mask
        )
        # w_skips = [w_skip4, w_skip3, w_skip2, w_skip1]
        
        n_dirs = self.nppc_latent_multi_level_model.n_dirs
        B = masked_spec_norm_log.shape[0]
        
        # Step 4: Compute error at each level (clean - distorted) and flatten W
        # We'll concatenate all W directions AND all errors, then compute loss
        
        w_flat_list = []
        err_flat_list = []
        level_sizes = []
        
        # Process bottleneck
        if w_bottleneck is not None:
            mask_b = level_masks['bottleneck']
            B_m, _, H_m, W_m = mask_b.shape
            mask_b_full = mask_b.expand(B_m, w_bottleneck.shape[2], H_m, W_m)  # [B, C, H, W]
            mask_b_full_w = mask_b_full.unsqueeze(1).expand(-1, n_dirs, -1, -1, -1)  # [B, n_dirs, C, H, W]
            
            # Flatten and mask W
            w_b_flat = w_bottleneck.flatten(start_dim=2)  # [B, n_dirs, C*H*W]
            mask_b_flat_w = mask_b_full_w.flatten(start_dim=2)  # [B, n_dirs, C*H*W]
            w_b_flat = w_b_flat * mask_b_flat_w
            w_flat_list.append(w_b_flat)
            
            # Compute and flatten error (clean - distorted)
            err_b = latent_clean - latent_distorted  # [B, C, H, W]
            err_b_flat = err_b.flatten(start_dim=1)  # [B, C*H*W]
            # Don't mask error (matching latent_space_nppc_step line 574-575)
            err_flat_list.append(err_b_flat)
            
            level_sizes.append(('bottleneck', w_b_flat.shape[2], w_bottleneck.shape[2:]))
        
        # Process skip connections
        skip_names = ['skip4', 'skip3', 'skip2', 'skip1']
        for i, (w_skip, name) in enumerate(zip(w_skips, skip_names)):
            if w_skip is not None:
                mask_s = level_masks[name]
                B_m, _, H_m, W_m = mask_s.shape
                mask_s_full = mask_s.expand(B_m, w_skip.shape[2], H_m, W_m)
                mask_s_full_w = mask_s_full.unsqueeze(1).expand(-1, n_dirs, -1, -1, -1)
                
                # Flatten and mask W
                w_s_flat = w_skip.flatten(start_dim=2)
                mask_s_flat_w = mask_s_full_w.flatten(start_dim=2)
                w_s_flat = w_s_flat * mask_s_flat_w
                w_flat_list.append(w_s_flat)
                
                # Compute and flatten error (clean - distorted) for this skip level
                err_s = skip_connections_clean[i] - skip_connections_distorted[i]  # [B, C, H, W]
                err_s_flat = err_s.flatten(start_dim=1)  # [B, C*H*W]
                err_flat_list.append(err_s_flat)
                
                level_sizes.append((name, w_s_flat.shape[2], w_skip.shape[2:]))
        
        # Step 5: Concatenate all levels and apply Gram-Schmidt to W
        w_all_flat = torch.cat(w_flat_list, dim=2)  # [B, n_dirs, total_features]
        w_all_flat = gram_schmidt_to_spec_mag(w_all_flat)
        
        # Concatenate all errors
        err_all_flat = torch.cat(err_flat_list, dim=1)  # [B, total_features]
        
        # Step 6: Compute loss in latent space (matching latent_space_nppc_step)
        w_norms = w_all_flat.norm(dim=2) + 1e-6
        w_hat = w_all_flat / w_norms[:, :, None]
        
        # Normalize error
        raw_err_norm = err_all_flat.norm(dim=1)
        err_norm = torch.clamp(raw_err_norm, min=0.1) + 1e-3  # Clamp like latent_space_nppc_step
        err_normalized = err_all_flat / err_norm[:, None]
        w_norms_normalized = w_norms / err_norm[:, None]
        
        # Clamp normalized norms to prevent explosions (matching latent_space_nppc_step)
        w_norms_normalized = torch.clamp(w_norms_normalized, max=2.0)
        
        # Compute projection
        err_proj = torch.einsum('bki,bi->bk', w_hat, err_normalized)
        
        # Compute losses
        reconst_err = 1 - err_proj.pow(2).sum(dim=1)
        second_moment_mse = (w_norms_normalized.pow(2) - err_proj.detach().pow(2)).pow(2)
        
        # Clamp second_moment_mse (matching latent_space_nppc_step)
        second_moment_mse = torch.clamp(second_moment_mse, max=1.0)
        
        # Check for NaN/Inf
        if torch.isnan(reconst_err).any() or torch.isinf(reconst_err).any() or \
           torch.isnan(second_moment_mse).any() or torch.isinf(second_moment_mse).any():
            print(f"\n🚨 NaN/Inf DETECTED in multi-level step at step {self.step}! Skipping batch.\n", flush=True)
            objective = torch.tensor(0.0, device=reconst_err.device, requires_grad=True)
            log = {
                'w_all_flat': w_all_flat.detach(),
                'err_norm': err_norm.detach(),
                'err_proj': torch.zeros_like(err_proj),
                'w_norms': torch.zeros_like(w_norms),
                'reconst_err': torch.zeros_like(reconst_err),
                'second_moment_mse': torch.zeros_like(second_moment_mse),
                'objective': objective.detach()
            }
            return reconst_err.detach(), objective, log
        
        # Final objective
        objective = self._calculate_final_objective(reconst_err, second_moment_mse)
        
        # Logging
        log = {
            'w_all_flat': w_all_flat.detach(),
            'err_norm': err_norm.detach(),
            'err_proj': err_proj.detach(),
            'w_norms': w_norms.detach(),
            'reconst_err': reconst_err.detach(),
            'second_moment_mse': second_moment_mse.detach(),
            'objective': objective.detach()
        }
        
        return reconst_err, objective, log

    def latent_space_nppc_multi_level_step_v2(self, batch):
        """
        Multi-level NPPC training V2: Per-layer Gram-Schmidt and loss.
        
        DIFFERENCE from v1:
        - V1: Concatenate all W, apply GLOBAL Gram-Schmidt, compute GLOBAL loss
        - V2: Apply Gram-Schmidt PER LAYER, compute loss PER LAYER, sum losses
        
        This allows each layer to have its own orthogonal basis, potentially
        learning different variations at different resolutions.
        """
        # Step 1: Preprocessing data (same as v1)
        masked_spec, mask, clean_spec = batch
        clean_spec_norm_log, mask, masked_spec_norm_log = utils.preprocess_data(clean_spec, masked_spec, mask)
        pred_spec_mag_norm_log = self.nppc_model.get_pred_spec_mag_norm(masked_spec_norm_log, mask)
        
        # Step 2: Get encoder outputs for clean and distorted spectrograms
        with torch.no_grad():
            latent_clean, skip_connections_clean = self.nppc_model.pretrained_restoration_model.net.encoder(clean_spec_norm_log)
            latent_distorted, skip_connections_distorted = self.nppc_model.pretrained_restoration_model.net.encoder(masked_spec_norm_log)
        
        # Step 3: Predict multi-level W directions
        masked_with_pred = torch.cat((masked_spec_norm_log, pred_spec_mag_norm_log), dim=1)
        
        w_bottleneck, w_skips, level_masks = self.nppc_latent_multi_level_model(
            masked_with_pred, 1 - mask
        )
        
        n_dirs = self.nppc_latent_multi_level_model.n_dirs
        B = masked_spec_norm_log.shape[0]
        
        # Step 4: Compute loss PER LAYER (not global!)
        # Each layer gets its own Gram-Schmidt and loss computation
        
        total_reconst_err = torch.zeros(B, device=masked_spec_norm_log.device)
        total_second_moment_mse = torch.zeros(B, n_dirs, device=masked_spec_norm_log.device)
        layer_losses = {}
        layer_projs = {}
        
        # Prepare layers data: (name, w, err, mask)
        layers_data = []
        
        # Bottleneck
        if w_bottleneck is not None:
            err_b = latent_clean - latent_distorted
            layers_data.append(('bottleneck', w_bottleneck, err_b, level_masks['bottleneck']))
        
        # Skip connections
        skip_names = ['skip4', 'skip3', 'skip2', 'skip1']
        for i, (w_skip, name) in enumerate(zip(w_skips, skip_names)):
            if w_skip is not None:
                err_s = skip_connections_clean[i] - skip_connections_distorted[i]
                layers_data.append((name, w_skip, err_s, level_masks[name]))
        
        # Process each layer independently
        for layer_name, w_layer, err_layer, mask_layer in layers_data:
            # Flatten W and apply mask
            B_l, _, H_l, W_l = mask_layer.shape
            C_l = w_layer.shape[2]
            
            mask_full = mask_layer.expand(B_l, C_l, H_l, W_l)
            mask_full_w = mask_full.unsqueeze(1).expand(-1, n_dirs, -1, -1, -1)
            
            w_flat = w_layer.flatten(start_dim=2)  # [B, n_dirs, C*H*W]
            mask_flat_w = mask_full_w.flatten(start_dim=2)
            w_flat = w_flat * mask_flat_w
            
            # Flatten error
            err_flat = err_layer.flatten(start_dim=1)  # [B, C*H*W]
            
            # Apply Gram-Schmidt to THIS LAYER ONLY
            w_flat = gram_schmidt_to_spec_mag(w_flat)
            
            # Normalize W
            w_norms = w_flat.norm(dim=2) + 1e-6
            w_hat = w_flat / w_norms[:, :, None]
            
            # Normalize error
            raw_err_norm = err_flat.norm(dim=1)
            err_norm = torch.clamp(raw_err_norm, min=0.1) + 1e-3
            err_normalized = err_flat / err_norm[:, None]
            w_norms_normalized = w_norms / err_norm[:, None]
            w_norms_normalized = torch.clamp(w_norms_normalized, max=2.0)
            
            # Project error onto W directions
            err_proj = torch.einsum('bki,bi->bk', w_hat, err_normalized)
            
            # Compute losses for this layer
            reconst_err_layer = 1 - err_proj.pow(2).sum(dim=1)  # [B]
            second_moment_mse_layer = (w_norms_normalized.pow(2) - err_proj.detach().pow(2)).pow(2)
            second_moment_mse_layer = torch.clamp(second_moment_mse_layer, max=1.0)
            
            # Accumulate losses
            total_reconst_err = total_reconst_err + reconst_err_layer
            total_second_moment_mse = total_second_moment_mse + second_moment_mse_layer
            
            # Store for logging
            layer_losses[layer_name] = reconst_err_layer.mean().item()
            layer_projs[layer_name] = err_proj.detach()
        
        # AVERAGE across layers (not sum!) to keep gradients stable
        n_layers = len(layers_data)
        if n_layers > 0:
            total_reconst_err = total_reconst_err / n_layers
            total_second_moment_mse = total_second_moment_mse / n_layers
        
        # Check for NaN/Inf
        if torch.isnan(total_reconst_err).any() or torch.isinf(total_reconst_err).any() or \
           torch.isnan(total_second_moment_mse).any() or torch.isinf(total_second_moment_mse).any():
            print(f"\n🚨 NaN/Inf DETECTED in multi-level v2 step at step {self.step}! Skipping batch.\n", flush=True)
            objective = torch.tensor(0.0, device=total_reconst_err.device, requires_grad=True)
            log = {
                'reconst_err': total_reconst_err.detach(),
                'second_moment_mse': total_second_moment_mse.detach(),
                'objective': objective.detach(),
                'layer_losses': layer_losses
            }
            return total_reconst_err.detach(), objective, log
        
        # Final objective (average of all layer losses)
        objective = self._calculate_final_objective(total_reconst_err, total_second_moment_mse)
        
        # Logging
        log = {
            'reconst_err': total_reconst_err.detach(),
            'second_moment_mse': total_second_moment_mse.detach(),
            'objective': objective.detach(),
            'layer_losses': layer_losses,
            'layer_projs': layer_projs
        }
        
        return total_reconst_err, objective, log

    def latent_space_nppc_mnist_step(self, batch):
        """
        Perform NPPC in latent space instead of output space.
        Args:
            batch: input batch (masked_spec, mask, clean_spec)

        Returns:
            reconst_err_latent, objective_latent, log
        """
        # Step 1: Preprocessing data
        # masked_spec, mask, clean_spec = batch
        # clean_spec_norm_log, mask, masked_spec_norm_log = utils.preprocess_data(clean_spec, masked_spec, mask)
        # pred_spec_mag_norm_log = self.nppc_model.get_pred_spec_mag_norm(masked_spec_norm_log, mask)


        x_org = batch[0]
        x_distorted = x_org * (1 - self.mask)
        broadcasted_mask =  self.mask.view(1, 1, 28, 28).expand(x_org.shape[0], 1, 28, 28)
        # x_predict = self.nppc_model.get_pred_spec_mag_norm(x_distorted, 1 - broadcasted_mask)
        self.nppc_model.pretrained_restoration_model.eval() # might not be necessary
        # Step 2: Encoding into latent space
        with torch.no_grad():
            x_predict = self.nppc_model.pretrained_restoration_model(x_distorted, self.mask)
            latent_clean , _ = self.nppc_model.pretrained_restoration_model.net.encoder(x_org)
            # latent_pred , _  = self.nppc_model.pretrained_restoration_model.net.encoder(x_distorted)
            # instead passing distorted at encoder of restoration model, let's passed the predict !
            latent_pred , _ = self.nppc_model.pretrained_restoration_model.net.encoder(x_predict)

            # x_predict = self.nppc_model.pretrained_restoration_model(x_distorted, 1 - broadcasted_mask)

        # Step 3: Latent Error computation
        latent_err = latent_clean - latent_pred
        latent_err_flat = latent_err.flatten(start_dim=1)  # [B, latent_features]

        # Step 4: Predict latent directions
        # w_latent = self.nppc_latent_model(masked_spec_norm_log, mask)
        masked_with_pred_spec_mag_norm = torch.cat(
            (x_distorted, x_predict),
            dim=1
        )

        w_latent, latent_mask = self.nppc_latent_model(masked_with_pred_spec_mag_norm, broadcasted_mask)
        latent_mask_flat = latent_mask.flatten(start_dim=1)  # [B, latent_features]
        # print(latent_mask_flat)

        latent_err_flat = latent_err_flat * latent_mask_flat

        latent_mask_flat_expanded = latent_mask_flat.unsqueeze(1)
        latent_mask_flat_broadcasted = latent_mask_flat_expanded.expand(-1, w_latent.shape[1], -1)

        w_latent_flat = w_latent.flatten(start_dim=2)  # [B, n_dirs, latent_features]

        w_latent_flat = w_latent_flat * latent_mask_flat_broadcasted

        # Step 5: Gram-Schmidt normalization (latent space)
        # still not implemented
        w_latent_flat = gram_schmidt_to_spec_mag(w_latent_flat)

        w_norms_latent = w_latent_flat.norm(dim=2) + 1e-6
        w_hat_latent = w_latent_flat / w_norms_latent[:,:, None]

        # Step 6: Project latent error
        latent_err_norm = latent_err_flat.norm(dim=1) + 1e-6
        latent_err_normalized = latent_err_flat / latent_err_norm[:, None]
        w_norms_latent_normalized = w_norms_latent / latent_err_norm[:, None]

        latent_err_proj = torch.einsum('bki,bi->bk', w_hat_latent, latent_err_normalized)

        # Step 7: Latent reconstruction and variance losses
        reconst_err_latent = 1 - latent_err_proj.pow(2).sum(dim=1)
        second_moment_mse_latent = (w_norms_latent_normalized.pow(2) - latent_err_proj.detach().pow(2)).pow(2)

        # Step 8: Final combined loss
        objective_latent = self._calculate_final_objective(
            reconst_err_latent,
            second_moment_mse_latent
        )

        # Logging dictionary
        log = {
            'latent_err_norm': latent_err_norm.detach(),
            'latent_err_proj': latent_err_proj.detach(),
            'w_norms': w_norms_latent.detach(),
            'reconst_err': reconst_err_latent.detach(),
            'second_moment_mse': second_moment_mse_latent.detach(),
            'objective': objective_latent.detach(),
            'w_latent': w_latent.detach()
        }

        return reconst_err_latent, objective_latent, log


    def save_checkpoint(self, checkpoint_path):
        """
        Save model checkpoint including model state, optimizer state, and training info

        Args:
            checkpoint_path: Path to save checkpoint
        """
        # Save the appropriate model based on training_mode
        if self.config.training_mode == "regular":
            model_state = self.nppc_model.state_dict()
        elif self.config.training_mode == "latent":
            model_state = self.nppc_latent_model.state_dict()
        elif self.config.training_mode in ["latent_multi_level", "latent_multi_level_v2"]:
            model_state = self.nppc_latent_multi_level_model.state_dict()
        else:
            raise ValueError(f"Unknown training_mode: {self.config.training_mode}")
        
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'training_mode': self.config.training_mode,
        }

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        if self.config.use_wandb:
            # Save checkpoint as artifact
            artifact = wandb.Artifact(
                name=self.config.wandb_artifact_name,
                type='model',
                description='Collection of nppc inpainting model checkpoints'
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

    def _get_and_save_metrics(self, checkpoint_dir, log_dict, n_epochs, n_steps, timestamp):
        """Save training metrics to JSON file"""
        final_metrics = {
            'timestamp': timestamp,
            'total_steps': self.step,
            'final_loss': log_dict['objective'].item(),
            'training_config': {
                'n_steps': n_steps,
                'n_epochs': n_epochs,
                'learning_rate': self.config.optimizer_configuration.args.get('lr'),
                'device': self.config.device,
                'batch_size': self.config.dataloader_configuration.batch_size,
                'audio_len': self.config.data_configuration.sub_sample_length_seconds,
                'missing_length_seconds': self.config.data_configuration.missing_length_seconds,
                'missing_start_seconds': self.config.data_configuration.missing_start_seconds,
                'length_audio_seconds': self.config.data_configuration.sub_sample_length_seconds,
                'nfft': self.config.data_configuration.stft_configuration.nfft,
                'n_dirs': self.config.nppc_model_configuration.audio_pc_wrapper_configuration.n_dirs
            }
        }
        metrics_path = os.path.join(
            checkpoint_dir,
            f"metrics_final_{timestamp}.json"
        )
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)

        # Log to wandb
        if self.config.use_wandb:
            # Log metrics
            wandb.log({
                "final_metrics/total_steps": final_metrics['total_steps'],
                "final_metrics/final_loss": final_metrics['final_loss'],
                "config/learning_rate": final_metrics['training_config']['learning_rate'],
                "config/batch_size": final_metrics['training_config']['batch_size'],
                "config/audio_len": final_metrics['training_config']['audio_len'],
                "config/missing_length": final_metrics['training_config']['missing_length_seconds'],
                "config/nfft": final_metrics['training_config']['nfft'],
                "config/n_dirs": final_metrics['training_config']['n_dirs']
            })

            # Save metrics file as artifact
            metrics_artifact = wandb.Artifact(
                name=f'metrics-{wandb.run.id}',
                type='metrics',
                description=f'Training metrics at step {self.step}'
            )
            metrics_artifact.add_file(metrics_path)
            wandb.log_artifact(metrics_artifact)

    def _calculate_final_objective(self, reconst_err, second_moment_mse):
        second_moment_loss_lambda = -1 + 2 * self.step / self.config.second_moment_loss_grace
        second_moment_loss_lambda = max(min(second_moment_loss_lambda, 1), 1e-6)
        # second_moment_loss_lambda = 1

        second_moment_loss_lambda *= self.config.second_moment_loss_lambda
        objective = reconst_err.mean() + second_moment_loss_lambda * second_moment_mse.mean()
        return objective

    def validate(self, val_dataloader):
        """Validation loop to compute loss on the validation set"""
        self.nppc_model.eval()
        val_losses = []
        val_reconst_err = []
        with torch.no_grad():
            for batch in val_dataloader:
                # Unpack batch including metadata
                masked_spec, mask_frames, clean_spec, masked_audio, metadata = batch

                # Move tensors to device
                masked_spec = masked_spec.to(self.device)
                mask_frames = mask_frames.to(self.device)
                clean_spec = clean_spec.to(self.device)

                batch_tensors = (masked_spec, mask_frames, clean_spec)

                # Get loss for this batch
                reconst_err, objective, _ = self.base_step(batch_tensors)
                val_losses.append(objective.item())
                val_reconst_err.append(reconst_err.mean().item())  # Calculate average validation loss
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_reconst_err = sum(val_reconst_err) / len(val_reconst_err)
        self.nppc_model.train()  # Set back to training mode
        return avg_val_loss, avg_val_reconst_err

    def validate_latent(self, val_dataloader):
        """Validation loop for latent space NPPC training"""
        self.nppc_latent_model.eval()
        val_losses = []
        val_reconst_err = []
        with torch.no_grad():
            for batch in val_dataloader:
                # Unpack batch including metadata
                masked_spec, mask_frames, clean_spec, masked_audio, metadata = batch

                # Move tensors to device
                masked_spec = masked_spec.to(self.device)
                mask_frames = mask_frames.to(self.device)
                clean_spec = clean_spec.to(self.device)

                batch_tensors = (masked_spec, mask_frames, clean_spec)

                # Get loss for this batch using latent space step
                reconst_err, objective, _ = self.latent_space_nppc_step(batch_tensors)
                val_losses.append(objective.item())
                val_reconst_err.append(reconst_err.mean().item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_reconst_err = sum(val_reconst_err) / len(val_reconst_err)
        self.nppc_latent_model.train()  # Set back to training mode
        return avg_val_loss, avg_val_reconst_err

    def _smooth_losses(self, losses, window_size=100):
        """
        Smooth losses using moving average
        Args:
            losses: List of loss values
            window_size: Size of the moving average window
        Returns:
            Smoothed loss values
        """
        smoothed = []
        for i in range(len(losses)):
            start_idx = max(0, i - window_size + 1)
            smoothed.append(sum(losses[start_idx:(i + 1)]) / (i - start_idx + 1))
        return smoothed


