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

import utils
from nppc_audio.trainer import NPPCAudioTrainer
from utils import OptimizerConfig, DataLoaderConfig , calculate_unet_baseline


class NPPCAudioInpaintingTrainerConfig(pydantic.BaseModel):
    nppc_model_configuration: NPPCModelConfig
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
        self.device = self.config.device
        # create data loader:
        dataset = AudioInpaintingDataset(config.data_configuration)

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

        # Initialize optimizer
        optimizer_class = getattr(optim, config.optimizer_configuration.type)
        self.optimizer = optimizer_class(
            self.nppc_model.parameters(),
            **config.optimizer_configuration.args,
            # weight_decay=1e-4
        )

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

            # Forward and backward pass
            reconst_err, objective, log_dict = self.base_step(batch)
            self.optimizer.zero_grad()
            objective.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.nppc_model.parameters(), max_norm=self.config.max_grad_norm)

            self.optimizer.step()

            # Store loss
            loss_history.append(objective.item())
            reconst_err_history.append(reconst_err.mean().item())

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
            if val_dataloader and self.step % self.config.log_interval == 0:
                val_loss, val_reconst_err = self.validate(val_dataloader)
                val_loss_history.append(val_loss)
                val_reconst_err_history.append(val_reconst_err)
                print(f" | Validation objective at Step {self.step}: {val_loss:.4f}")
                print(f" | Validation Reconstract Error at Step {self.step}: {val_reconst_err:.4f}")

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

            if val_dataloader:
                wandb.log({
                    "val/final_loss": val_loss,
                    "val/best_loss": min(val_loss_history) if val_loss_history else None
                })

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
            curr_second_moment = (w_norms[:,i].pow(2) - singular_values[:,i].pow(2)).pow(2)
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
    #     base step function for training the nppc for the inpainting audio task with MC Dropout
    #     """
    #     # Preprocess data as before
    #     masked_spec, mask, clean_spec = batch
    #     clean_spec_mag_norm_log, mask, masked_spec_mag_log = utils.preprocess_data(clean_spec, masked_spec, mask)
    #
    #     # Generate w_mat once (without changing dropout state)
    #     self.nppc_model.pretrained_restoration_model.eval()
    #
    #     w_mat = self.nppc_model(masked_spec_mag_log, mask)  # [B,n_dirs,F,T]
    #
    #     # Enable dropout for MC sampling
    #     self.nppc_model.pretrained_restoration_model.train()  # Ensure dropout is active
    #
    #     # Collect multiple predictions using MC Dropout
    #     n_mc_samples = 10
    #     pred_specs = []
    #     for _ in range(n_mc_samples):
    #         pred_spec = self.nppc_model.get_pred_spec_mag_norm(masked_spec_mag_log, mask)
    #         pred_specs.append(pred_spec)
    #
    #     # Process w_mat as before
    #     w_mat_ = w_mat.flatten(2)
    #     w_norms = w_mat_.norm(dim=2) + 1e-6
    #     w_hat_mat = w_mat_ / w_norms[:, :, None]
    #
    #     # Calculate reconstruction errors for all MC samples
    #     reconst_errs = []
    #     for pred_spec in pred_specs:
    #         err = (clean_spec_mag_norm_log - pred_spec).flatten(1)  # [B,F*T]
    #
    #         # Normalizing by the error's norm
    #         err_norm = err.norm(dim=1) + 1e-6
    #         err = err / err_norm[:, None]
    #         w_norms_scaled = w_norms / err_norm[:, None]
    #
    #         # W hat loss
    #         err_proj = torch.einsum('bki,bi->bk', w_hat_mat, err)
    #         reconst_err = 1 - err_proj.pow(2).sum(dim=1)
    #         reconst_errs.append(reconst_err)
    #
    #     # Combine reconstruction errors from all MC samples
    #     reconst_err_combined = torch.stack(reconst_errs).mean(dim=0)
    #
    #     # Second moment loss remains the same
    #     second_moment_mse = (w_norms.pow(2) - err_proj.detach().pow(2)).pow(2)
    #
    #     # Compute final objective
    #     objective = self._calculate_final_objective(reconst_err_combined, second_moment_mse)
    #
    #     # Store logs
    #     log = {
    #         'w_mat': w_mat.detach(),
    #         'err_norm': err_norm.detach(),
    #         'err_proj': err_proj.detach(),
    #         'w_norms': w_norms.detach(),
    #         'reconst_err': reconst_err_combined.detach(),
    #         'second_moment_mse': second_moment_mse.detach(),
    #         'objective': objective.detach(),
    #         'mc_reconst_errs': torch.stack(reconst_errs).detach(),  # Store individual MC errors
    #         'pred_specs': torch.stack(pred_specs).detach()  # Store all MC predictions
    #     }
    #
    #     return reconst_err_combined, objective, log

    # def base_step(self, batch):
    #     """
    #     base step function for training the nppc for the inpainting audio task with MC Dropout
    #     """
    #     # Preprocess data
    #     masked_spec, mask, clean_spec = batch
    #     clean_spec_mag_norm_log, mask, masked_spec_mag_log = utils.preprocess_data(clean_spec, masked_spec, mask)
    #
    #     # Get w_mat with restoration model in eval mode
    #     self.nppc_model.pretrained_restoration_model.eval()
    #     w_mat = self.nppc_model(masked_spec_mag_log, mask)  # [B,n_dirs,F,T]
    #     n_dirs = w_mat.shape[1]
    #
    #     # Enable dropout for MC sampling
    #     self.nppc_model.pretrained_restoration_model.train()
    #
    #     # Collect exactly n_dirs MC samples
    #     pred_specs = []
    #     for _ in range(n_dirs):  # n_mc_samples = n_dirs
    #         pred_spec = self.nppc_model.get_pred_spec_mag_norm(masked_spec_mag_log, mask)
    #         pred_specs.append(pred_spec)
    #
    #     # Process w_mat
    #     w_mat_ = w_mat.flatten(2)
    #     w_norms = w_mat_.norm(dim=2) + 1e-6
    #     w_hat_mat = w_mat_ / w_norms[:, :, None]
    #
    #     # Calculate errors for all MC samples
    #     errors = []
    #     for pred_spec in pred_specs:
    #         err = (clean_spec_mag_norm_log - pred_spec).flatten(1)
    #         err_norm = err.norm(dim=1) + 1e-6
    #         err = err / err_norm[:, None]
    #         errors.append(err)
    #
    #     # Calculate projection matrix
    #     batch_size = w_mat.shape[0]
    #     projection_matrix = torch.zeros(n_dirs, n_dirs, batch_size)  # [n_dirs, n_dirs, B]
    #
    #     # Calculate all projections
    #     for i, err in enumerate(errors):
    #         for j in range(n_dirs):
    #             w_dir = w_hat_mat[:, j:j + 1, :]  # [B, 1, F*T]
    #             err_proj = torch.einsum('bki,bi->bk', w_dir, err)  # [B, 1]
    #             projection_cost = 1 - err_proj.pow(2).sum(dim=1)
    #             projection_matrix[i, j] = projection_cost
    #
    #     # Find optimal assignment for each batch item
    #     batch_reconst_errs = []
    #     for b in range(batch_size):
    #         cost_matrix = projection_matrix[:, :, b].detach().cpu().numpy()
    #         row_ind, col_ind = linear_sum_assignment(cost_matrix)
    #         min_cost = cost_matrix[row_ind, col_ind].mean()
    #         batch_reconst_errs.append(min_cost)
    #
    #     reconst_err_combined = torch.tensor(batch_reconst_errs, device=w_mat.device)
    #
    #     # Second moment loss remains the same
    #     err_projs = torch.einsum('bki,bi->bk', w_hat_mat, errors[0])  # Using first error for simplicity
    #     second_moment_mse = (w_norms.pow(2) - err_projs.detach().pow(2)).pow(2)
    #
    #     # Compute final objective
    #     objective = self._calculate_final_objective(reconst_err_combined, second_moment_mse)
    #
    #     # Store logs
    #     log = {
    #         'w_mat': w_mat.detach(),
    #         'reconst_err': reconst_err_combined.detach(),
    #         'second_moment_mse': second_moment_mse.detach(),
    #         'objective': objective.detach(),
    #         'pred_specs': torch.stack(pred_specs).detach(),
    #         'projection_matrix': projection_matrix.detach()  # Useful for debugging
    #     }
    #
    #     return reconst_err_combined, objective, log

    # def base_step(self, batch):
    #     """
    #     base step function for training the nppc for the inpainting audio task with MC Dropout
    #     """
    #     # Preprocess data
    #     masked_spec, mask, clean_spec = batch
    #     clean_spec_mag_norm_log, mask, masked_spec_mag_log = utils.preprocess_data(clean_spec, masked_spec, mask)
    #
    #     # Get w_mat with restoration model in eval mode
    #     self.nppc_model.pretrained_restoration_model.eval()
    #     w_mat = self.nppc_model(masked_spec_mag_log, mask)  # [B,n_dirs,F,T]
    #     n_dirs = w_mat.shape[1]
    #
    #     # Enable dropout for MC sampling
    #     self.nppc_model.pretrained_restoration_model.train()
    #
    #     # Collect exactly n_dirs MC samples
    #     pred_specs = []
    #     for _ in range(n_dirs):  # n_mc_samples = n_dirs
    #         pred_spec = self.nppc_model.get_pred_spec_mag_norm(masked_spec_mag_log, mask)
    #         pred_specs.append(pred_spec)
    #
    #     # Process w_mat
    #     w_mat_ = w_mat.flatten(2)
    #     w_norms = w_mat_.norm(dim=2) + 1e-6
    #     w_hat_mat = w_mat_ / w_norms[:, :, None]
    #
    #     # Calculate errors for all MC samples
    #     errors = []
    #     for pred_spec in pred_specs:
    #         err = (clean_spec_mag_norm_log - pred_spec).flatten(1)
    #         err_norm = err.norm(dim=1) + 1e-6
    #         err = err / err_norm[:, None]
    #         errors.append(err)
    #
    #     # Option A: Match each direction with corresponding error
    #     reconst_errs = []
    #     for i in range(n_dirs):
    #         # Get i-th direction and i-th error
    #         w_dir = w_hat_mat[:, i:i + 1, :]  # [B, 1, F*T]
    #         err = errors[i]  # Take i-th error
    #
    #         # Project this error onto this direction
    #         err_proj = torch.einsum('bki,bi->bk', w_dir, err)  # [B, 1]
    #         reconst_err = 1 - err_proj.pow(2).sum(dim=1)
    #         reconst_errs.append(reconst_err)
    #
    #     # Combine errors from all direction-error pairs
    #     reconst_err_combined = torch.stack(reconst_errs).mean(dim=0)  # Average across directions
    #
    #     # Second moment loss using all matched pairs
    #     second_moment_mses = []
    #     for i in range(n_dirs):
    #         err_proj = torch.einsum('bki,bi->bk', w_hat_mat[:, i:i + 1, :], errors[i])
    #         second_moment_mse = (w_norms[:, i].pow(2) - err_proj.detach().pow(2)).pow(2)
    #         second_moment_mses.append(second_moment_mse)
    #
    #     second_moment_mse = torch.stack(second_moment_mses).mean(dim=0)
    #
    #     # Compute final objective
    #     objective = self._calculate_final_objective(reconst_err_combined, second_moment_mse)
    #
    #     # Store logs
    #     log = {
    #         'w_mat': w_mat.detach(),
    #         'reconst_err': reconst_err_combined.detach(),
    #         'second_moment_mse': second_moment_mse.detach(),
    #         'objective': objective.detach(),
    #         'pred_specs': torch.stack(pred_specs).detach(),
    #         'individual_reconst_errs': torch.stack(reconst_errs).detach()  # For debugging
    #     }
    #
    #     return reconst_err_combined, objective, log

    def save_checkpoint(self, checkpoint_path):
        """
        Save model checkpoint including model state, optimizer state, and training info

        Args:
            checkpoint_path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.nppc_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
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
                # Move batch to device
                masked_spec, mask, clean_spec = [x.to(self.device) for x in batch]

                # Get loss for this batch
                reconst_err, objective, _ = self.base_step((masked_spec, mask, clean_spec))
                val_losses.append(objective.item())
                val_reconst_err.append(reconst_err.mean().item())  # Calculate average validation loss
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_reconst_err = sum(val_reconst_err) / len(val_reconst_err)
        self.nppc_model.train()  # Set back to training mode
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
