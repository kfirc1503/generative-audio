import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb

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
from dataset.audio_dataset_inpainting import AudioInpaintingDataset, AudioInpaintingConfig

import utils
from nppc_audio.trainer import NPPCAudioTrainer
from utils import OptimizerConfig, DataLoaderConfig


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
            pin_memory=config.dataloader_configuration.pin_memory
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
            masked_spec, mask, clean_spec, _ = [x.to(self.device) for x in batch]
            batch = (masked_spec, mask, clean_spec)

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
        # w_norms_cpu = w_norms.detach().cpu()
        # err_proj_cpu = err_proj.detach().cpu()
        w_mat_cpu = w_mat.detach().cpu().numpy()
        second_moment_mse = (w_norms.pow(2) - err_proj.detach().pow(2)).pow(2)
        # second_moment_mse_cpu = second_moment_mse.mean().detach().cpu()
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
